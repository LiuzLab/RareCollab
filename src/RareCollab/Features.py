
"""
RareCollab feature generation pipeline.

Replaces the upstream AI-MARRVEL Nextflow workflow with a single-notebook
Python pipeline. Takes raw VCFs + HPO files, produces variant feature
matrices ready for the diagnostic engine.

Public functions:
    ProcessVCF(samplesheet, work_dir, references, fasta_references, ...)
        VCF preprocessing pipeline (BGZIP -> gVCF conversion -> filter -> ...).
    
    GenerateFeatures(samplesheet, work_dir, references, fasta_references,
                     singularity_images, ref_ver, config=None, ...)
        Feature generation pipeline (VEP -> HPO sim -> tier -> merge -> VTG).
"""

import os
import re
import sys
import json
import gzip
import shutil
import time
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm

# Relative imports from internal _lib subpackage
from ._lib._cpu import get_available_cpus as _get_available_cpus
from ._lib import _rna_worker as _rna
from ._lib._hpo_sim_worker import run_hpo_sim_parallel
from ._lib._phrank_worker import run_phrank_parallel
from ._lib._tier_worker import run_annotate_tier_parallel
from ._lib._join_phrank_worker import run_join_phrank_parallel
from ._lib._merge_worker import run_merge_parallel


def ENSURE_BGZIP_VCF(vcf_path, sample_id, work_dir, overwrite=False):
    vcf_path = Path(vcf_path)
    work_dir = Path(work_dir)

    sample_dir = work_dir / "process_vcf" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_vcf = sample_dir / f"{sample_id}.vcf.gz"
    output_tbi = sample_dir / f"{sample_id}.vcf.gz.tbi"

    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found for sample {sample_id}: {vcf_path}")

    if overwrite:
        for p in [output_vcf, output_tbi]:
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        if output_vcf.exists() and output_tbi.exists():
            #print(f"[SKIP] {sample_id}: standardized VCF and TBI already exist.")
            return output_vcf, output_tbi
    
    file_type_result = subprocess.run(["file", "-bL", str(vcf_path)], check=True, text=True, capture_output=True)
    file_type = file_type_result.stdout.strip()

    # ---- Case 1: original VCF is already BGZF ----
    if "BGZF" in file_type:
        #print(f"[INFO] {sample_id}: VCF is already BGZF. Creating symlink.")
        os.symlink(vcf_path.resolve(), output_vcf)
        original_tbi = Path(str(vcf_path) + ".tbi")
        if original_tbi.exists():
            #print(f"[INFO] {sample_id}: original TBI exists. Creating TBI symlink.")
            os.symlink(original_tbi.resolve(), output_tbi)
            action = "symlink_bgzf_vcf_and_tbi"
        else:
            #print(f"[INFO] {sample_id}: original TBI does not exist. Creating new TBI.")
            subprocess.run(["tabix", "-f", "-p", "vcf", str(output_vcf)],check=True)
            action = "symlink_bgzf_vcf_and_create_tbi"
    # ---- Case 2: original VCF is regular gzip but not BGZF ----
    elif "gzip compressed data" in file_type:
        #print(f"[INFO] {sample_id}: VCF is regular gzip. Recompressing to BGZF.")
        with open(output_vcf, "wb") as out:
            gunzip_process = subprocess.Popen(["gunzip", "-c", str(vcf_path)], stdout=subprocess.PIPE)

            subprocess.run(["bgzip"], stdin=gunzip_process.stdout, stdout=out, check=True)
            if gunzip_process.stdout is not None:
                gunzip_process.stdout.close()
            gunzip_returncode = gunzip_process.wait()
        if gunzip_returncode != 0:
            raise RuntimeError(f"gunzip failed for sample {sample_id}: {vcf_path}")
        subprocess.run(["tabix", "-f", "-p", "vcf", str(output_vcf)], check=True)
        action = "recompress_gzip_to_bgzip_and_create_tbi"
    # ---- Case 3: original VCF is plain text ----
    elif ("ASCII text" in file_type or "Unicode text" in file_type or "Variant Call Format" in file_type):
        #print(f"[INFO] {sample_id}: VCF is plain text. Compressing to BGZF.")
        with open(output_vcf, "wb") as out:
            subprocess.run(["bgzip", "-c", str(vcf_path)], stdout=out, check=True)

        subprocess.run(["tabix", "-f", "-p", "vcf", str(output_vcf)], check=True)
        action = "compress_plain_vcf_to_bgzip_and_create_tbi"
    # ---- Unknown format ----
    else:
        raise ValueError(
            f"Unrecognized VCF format for sample {sample_id}: {vcf_path}\n"
            f"Detected file type: {file_type}")

    # ---- Final checks ----
    if not output_vcf.exists():
        raise RuntimeError(f"Output VCF was not created for sample {sample_id}: {output_vcf}")
    if not output_tbi.exists():
        raise RuntimeError(f"Output TBI was not created for sample {sample_id}: {output_tbi}")
    
    # ---- Metadata for debugging ----
    metadata = {"sample_id": sample_id,
                "original_vcf_path": str(vcf_path.resolve()),
                "bgzip_vcf_path": str(output_vcf.resolve()),
                "bgzip_tbi_path": str(output_tbi.resolve()),
                "file_type": file_type,
                "action": action}

    metadata_path = sample_dir / "ENSURE_BGZIP_VCF.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    #print(f"[DONE] {sample_id}: {action}")

    return output_vcf, output_tbi

def CONVERT_GVCF(vcf_path, sample_id, work_dir, fasta_path, chrmap_file, overwrite=False):
    vcf_path = Path(vcf_path)
    work_dir = Path(work_dir)
    fasta_path = Path(fasta_path)
    chrmap_file = Path(chrmap_file)

    input_tbi = Path(str(vcf_path) + ".tbi")
    if not input_tbi.exists():
        raise FileNotFoundError(f"Input VCF index .tbi not found for sample {sample_id}: {input_tbi}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {fasta_path}")
    if not chrmap_file.exists():
        raise FileNotFoundError(f"chrmap_file not found: {chrmap_file}")
    
    # Keep all VCF preprocessing outputs in the same sample folder
    sample_dir = work_dir / "process_vcf" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_vcf = sample_dir / f"{sample_id}.nog.vcf.gz"
    output_tbi = sample_dir / f"{sample_id}.nog.vcf.gz.tbi"
    
    if overwrite:
        for p in sample_dir.glob("step*.vcf.gz*"):
            if p.exists() or p.is_symlink():
                p.unlink()
        for p in [output_vcf, output_tbi]:
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        if output_vcf.exists() and output_tbi.exists():
            return output_vcf, output_tbi
    
    # Check whether this is gVCF
    is_gvcf = False
    with gzip.open(vcf_path, "rt", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break
            if "<NON_REF>" in line:
                is_gvcf = True
                break
    if not is_gvcf:
        os.symlink(vcf_path.resolve(), output_vcf)
        os.symlink(input_tbi.resolve(), output_tbi)
        action = "not_gvcf_symlink_vcf_and_tbi"   
    else:
        step1 = sample_dir / "step1.vcf.gz"
        step1_1 = sample_dir / "step1_1.vcf.gz"
        step2 = sample_dir / "step2.vcf.gz"
        step3 = sample_dir / "step3.vcf.gz"
        step4 = sample_dir / "step4.vcf.gz"
        
        subprocess.run(["bcftools", "annotate", "--rename-chrs", str(chrmap_file), "-x", "ID", str(vcf_path), "-Oz", "-o", str(step1)], check=True)
        subprocess.run(["tabix", "-f", "-p", "vcf", str(step1)], check=True)
        standard_chrs = ",".join([str(i) for i in range(1, 23)] + ["X", "Y"])
        subprocess.run(["bcftools", "view", "-r", standard_chrs, str(step1), "-Oz", "-o", str(step1_1)], check=True)
        subprocess.run(["bcftools", "sort", str(step1_1), "-Oz", "-o", str(step2)], check=True)
        subprocess.run(["tabix", "-f", "-p", "vcf", str(step2)], check=True)
        subprocess.run(["gatk", "GenotypeGVCFs", "-R", str(fasta_path), "-V", str(step2), "-O", str(step3), "--allow-old-rms-mapping-quality-annotation-data"], check=True)
        subprocess.run(["gatk", "VariantFiltration", "-V", str(step3), "-O", str(step4), "--filter-expression", "QUAL < 30.0", "--filter-name", "LowQual"], check=True)
        shutil.move(str(step4), str(output_vcf))
        subprocess.run(["tabix", "-f", "-p", "vcf", str(output_vcf)], check=True)

        for p in sample_dir.glob("step*.vcf.gz*"):
            if p.exists() or p.is_symlink():
                p.unlink()
        action = "convert_gvcf_to_regular_vcf"

    #Final Check:
    if not output_vcf.exists():
        raise RuntimeError(f"CONVERT_GVCF output VCF was not created for sample {sample_id}: {output_vcf}")
    if not output_tbi.exists():
        raise RuntimeError(f"CONVERT_GVCF output TBI was not created for sample {sample_id}: {output_tbi}")

    metadata = {
        "sample_id": sample_id,
        "input_vcf_path": str(vcf_path.resolve()),
        "input_tbi_path": str(input_tbi.resolve()),
        "output_vcf_path": str(output_vcf.resolve()),
        "output_tbi_path": str(output_tbi.resolve()),
        "is_gvcf": is_gvcf,
        "action": action,
        "fasta_path": str(fasta_path.resolve()),
        "chrmap_file": str(chrmap_file.resolve()),
    }

    metadata_path = sample_dir / "CONVERT_GVCF.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return output_vcf, output_tbi

def NORMALIZE_CHR_AND_ID(vcf_path, sample_id, work_dir, chrmap_file, overwrite=False):
    vcf_path = Path(vcf_path)
    work_dir = Path(work_dir)
    chrmap_file = Path(chrmap_file)

    if not vcf_path.exists():
        raise FileNotFoundError(f"Input VCF not found for sample {sample_id}: {vcf_path}")

    input_tbi = Path(str(vcf_path) + ".tbi")
    if not input_tbi.exists():
        raise FileNotFoundError(f"Input VCF index .tbi not found for sample {sample_id}: {input_tbi}")

    if not chrmap_file.exists():
        raise FileNotFoundError(f"chrmap_file not found: {chrmap_file}")

    sample_dir = work_dir / "process_vcf" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    tmp_vcf = sample_dir / f"{sample_id}.chrid.tmp.vcf.gz"
    output_vcf = sample_dir / f"{sample_id}.chrid.vcf.gz"
    output_tbi = sample_dir / f"{sample_id}.chrid.vcf.gz.tbi"

    if overwrite:
        for p in [tmp_vcf, Path(str(tmp_vcf) + ".tbi"), output_vcf, output_tbi]:
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        if output_vcf.exists() and output_tbi.exists():
            return output_vcf, output_tbi

    subprocess.run(["bcftools", "annotate", "--rename-chrs", str(chrmap_file), "-x", "ID", str(vcf_path), "-Oz", "-o", str(tmp_vcf)], check=True)
    subprocess.run(["bcftools", "annotate", "--set-id", r"+%CHROM\_%POS\_%REF\_%FIRST_ALT", str(tmp_vcf), "-Oz", "-o", str(output_vcf)], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(output_vcf)], check=True)

    for p in [tmp_vcf, Path(str(tmp_vcf) + ".tbi")]:
        if p.exists() or p.is_symlink():
            p.unlink()

    if not output_vcf.exists():
        raise RuntimeError(f"NORMALIZE_CHR_AND_ID output VCF was not created for sample {sample_id}: {output_vcf}")

    if not output_tbi.exists():
        raise RuntimeError(f"NORMALIZE_CHR_AND_ID output TBI was not created for sample {sample_id}: {output_tbi}")

    metadata = {
        "sample_id": sample_id,
        "input_vcf_path": str(vcf_path.resolve()),
        "input_tbi_path": str(input_tbi.resolve()),
        "output_vcf_path": str(output_vcf.resolve()),
        "output_tbi_path": str(output_tbi.resolve()),
        "chrmap_file": str(chrmap_file.resolve()),
        "action": "normalize_chromosome_and_variant_id",
    }

    metadata_path = sample_dir / "NORMALIZE_CHR_AND_ID.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return output_vcf, output_tbi

def FILTER_UNPASSED(vcf_path, sample_id, work_dir, overwrite=False):
    vcf_path = Path(vcf_path)
    work_dir = Path(work_dir)

    if not vcf_path.exists():
        raise FileNotFoundError(f"Input VCF not found for sample {sample_id}: {vcf_path}")

    input_tbi = Path(str(vcf_path) + ".tbi")
    if not input_tbi.exists():
        raise FileNotFoundError(f"Input VCF index .tbi not found for sample {sample_id}: {input_tbi}")

    sample_dir = work_dir / "process_vcf" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_vcf = sample_dir / f"{sample_id}.filt.vcf.gz"
    output_tbi = sample_dir / f"{sample_id}.filt.vcf.gz.tbi"

    if overwrite:
        for p in [output_vcf, output_tbi]:
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        if output_vcf.exists() and output_tbi.exists():
            return output_vcf, output_tbi

    subprocess.run(["bcftools", "filter", str(vcf_path), "-i", 'FILTER == "PASS"', "-Oz", "-o", str(output_vcf)], check=True)

    count_result = subprocess.run(
        f"bcftools view -H {output_vcf} | wc -l",
        shell=True,
        check=True,
        text=True,
        capture_output=True,
    )
    variant_count = int(count_result.stdout.strip())

    if variant_count == 0:
        print(f"[INFO] {sample_id}: no variants passed FILTER == PASS. Proceeding with unfiltered VCF.")
        if output_vcf.exists() or output_vcf.is_symlink():
            output_vcf.unlink()
        os.symlink(vcf_path.resolve(), output_vcf)

    subprocess.run(["tabix", "-f", "-p", "vcf", str(output_vcf)], check=True)

    if not output_vcf.exists():
        raise RuntimeError(f"FILTER_UNPASSED output VCF was not created for sample {sample_id}: {output_vcf}")
    if not output_tbi.exists():
        raise RuntimeError(f"FILTER_UNPASSED output TBI was not created for sample {sample_id}: {output_tbi}")

    metadata = {
        "sample_id": sample_id,
        "input_vcf_path": str(vcf_path.resolve()),
        "input_tbi_path": str(input_tbi.resolve()),
        "output_vcf_path": str(output_vcf.resolve()),
        "output_tbi_path": str(output_tbi.resolve()),
        "variant_count_after_filter_pass": variant_count,
        "action": "filter_pass_variants" if variant_count > 0 else "no_pass_variants_symlink_unfiltered_vcf",
    }

    metadata_path = sample_dir / "FILTER_UNPASSED.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return output_vcf, output_tbi

def FILTER_NONREF_VARIANTS(vcf_path, sample_id, work_dir, overwrite=False):
    vcf_path = Path(vcf_path)
    work_dir = Path(work_dir)

    if not vcf_path.exists():
        raise FileNotFoundError(f"Input VCF not found for sample {sample_id}: {vcf_path}")

    input_tbi = Path(str(vcf_path) + ".tbi")
    if not input_tbi.exists():
        raise FileNotFoundError(f"Input VCF index .tbi not found for sample {sample_id}: {input_tbi}")

    sample_dir = work_dir / "process_vcf" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_vcf = sample_dir / f"{sample_id}.filt.nonref.vcf.gz"
    output_tbi = sample_dir / f"{sample_id}.filt.nonref.vcf.gz.tbi"

    if overwrite:
        for p in [output_vcf, output_tbi]:
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        if output_vcf.exists() and output_tbi.exists():
            return output_vcf, output_tbi

    subprocess.run(["bcftools", "view", "-i", 'COUNT(FMT/GT!="0/0")>0', str(vcf_path), "-Oz", "-o", str(output_vcf)], check=True)

    count_result = subprocess.run(f"bcftools view -H {output_vcf} | wc -l", shell=True, check=True, text=True, capture_output=True)
    variant_count = int(count_result.stdout.strip())

    #print(f"[INFO] {sample_id}: Removed homozygous reference records. Variants remaining: {variant_count}")

    subprocess.run(["tabix", "-f", "-p", "vcf", str(output_vcf)], check=True)

    if not output_vcf.exists():
        raise RuntimeError(f"FILTER_NONREF_VARIANTS output VCF was not created for sample {sample_id}: {output_vcf}")
    if not output_tbi.exists():
        raise RuntimeError(f"FILTER_NONREF_VARIANTS output TBI was not created for sample {sample_id}: {output_tbi}")

    metadata = {
        "sample_id": sample_id,
        "input_vcf_path": str(vcf_path.resolve()),
        "input_tbi_path": str(input_tbi.resolve()),
        "output_vcf_path": str(output_vcf.resolve()),
        "output_tbi_path": str(output_tbi.resolve()),
        "variant_count_after_filter_nonref": variant_count,
        "action": "filter_non_homozygous_reference_variants",
    }

    metadata_path = sample_dir / "FILTER_NONREF_VARIANTS.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return output_vcf, output_tbi

def FILTER_MITO_AND_UNKNOWN_CHR(vcf_path, sample_id, work_dir, overwrite=False):
    vcf_path = Path(vcf_path)
    work_dir = Path(work_dir)

    if not vcf_path.exists():
        raise FileNotFoundError(f"Input VCF not found for sample {sample_id}: {vcf_path}")

    input_tbi = Path(str(vcf_path) + ".tbi")
    if not input_tbi.exists():
        raise FileNotFoundError(f"Input VCF index .tbi not found for sample {sample_id}: {input_tbi}")

    sample_dir = work_dir / "process_vcf" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    tmp_vcf = sample_dir / f"{sample_id}.filt.rmMT.vcf"
    output_vcf = sample_dir / f"{sample_id}.filt.rmMT.vcf.gz"
    output_tbi = sample_dir / f"{sample_id}.filt.rmMT.vcf.gz.tbi"

    if overwrite:
        for p in [tmp_vcf, output_vcf, output_tbi]:
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        if output_vcf.exists() and output_tbi.exists():
            return output_vcf, output_tbi
        # Clean orphans from previous crashed run
        for p in [tmp_vcf, output_vcf, output_tbi]:
            if p.exists() or p.is_symlink():
                p.unlink()

    standard_chrs = ",".join([str(i) for i in range(1, 23)] + ["X", "Y"])

    subprocess.run(["bcftools", "view", "-r", standard_chrs, str(vcf_path), "-o", str(tmp_vcf)], check=True)
    subprocess.run(["bgzip", str(tmp_vcf)], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(output_vcf)], check=True)

    if not output_vcf.exists():
        raise RuntimeError(f"FILTER_MITO_AND_UNKNOWN_CHR output VCF was not created for sample {sample_id}: {output_vcf}")
    if not output_tbi.exists():
        raise RuntimeError(f"FILTER_MITO_AND_UNKNOWN_CHR output TBI was not created for sample {sample_id}: {output_tbi}")

    count_result = subprocess.run(f"bcftools view -H {output_vcf} | wc -l", shell=True, check=True, text=True, capture_output=True)
    variant_count = int(count_result.stdout.strip())

    metadata = {
        "sample_id": sample_id,
        "input_vcf_path": str(vcf_path.resolve()),
        "input_tbi_path": str(input_tbi.resolve()),
        "output_vcf_path": str(output_vcf.resolve()),
        "output_tbi_path": str(output_tbi.resolve()),
        "variant_count_after_filter_mito_unknown_chr": variant_count,
        "action": "filter_mitochondrial_and_unknown_chromosomes",
    }

    metadata_path = sample_dir / "FILTER_MITO_AND_UNKNOWN_CHR.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return output_vcf, output_tbi

def FILTER_BLACKLIST_VARIANTS(vcf_path, sample_id, work_dir, gnomad_refs, overwrite=False):
    vcf_path = Path(vcf_path)
    work_dir = Path(work_dir)

    if not vcf_path.exists():
        raise FileNotFoundError(f"Input VCF not found for sample {sample_id}: {vcf_path}")

    input_tbi = Path(str(vcf_path) + ".tbi")
    if not input_tbi.exists():
        raise FileNotFoundError(f"Input VCF index .tbi not found for sample {sample_id}: {input_tbi}")

    if not gnomad_refs.genome_vcf.exists():
        raise FileNotFoundError(f"gnomAD genome blacklist VCF not found: {gnomad_refs.genome_vcf}")
    if not gnomad_refs.genome_tbi.exists():
        raise FileNotFoundError(f"gnomAD genome blacklist TBI not found: {gnomad_refs.genome_tbi}")
    if not gnomad_refs.exome_vcf.exists():
        raise FileNotFoundError(f"gnomAD exome blacklist VCF not found: {gnomad_refs.exome_vcf}")
    if not gnomad_refs.exome_tbi.exists():
        raise FileNotFoundError(f"gnomAD exome blacklist TBI not found: {gnomad_refs.exome_tbi}")

    sample_dir = work_dir / "process_vcf" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    isec_tmp1 = sample_dir / "isec_tmp1"
    isec_tmp2 = sample_dir / "isec_tmp2"
    isec_tmp3 = sample_dir / "isec_tmp3"

    output_vcf_uncompressed = sample_dir / f"{sample_id}.filt.rmBL.vcf"
    output_vcf = sample_dir / f"{sample_id}.filt.rmBL.vcf.gz"
    output_tbi = sample_dir / f"{sample_id}.filt.rmBL.vcf.gz.tbi"

    if overwrite:
        for d in [isec_tmp1, isec_tmp2, isec_tmp3]:
            if d.exists():
                shutil.rmtree(d)
        for p in [output_vcf_uncompressed, output_vcf, output_tbi]:
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        if output_vcf.exists() and output_tbi.exists():
            return output_vcf, output_tbi

    isec_tmp1.mkdir(mode=0o777, parents=True, exist_ok=True)
    subprocess.run(["bcftools", "isec", "-p", str(isec_tmp1), "-w", "1", "-Oz", str(vcf_path), str(gnomad_refs.genome_vcf)], check=True)

    isec_tmp2.mkdir(mode=0o777, parents=True, exist_ok=True)
    subprocess.run(["bcftools", "isec", "-p", str(isec_tmp2), "-w", "1", "-Oz", str(vcf_path), str(gnomad_refs.exome_vcf)], check=True)

    isec_tmp3.mkdir(mode=0o777, parents=True, exist_ok=True)
    subprocess.run(["bcftools", "isec", "-p", str(isec_tmp3), "-Ov", str(isec_tmp1 / "0000.vcf.gz"), str(isec_tmp2 / "0000.vcf.gz")], check=True)

    isec_result = isec_tmp3 / "0002.vcf"
    if not isec_result.exists():
        raise RuntimeError(f"Expected bcftools isec output was not created for sample {sample_id}: {isec_result}")

    isec_result.rename(output_vcf_uncompressed)

    subprocess.run(["bgzip", str(output_vcf_uncompressed)], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(output_vcf)], check=True)

    if not output_vcf.exists():
        raise RuntimeError(f"FILTER_BLACKLIST_VARIANTS output VCF was not created for sample {sample_id}: {output_vcf}")
    if not output_tbi.exists():
        raise RuntimeError(f"FILTER_BLACKLIST_VARIANTS output TBI was not created for sample {sample_id}: {output_tbi}")

    count_result = subprocess.run(f"bcftools view -H {output_vcf} | wc -l", shell=True, check=True, text=True, capture_output=True)
    variant_count = int(count_result.stdout.strip())

    metadata = {
        "sample_id": sample_id,
        "input_vcf_path": str(vcf_path.resolve()),
        "input_tbi_path": str(input_tbi.resolve()),
        "gnomad_genome_vcf": str(gnomad_refs.genome_vcf.resolve()),
        "gnomad_genome_tbi": str(gnomad_refs.genome_tbi.resolve()),
        "gnomad_exome_vcf": str(gnomad_refs.exome_vcf.resolve()),
        "gnomad_exome_tbi": str(gnomad_refs.exome_tbi.resolve()),
        "output_vcf_path": str(output_vcf.resolve()),
        "output_tbi_path": str(output_tbi.resolve()),
        "variant_count_after_filter_blacklist": variant_count,
        "action": "filter_blacklist_variants_with_gnomad_genome_and_exome",
    }

    metadata_path = sample_dir / "FILTER_BLACKLIST_VARIANTS.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    for d in [isec_tmp1, isec_tmp2, isec_tmp3]:
        if d.exists():
            shutil.rmtree(d)

    return output_vcf, output_tbi

def ENSURE_VCF_NON_EMPTY(vcf_path, sample_id, work_dir, overwrite=False):
    vcf_path = Path(vcf_path)
    work_dir = Path(work_dir)

    if not vcf_path.exists():
        raise FileNotFoundError(f"Input VCF not found for sample {sample_id}: {vcf_path}")

    input_tbi = Path(str(vcf_path) + ".tbi")
    if not input_tbi.exists():
        raise FileNotFoundError(f"Input VCF index .tbi not found for sample {sample_id}: {input_tbi}")

    sample_dir = work_dir / "process_vcf" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    non_empty_flag = sample_dir / f"{sample_id}.vcf_non_empty.txt"

    if overwrite:
        if non_empty_flag.exists() or non_empty_flag.is_symlink():
            non_empty_flag.unlink()
    else:
        if non_empty_flag.exists():
            return vcf_path

    count_result = subprocess.run(f"bcftools view -H {vcf_path} | wc -l", shell=True, check=True, text=True, capture_output=True)
    variant_count = int(count_result.stdout.strip())

    if variant_count > 0:
        with open(non_empty_flag, "w") as f:
            f.write(f"{variant_count}\n")
        #print(f"[INFO] {sample_id}: Final pre-processed VCF has {variant_count} variants.")
    else:
        raise RuntimeError(
            f"All variants contained only GT=0/0 or were filtered for sample {sample_id}. "
            f"Nothing left to process. Provide a VCF with non-reference calls or relax upstream filtering thresholds."
        )

    metadata = {
        "sample_id": sample_id,
        "input_vcf_path": str(vcf_path.resolve()),
        "input_tbi_path": str(input_tbi.resolve()),
        "variant_count_final_preprocessed_vcf": variant_count,
        "non_empty_flag": str(non_empty_flag.resolve()),
        "action": "ensure_vcf_non_empty",
    }

    metadata_path = sample_dir / "ENSURE_VCF_NON_EMPTY.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return vcf_path

def ProcessVCF(samplesheet, max_workers, work_dir, references, fasta_references, overwrite=False, keep_intermediate=True):
    #1.Make sure vcf is BGZIP and has .tbi
    print('1/8. Checking and Converting files into BGZIP ...')
    tasks = []
    ok = fail = 0
    results = {}

    with ThreadPoolExecutor(max_workers = max_workers) as ex:
        for row in samplesheet.itertuples(index=True):
            future = ex.submit(ENSURE_BGZIP_VCF, row.vcf_path, row.sampleID, work_dir, overwrite)
            tasks.append({'index':row.Index,"sampleID": row.sampleID,"vcf_path": row.vcf_path,'future':future})    
        with tqdm(total=len(tasks), desc="ENSURE BGZIP VCF") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    output_vcf, output_tbi = future.result()
                    results[row_index] = {"bgzip_vcf_path": str(output_vcf), "bgzip_tbi_path": str(output_tbi)}
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] ENSURE_BGZIP_VCF failed for sample {sample_id}")
                    print(f"[ERROR] Original VCF: {task['vcf_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    processed_samplesheet = samplesheet.copy()
    for row_index, result in results.items():
        processed_samplesheet.loc[row_index, "bgzip_vcf_path"] = result["bgzip_vcf_path"]
    
    if fail > 0:
        raise RuntimeError(f"ENSURE_BGZIP_VCF failed for {fail} sample(s).")

    #2. Make sure it's not GVCF:
    print('2/8. Checking and Converting .GVCF into .VCF ...')
    tasks = []
    results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in processed_samplesheet.itertuples(index=True):
            future = ex.submit(CONVERT_GVCF, row.bgzip_vcf_path, row.sampleID, work_dir, fasta_references.fasta, references.chrmap_file, overwrite)
            tasks.append({"index": row.Index, "sampleID": row.sampleID, "vcf_path": row.bgzip_vcf_path, "future": future,})

        with tqdm(total=len(tasks), desc="CONVERT GVCF") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    output_vcf, output_tbi = future.result()
                    results[row_index] = {"nog_vcf_path": str(output_vcf), "nog_tbi_path": str(output_tbi)}
                    ok += 1

                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] CONVERT_GVCF failed for sample {sample_id}")
                    print(f"[ERROR] Input VCF: {task['vcf_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    processed_samplesheet["nog_vcf_path"] = None
    for row_index, result in results.items():
        processed_samplesheet.loc[row_index, "nog_vcf_path"] = result["nog_vcf_path"]
    
    if fail > 0:
        raise RuntimeError(f"CONVERT_GVCF failed for {fail} sample(s).")
    
    # 3. Normalize chromosome names and variant IDs
    print('3/8. Normalizing chromosome names and variant IDs ...')
    tasks = []
    results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in processed_samplesheet.itertuples(index=True):
            future = ex.submit(NORMALIZE_CHR_AND_ID, row.nog_vcf_path, row.sampleID, work_dir, references.chrmap_file, overwrite)
            tasks.append({"index": row.Index, "sampleID": row.sampleID, "vcf_path": row.nog_vcf_path, "future": future})

        with tqdm(total=len(tasks), desc="NORMALIZE CHR AND ID") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    output_vcf, output_tbi = future.result()
                    results[row_index] = {"chrid_vcf_path": str(output_vcf), "chrid_tbi_path": str(output_tbi)}
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] NORMALIZE_CHR_AND_ID failed for sample {sample_id}")
                    print(f"[ERROR] Input VCF: {task['vcf_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    processed_samplesheet["chrid_vcf_path"] = None
    for row_index, result in results.items():
        processed_samplesheet.loc[row_index, "chrid_vcf_path"] = result["chrid_vcf_path"]

    if fail > 0:
        raise RuntimeError(f"NORMALIZE_CHR_AND_ID failed for {fail} sample(s).")

    # 4. Filter variants that passed quality filters
    print('4/8. Filtering variants passing quality filters ...')
    tasks = []
    results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in processed_samplesheet.itertuples(index=True):
            future = ex.submit(FILTER_UNPASSED, row.chrid_vcf_path, row.sampleID, work_dir, overwrite)
            tasks.append({"index": row.Index, "sampleID": row.sampleID, "vcf_path": row.chrid_vcf_path, "future": future})

        with tqdm(total=len(tasks), desc="FILTER UNPASSED") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    output_vcf, output_tbi = future.result()
                    results[row_index] = {"filt_vcf_path": str(output_vcf), "filt_tbi_path": str(output_tbi)}
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] FILTER_UNPASSED failed for sample {sample_id}")
                    print(f"[ERROR] Input VCF: {task['vcf_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    processed_samplesheet["filt_vcf_path"] = None
    for row_index, result in results.items():
        processed_samplesheet.loc[row_index, "filt_vcf_path"] = result["filt_vcf_path"]

    if fail > 0:
        raise RuntimeError(f"FILTER_UNPASSED failed for {fail} sample(s).")

    # 5. Filter out homozygous reference records
    print('5/8. Filtering out homozygous reference records ...')
    tasks = []
    results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in processed_samplesheet.itertuples(index=True):
            future = ex.submit(FILTER_NONREF_VARIANTS, row.filt_vcf_path, row.sampleID, work_dir, overwrite)
            tasks.append({"index": row.Index, "sampleID": row.sampleID, "vcf_path": row.filt_vcf_path, "future": future})

        with tqdm(total=len(tasks), desc="FILTER NONREF VARIANTS") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    output_vcf, output_tbi = future.result()
                    results[row_index] = {"filt_nonref_vcf_path": str(output_vcf)}
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] FILTER_NONREF_VARIANTS failed for sample {sample_id}")
                    print(f"[ERROR] Input VCF: {task['vcf_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    processed_samplesheet["filt_nonref_vcf_path"] = None

    for row_index, result in results.items():
        processed_samplesheet.loc[row_index, "filt_nonref_vcf_path"] = result["filt_nonref_vcf_path"]

    if fail > 0:
        raise RuntimeError(f"FILTER_NONREF_VARIANTS failed for {fail} sample(s).")

    # 6. Filter mitochondrial and unknown chromosomes
    print('6/8. Removing mitochondrial and unknown chromosomes ...')
    tasks = []
    results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in processed_samplesheet.itertuples(index=True):
            future = ex.submit(FILTER_MITO_AND_UNKNOWN_CHR, row.filt_nonref_vcf_path, row.sampleID, work_dir, overwrite)
            tasks.append({"index": row.Index, "sampleID": row.sampleID, "vcf_path": row.filt_nonref_vcf_path, "future": future})

        with tqdm(total=len(tasks), desc="FILTER MITO AND UNKNOWN CHR") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    output_vcf, output_tbi = future.result()
                    results[row_index] = {"filt_rmMT_vcf_path": str(output_vcf)}
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] FILTER_MITO_AND_UNKNOWN_CHR failed for sample {sample_id}")
                    print(f"[ERROR] Input VCF: {task['vcf_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    processed_samplesheet["filt_rmMT_vcf_path"] = None

    for row_index, result in results.items():
        processed_samplesheet.loc[row_index, "filt_rmMT_vcf_path"] = result["filt_rmMT_vcf_path"]

    if fail > 0:
        raise RuntimeError(f"FILTER_MITO_AND_UNKNOWN_CHR failed for {fail} sample(s).")
    
    # 7. Filter blacklist variants
    print('7/8. Filtering out Gnomad blacklist variants ...')
    tasks = []
    results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in processed_samplesheet.itertuples(index=True):
            future = ex.submit(FILTER_BLACKLIST_VARIANTS, row.filt_rmMT_vcf_path, row.sampleID, work_dir, references.gnomad, overwrite)
            tasks.append({"index": row.Index, "sampleID": row.sampleID, "vcf_path": row.filt_rmMT_vcf_path, "future": future})

        with tqdm(total=len(tasks), desc="FILTER BLACKLIST VARIANTS") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    output_vcf, output_tbi = future.result()
                    results[row_index] = {"filt_rmBL_vcf_path": str(output_vcf)}
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] FILTER_BLACKLIST_VARIANTS failed for sample {sample_id}")
                    print(f"[ERROR] Input VCF: {task['vcf_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    processed_samplesheet["filt_rmBL_vcf_path"] = None

    for row_index, result in results.items():
        processed_samplesheet.loc[row_index, "filt_rmBL_vcf_path"] = result["filt_rmBL_vcf_path"]

    if fail > 0:
        raise RuntimeError(f"FILTER_BLACKLIST_VARIANTS failed for {fail} sample(s).")

    # 8. Ensure final VCF is non-empty
    print('8/8 Checking final VCF is non-empty ...')
    tasks = []
    results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in processed_samplesheet.itertuples(index=True):
            future = ex.submit(ENSURE_VCF_NON_EMPTY, row.filt_rmBL_vcf_path, row.sampleID, work_dir, overwrite)
            tasks.append({"index": row.Index, "sampleID": row.sampleID, "vcf_path": row.filt_rmBL_vcf_path, "future": future})

        with tqdm(total=len(tasks), desc="ENSURE VCF NON EMPTY") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    output_vcf = future.result()
                    results[row_index] = {"final_vcf_path": str(output_vcf)}
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] ENSURE_VCF_NON_EMPTY failed for sample {sample_id}")
                    print(f"[ERROR] Input VCF: {task['vcf_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(ok=ok, fail=fail)

    processed_samplesheet["final_vcf_path"] = None

    for row_index, result in results.items():
        processed_samplesheet.loc[row_index, "final_vcf_path"] = result["final_vcf_path"]

    if fail > 0:
        raise RuntimeError(f"ENSURE_VCF_NON_EMPTY failed for {fail} sample(s).")
    
    if not keep_intermediate:
        print(f"Remove all intermediate results in {work_dir}/process_vcf/<sample_id>/")
        intermediate_cols = ["filt_vcf_path", "filt_nonref_vcf_path", "filt_rmMT_vcf_path"]

        for col in intermediate_cols:
            if col in processed_samplesheet.columns:
                for vcf_path in processed_samplesheet[col].dropna():
                    vcf_path = Path(vcf_path)
                    tbi_path = Path(str(vcf_path) + ".tbi")
                    for p in [vcf_path, tbi_path]:
                        if p.exists() or p.is_symlink():
                            p.unlink()

        processed_samplesheet = processed_samplesheet.drop(columns=[col for col in intermediate_cols if col in processed_samplesheet.columns])
        return processed_samplesheet
    else:
        print(f"Keep all intermediate results in {work_dir}/process_vcf/<sample_id>/")
        return processed_samplesheet


    
def SPLIT_VCF_BY_CHROMOSOME(vcf_path, sample_id, work_dir, overwrite=False):
    vcf_path = Path(vcf_path)
    work_dir = Path(work_dir)

    if not vcf_path.exists():
        raise FileNotFoundError(f"Input VCF not found for sample {sample_id}: {vcf_path}")

    input_tbi = Path(str(vcf_path) + ".tbi")
    if not input_tbi.exists():
        raise FileNotFoundError(f"Input VCF index .tbi not found for sample {sample_id}: {input_tbi}")

    sample_dir = work_dir / "features" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for p in sample_dir.glob("chr*.vcf.gz"):
            if p.exists() or p.is_symlink():
                p.unlink()
        for p in sample_dir.glob("chr*.vcf.gz.tbi"):
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        existing = sorted(sample_dir.glob("chr*.vcf.gz"))
        if existing:
            return existing

    # ---- Dynamically get the list of chromosomes actually present in the VCF ----
    # Necessary because upstream filters may have removed all variants on some chromosomes.
    chrom_result = subprocess.run(
        f"bcftools query -f '%CHROM\\n' {vcf_path} | sort -u",
        shell=True,
        check=True,
        text=True,
        capture_output=True,
    )
    chromosomes = [c.strip() for c in chrom_result.stdout.splitlines() if c.strip()]

    if not chromosomes:
        raise RuntimeError(
            f"SPLIT_VCF_BY_CHROMOSOME found no chromosomes in VCF for sample {sample_id}: {vcf_path}"
        )

    # ---- Split the VCF by chromosome ----
    output_vcfs = []
    for chrom in chromosomes:
        out_vcf = sample_dir / f"chr{chrom}.vcf.gz"
        subprocess.run(
            ["bcftools", "view", "-r", chrom, str(vcf_path), "-Oz", "-o", str(out_vcf)],
            check=True,
        )
        if not out_vcf.exists():
            raise RuntimeError(
                f"SPLIT_VCF_BY_CHROMOSOME failed to create {out_vcf} for sample {sample_id}"
            )
        output_vcfs.append(out_vcf)

    return output_vcfs

def ANNOTATE_BY_VEP(chr_vcf_path, sample_id, work_dir, references, fasta_references,
                    singularity_images, ref_ver, fork=4, buffer_size = 50, overwrite=False):
    chr_vcf_path = Path(chr_vcf_path)
    work_dir = Path(work_dir)

    if not chr_vcf_path.exists():
        raise FileNotFoundError(
            f"Input chr VCF not found for sample {sample_id}: {chr_vcf_path}"
        )

    sample_dir = work_dir / "features" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    chr_label = chr_vcf_path.name.replace(".vcf.gz", "")
    output_txt = sample_dir / f"{chr_label}.vep.txt"
    log_path = sample_dir / f"{chr_label}.vep.log"

    if overwrite:
        for p in [output_txt, log_path]:
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        if output_txt.exists():
            return output_txt

    if ref_ver not in {"hg19", "hg38"}:
        raise ValueError(f"Unsupported ref_ver: {ref_ver}. Expected 'hg19' or 'hg38'.")
    ref_assembly = "GRCh38" if ref_ver == "hg38" else "GRCh37"

    vep_refs = references.vep
    sif_path = singularity_images.vep

    bind_paths = sorted({
        str(vep_refs.cache_dir),
        str(vep_refs.plugins_dir.parent),
        str(work_dir),
    })
    bind_args = []
    for p in bind_paths:
        bind_args.extend(["--bind", f"{p}:{p}"])

    vep_cmd = [
        "singularity", "exec",
        *bind_args,
        str(sif_path),
        "/opt/vep/src/ensembl-vep/vep",
        "--dir_cache", str(vep_refs.cache_dir),
        "--dir_plugins", str(vep_refs.plugins_dir),
        "--fork", str(fork),
        "--everything",
        "--format", "vcf",
        "--cache",
        "--offline",
        "--tab",
        "--force_overwrite",
        "--species", "homo_sapiens",
        "--assembly", ref_assembly,
        "--individual", "all",
        "--buffer_size", str(buffer_size),
        "--custom",
        f"{vep_refs.custom_gnomad},gnomADg,vcf,exact,0,AF,AF_popmax,controls_nhomalt",
        "--custom",
        f"{vep_refs.custom_clinvar},clinvar,vcf,exact,0,CLNREVSTAT,CLNSIG,CLNSIGCONF",
        "--custom",
        f"{vep_refs.custom_hgmd},hgmd,vcf,exact,0,CLASS,GENE,PHEN,RANKSCORE",
        "--af_gnomad",
        "--plugin", f"REVEL,{vep_refs.plugin_revel},ALL",
        "--plugin",
        f"SpliceAI,snv={vep_refs.plugin_spliceai_snv},indel={vep_refs.plugin_spliceai_indel},cutoff=0.5",
        "--plugin", f"CADD,{vep_refs.plugin_cadd},ALL",
        "--plugin", f"dbNSFP,{vep_refs.plugin_dbnsfp},ALL",
        "--input_file", str(chr_vcf_path),
        "--output_file", str(output_txt),
    ]

    # Redirect VEP's noisy stdout/stderr to a per-task log file so the notebook
    # stays clean. The log file lives next to the output for easy debugging
    # if a task fails.
    with open(log_path, "w") as log_f:
        try:
            subprocess.run(vep_cmd, check=True, stdout=log_f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            # Re-raise with the log path appended so the orchestrator's error
            # message points the user to the log.
            raise subprocess.CalledProcessError(
                e.returncode, e.cmd,
                output=f"VEP failed. See log for details: {log_path}",
            ) from e

    if not output_txt.exists():
        raise RuntimeError(
            f"ANNOTATE_BY_VEP output was not created for sample {sample_id}, "
            f"chr {chr_label}: {output_txt}  (log: {log_path})"
        )

    return output_txt

def ANNOTATE_BY_MODULES(vep_txt_path, sample_id, hgmd_sim_path, omim_sim_path,
                        work_dir, references, ref_ver,
                        enable_lit=False, overwrite=False):
    """
    Run AIM's feature_main.py on one (sample, chr) to produce a per-chr scores CSV.
    Direct port of the Nextflow ANNOTATE_BY_MODULES process.
    """
    vep_txt_path = Path(vep_txt_path)
    work_dir = Path(work_dir)

    vep_basename = vep_txt_path.name[:-len(".txt")]          # "chr1.vep"
    chr_label = vep_txt_path.name.replace(".vep.txt", "")    # "chr1"

    sample_dir = work_dir / "features" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_csv = sample_dir / f"{vep_basename}_scores.csv"
    staging_dir = sample_dir / f"{chr_label}.staging"

    if overwrite:
        if output_csv.exists():
            output_csv.unlink()
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    else:
        if output_csv.exists():
            return output_csv

    # Stage: feature_main.py's data_loaders reads "annotate/anno_hg*/..." from cwd.
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "annotate").symlink_to(references.ref_annot_dir.resolve())

    cmd = [
        sys.executable,
        str(references.aim_bin_dir / "feature_main.py"),
        "-inFileType", "vepAnnotTab",
        "-patientFileType", "one",
        "-diseaseInh", "AD",
        "-genomeRef", ref_ver,
        "-varFile", str(vep_txt_path.resolve()),
        "-patientHPOsimiOMIM", str(Path(omim_sim_path).resolve()),
        "-patientHPOsimiHGMD", str(Path(hgmd_sim_path).resolve()),
    ]
    if enable_lit:
        cmd.append("-enableLIT")

    subprocess.run(cmd, cwd=str(staging_dir), check=True)

    # feature_main.py writes scores.csv in cwd; rename + move to features/<sample>/
    (staging_dir / "scores.csv").replace(output_csv)
    shutil.rmtree(staging_dir, ignore_errors=True)

    return output_csv


def Process_Pred_Score(sample_id, pred_path, score_path,
                       protein_coding_genes, mane_trascripts):
    """
    Join feature matrix (matrix.txt) with per-transcript gene info
    (scores.txt.gz). For each (varId, geneSymbol), keep the transcript
    with the highest disease-likelihood score.
    """
    # Load feature matrix (RareCollab matrix.txt is tab-separated)
    Predictions = pd.read_csv(pred_path, sep="\t", engine="pyarrow")
    Predictions = Predictions.rename(columns={Predictions.columns[0]: "varId"})
    if Predictions.shape[0] < 1 or Predictions.shape[1] < 1:
        raise ValueError(
            f"Prediction file for {sample_id} is not correct "
            "- Too few variants or too few features"
        )

    # Score file: only read needed columns
    header_cols = set(pd.read_csv(score_path, sep="\t", compression="infer", nrows=0).columns)

    _VTG_DESIRED_COLS = {"geneSymbol", "varId", "HGVSc", "HGVSp"}
    use_cols = sorted(_VTG_DESIRED_COLS & header_cols)
    if len(use_cols) != len(_VTG_DESIRED_COLS):
        tmp = set(_VTG_DESIRED_COLS) - set(use_cols)
        raise ValueError(f"Column(s): {tmp} NOT found in {score_path}")

    dtype_map = {c: "string" for c in use_cols}
    Variant_Symbol = pd.read_csv(
        score_path, sep="\t", compression="infer",
        usecols=use_cols, dtype=dtype_map,
    )

    # Filter to protein-coding genes
    Variant_Symbol = Variant_Symbol[
        Variant_Symbol['geneSymbol'].isin(protein_coding_genes)
    ].copy()

    # Split HGVSc to transcript_id + HGVSc_core
    tmp = Variant_Symbol["HGVSc"].str.split(":", n=1, expand=True)
    Variant_Symbol["transcript_id"] = tmp[0].str.replace(r"\..*$", "", regex=True).fillna("-")
    Variant_Symbol["HGVSc_core"] = tmp[1].fillna("-")

    # Disease-likelihood transcript score
    Variant_Symbol['is_mane'] = Variant_Symbol['transcript_id'].isin(mane_trascripts)
    Variant_Symbol['has_hgvsp'] = ~Variant_Symbol['HGVSp'].isin({'-'})
    Variant_Symbol['has_hgvsc'] = ~Variant_Symbol['HGVSc'].isin({'-'})
    Variant_Symbol["transcript_score"] = (
        4 * Variant_Symbol["is_mane"].astype(int)
        + 2 * Variant_Symbol["has_hgvsp"].astype(int)
        + Variant_Symbol["has_hgvsc"].astype(int)
    )

    # Per (varId, geneSymbol): keep highest-score transcript
    idx = Variant_Symbol.groupby(['varId', 'geneSymbol'])["transcript_score"].idxmax()
    Variant_Symbol = Variant_Symbol.loc[idx].reset_index(drop=True)

    if len(Variant_Symbol) < 1:
        print(f"No intersection between pred and score - {sample_id}")
    
    Variant_Symbol["is_causal"] = 0
    Merged = Predictions.merge(Variant_Symbol, on='varId', how='inner')
    Merged["identifier"] = sample_id
    return Merged


def vtg_process_one(sample_id, pred_path, score_path,
                    protein_coding_genes, mane_trascripts,
                    output_path: Path, overwrite: bool = False) -> int:
    """
    Run VarToGene for one sample, write <sample>.vartogene.feather.
    Atomic write via .tmp + os.replace for crash safety.
    """
    if output_path.exists() and not overwrite:
        return 1

    tmp = output_path.parent / f".{output_path.name}.tmp"
    MergedFile = Process_Pred_Score(
        sample_id, pred_path, score_path,
        protein_coding_genes, mane_trascripts,
    )
    MergedFile.to_feather(tmp)
    os.replace(tmp, output_path)
    return 1

def run_var_to_gene_parallel(samplesheet, references, max_workers, overwrite=False):
    """
    Step 9 worker: join matrix.txt features with per-transcript gene info,
    write <sample>.vartogene.feather to same merged/<sample>/ dir.

    Returns dict[row_index -> str path] for samplesheet update.
    """
    # Load gene/transcript references once
    gencode = pd.read_feather(references.rarecollab.gencode_annot)
    mane = pd.read_feather(references.rarecollab.mane_transcript)
    protein_coding_genes = set(
        gencode.loc[gencode['gene_type'] == 'protein_coding', 'gene_name']
    )
    mane_trascripts = set(
        mane['transcript_id'].str.split('.').str[0].dropna()
    )

    output_paths = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for row in samplesheet.itertuples(index=True):
            sample_id = row.sampleID
            pred_path = Path(row.matrix_txt_path)
            score_path = Path(row.scores_long_path)
            output_path = pred_path.parent / f"{sample_id}.vartogene.feather"

            fut = ex.submit(
                vtg_process_one,
                sample_id, pred_path, score_path,
                protein_coding_genes, mane_trascripts,
                output_path, overwrite,
            )
            futures[fut] = (row.Index, output_path, sample_id)

        with tqdm(total=len(futures), desc="VAR TO GENE") as pbar:
            for fut in as_completed(futures):
                row_idx, out_path, sample_id = futures[fut]
                try:
                    fut.result()
                    output_paths[row_idx] = str(out_path)
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] VAR_TO_GENE failed for {sample_id}")
                    print(f"[ERROR] {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    if fail > 0:
        raise RuntimeError(f"VAR_TO_GENE failed for {fail} sample(s).")

    return output_paths


def GenerateFeatures(samplesheet, work_dir, references,
                    fasta_references, singularity_images, 
                    ref_ver, config=None,overwrite=False):
    """
    Generate AIM singleton features for all samples in the samplesheet.

    Parallelism is controlled per step via the `config` dict. If omitted,
    a conservative default (1 worker per step) is used. To auto-pick values
    based on available CPUs, call recommend_worker_config() first.

    config keys (all int, default 1):
        split_workers       Step 1, SPLIT_VCF_BY_CHROMOSOME       (per sample)
        vep_workers         Step 2, ANNOTATE_BY_VEP               (per sample x chr)
        vep_fork            Step 2, VEP --fork (cores per task)
        vep_buffer_size     Step 2, VEP --buffer_size             (default 50)
        hpo_workers         Step 3, HPO_SIM                        (per sample)
        modules_workers     Step 4, ANNOTATE_BY_MODULES           (per sample x chr)
        phrank_workers      Step 5, PHRANK_SCORING                 (per sample)
        tier_workers        Step 6, ANNOTATE_TIER                  (per sample x chr)
        join_phrank_workers Step 7, JOIN_PHRANK                    (per sample x chr)
        merge_workers       Step 8, MERGE_SCORES_BY_CHROMOSOME    (per sample)
        vartogene_workers   Step 9, VAR_TO_GENE                    (per sample)
    """
    default_config = {
        "split_workers": 1,
        "vep_workers": 1,
        "vep_fork": 1,
        "vep_buffer_size": 50,
        "hpo_workers": 1,
        "modules_workers": 1,
        "phrank_workers": 1,
        "tier_workers": 1, 
        "join_phrank_workers": 1,
        "merge_workers": 1,
        "vartogene_workers": 1,
    }
    cfg = {**default_config, **(config or {})}

    #print("GenerateFeatures config:")
    #for k, v in cfg.items():
    #    print(f"  {k} = {v}")
    
    # 1. Split VCF by chromosome
    samplesheet = samplesheet.copy()
    print('1/9. Splitting VCF by chromosome ...')
    tasks = []
    results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=cfg["split_workers"]) as ex:
        for row in samplesheet.itertuples(index=True):
            future = ex.submit(SPLIT_VCF_BY_CHROMOSOME, row.final_vcf_path, row.sampleID, work_dir, overwrite)
            tasks.append({"index": row.Index, "sampleID": row.sampleID, "vcf_path": row.final_vcf_path, "future": future})

        with tqdm(total=len(tasks), desc="SPLIT VCF BY CHROMOSOME") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    chr_vcfs = future.result()
                    results[row_index] = {"chr_vcf_paths": [str(p) for p in chr_vcfs]}
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] SPLIT_VCF_BY_CHROMOSOME failed for sample {sample_id}")
                    print(f"[ERROR] Input VCF: {task['vcf_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    samplesheet["chr_vcf_paths"] = None
    samplesheet["chr_vcf_paths"] = samplesheet["chr_vcf_paths"].astype(object)
    for row_index, result in results.items():
        samplesheet.at[row_index, "chr_vcf_paths"] = result["chr_vcf_paths"]

    if fail > 0:
        raise RuntimeError(f"SPLIT_VCF_BY_CHROMOSOME failed for {fail} sample(s).")

    
    # 2. Annotate by VEP (parallel per (sample, chr))
    print('2/9. Annotating each chr VCF with VEP ...')
    tasks = []
    vep_results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=cfg["vep_workers"]) as ex:
        for row in samplesheet.itertuples(index=True):
            for chr_vcf in row.chr_vcf_paths:
                future = ex.submit(
                    ANNOTATE_BY_VEP, chr_vcf, row.sampleID, work_dir,
                    references, fasta_references, singularity_images,
                    ref_ver, cfg["vep_fork"], cfg["vep_buffer_size"], overwrite,
                )
                tasks.append({
                    "index": row.Index,
                    "sampleID": row.sampleID,
                    "chr_vcf": chr_vcf,
                    "future": future,
                })

        with tqdm(total=len(tasks), desc="ANNOTATE BY VEP") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]
                chr_vcf = task["chr_vcf"]

                try:
                    vep_txt = future.result()
                    vep_results.setdefault(row_index, []).append(str(vep_txt))
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] ANNOTATE_BY_VEP failed for sample {sample_id}")
                    print(f"[ERROR] Input chr VCF: {chr_vcf}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    samplesheet["vep_txt_paths"] = None
    samplesheet["vep_txt_paths"] = samplesheet["vep_txt_paths"].astype(object)
    for row_index, paths in vep_results.items():
        samplesheet.at[row_index, "vep_txt_paths"] = sorted(paths)

    if fail > 0:
        raise RuntimeError(f"ANNOTATE_BY_VEP failed for {fail} task(s).")

    
    # 3. HPO similarity scoring (per sample) — ProcessPool
    print('3/9. Computing HPO similarity to HGMD genes (-cz) and OMIM diseases (-dx) ...')
    hpo_results = run_hpo_sim_parallel(
        samplesheet=samplesheet,
        work_dir=work_dir,
        references=references,
        max_workers=cfg["hpo_workers"],
        overwrite=overwrite,
    )

    samplesheet["hgmd_sim_path"] = None
    samplesheet["omim_sim_path"] = None
    for row_index, result in hpo_results.items():
        samplesheet.loc[row_index, "hgmd_sim_path"] = result["hgmd_sim_path"]
        samplesheet.loc[row_index, "omim_sim_path"] = result["omim_sim_path"]


    # 4. Annotate by feature modules (per sample, per chr)
    print('4/9. Annotating VEP output with AIM feature modules ...')
    tasks = []
    ann_results = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=cfg["modules_workers"]) as ex:
        for row in samplesheet.itertuples(index=True):
            for vep_txt in row.vep_txt_paths:
                future = ex.submit(
                    ANNOTATE_BY_MODULES,
                    vep_txt, row.sampleID,
                    row.hgmd_sim_path, row.omim_sim_path,
                    work_dir, references, ref_ver,
                    False,
                    overwrite,
                )
                tasks.append({
                    "index": row.Index,
                    "sampleID": row.sampleID,
                    "vep_txt": vep_txt,
                    "future": future,
                })

        with tqdm(total=len(tasks), desc="ANNOTATE BY MODULES") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]
                vep_txt = task["vep_txt"]

                try:
                    scores_csv = future.result()
                    ann_results.setdefault(row_index, []).append(str(scores_csv))
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] ANNOTATE_BY_MODULES failed for sample {sample_id}")
                    print(f"[ERROR] Input VEP: {vep_txt}")
                    print(f"[ERROR] {type(e).__name__}: {e}")

                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    samplesheet["scores_csv_paths"] = None
    samplesheet["scores_csv_paths"] = samplesheet["scores_csv_paths"].astype(object)
    for row_index, paths in ann_results.items():
        samplesheet.at[row_index, "scores_csv_paths"] = sorted(paths)

    if fail > 0:
        raise RuntimeError(f"ANNOTATE_BY_MODULES failed for {fail} task(s).")

    
    # 5. Phrank scoring (per sample)
    print('5/9. Computing Phrank scores ...')
    phrank_results = run_phrank_parallel(
        samplesheet=samplesheet,
        work_dir=work_dir,
        references=references,
        max_workers=cfg["phrank_workers"],
        overwrite=overwrite)

    # Update samplesheet
    samplesheet["phrank_txt_path"] = None
    for row_index, path in phrank_results.items():
        samplesheet.loc[row_index, "phrank_txt_path"] = path

    # step 6. Annotating tier classifications (per chr)
    print('6/9. Annotating tier classifications ...')
    tier_results = run_annotate_tier_parallel(
        samplesheet=samplesheet,
        work_dir=work_dir,
        references=references,
        ref_ver=ref_ver,
        max_workers=cfg["tier_workers"],
        overwrite=overwrite,
    )

    samplesheet["tier_tsv_paths"] = None
    samplesheet["tier_tsv_paths"] = samplesheet["tier_tsv_paths"].astype(object)
    for row_index, paths in tier_results.items():
        samplesheet.at[row_index, "tier_tsv_paths"] = paths

    # Step 7: JOIN_PHRANK
    print('7/9. Joining phrank + clinvar/hgmd expansions ...')
    join_results = run_join_phrank_parallel(
        samplesheet=samplesheet,
        work_dir=work_dir,
        references=references,
        ref_ver=ref_ver,
        max_workers=cfg["join_phrank_workers"],
        overwrite=overwrite,
    )

    samplesheet["joined_scores_paths"] = None
    samplesheet["joined_scores_paths"] = samplesheet["joined_scores_paths"].astype(object)
    for row_index, paths in join_results.items():
        samplesheet.at[row_index, "joined_scores_paths"] = paths
    
    # Step 8: MERGE_SCORES_BY_CHROMOSOME
    print('8/9. Merging scores + final feature engineering ...')
    merge_results = run_merge_parallel(
        samplesheet=samplesheet,
        work_dir=work_dir,
        references=references,
        ref_ver=ref_ver,
        max_workers=cfg["merge_workers"],
        overwrite=overwrite,
    )

    samplesheet["matrix_txt_path"] = None
    samplesheet["scores_long_path"] = None
    for row_index, result in merge_results.items():
        samplesheet.loc[row_index, "matrix_txt_path"] = result["matrix_txt_path"]
        samplesheet.loc[row_index, "scores_long_path"] = result["scores_long_path"]

    # ===== Step 9: VAR_TO_GENE (new) =====
    print('9/9. Variant-to-Gene join + transcript dedup ...')
    vtg_results = run_var_to_gene_parallel(
        samplesheet=samplesheet,
        references=references,
        max_workers=cfg["vartogene_workers"],
        overwrite=overwrite,
    )
    samplesheet["vartogene_feather_path"] = None
    for row_index, path in vtg_results.items():
        samplesheet.loc[row_index, "vartogene_feather_path"] = path

    # ===== Save samplesheet (existing) =====
    samplesheet_out_path = Path(work_dir) / "samplesheet_with_paths.csv"
    samplesheet.to_csv(samplesheet_out_path, index=False)
    print(f"Saved samplesheet to: {samplesheet_out_path}")
    
    return samplesheet

#------------------------------------------------------------
#------------------------------------------------------------
#--------------------------RNA module------------------------
#------------------------------------------------------------
#------------------------------------------------------------

# ===========================================================================
# RNA pipeline
#
# ProcessBAM -> RunFRASER / RunOutrider / RunASE (any order) -> PrepareRNAEvidence
#
# Each Run* takes rna_cohort and returns a copy with one path column added, so a
# single variable carries through. None is never returned: `rna_cohort =
# RunFRASER(...)` would otherwise erase the caller's cohort. No row is ever
# dropped either - the three analyses count different things and fail on
# different samples, so one step's failure must not shrink the input to the next.
#
# Machinery lives in _lib/_rna_worker; only these five are public.
# ===========================================================================


def ProcessBAM(samplesheet, work_path, rna_background=None, overwrite=False,
               max_workers=None, threads_per_sample=None,
               max_reads=200000, target_informative=2000):
    """
    Index every RNA BAM in the cohort and resolve its library layout.

    Cases come from `samplesheet` (rows with a non-empty rna_path), controls
    from `rna_background`. Both go into one cohort, because FRASER and OUTRIDER
    fit a single model across all samples.

    Cached per sample under <work_path>/BAMInfo/<sampleID>/bam_info.json, reused
    while the BAM is unchanged, the samplesheet values are unchanged and the
    index is still there.

    Args:
        samplesheet: DataFrame from Setup.LoadSamplesheet.
        work_path: root work directory.
        rna_background: optional DataFrame from Setup.LoadRNABackground.
        overwrite: re-run detection even when a cached result is valid.
        max_workers, threads_per_sample: parallelism; derived from the core
            count and cohort size when None.
        max_reads: alignments sampled per BAM, split across regions.
        target_informative: spliced XS-tagged reads to aim for before stopping.

    Returns:
        DataFrame with ['sampleID', 'rna_path', 'vcf_path', 'strand',
        'pairedEnd', 'SeqLevelStyle', 'sample_type']. vcf_path is carried
        through so RunASE has everything it needs from this one table; controls
        hold an empty string. Empty DataFrame if no sample has RNA.
    """
    if shutil.which("samtools") is None:
        raise RuntimeError("samtools was not found on PATH. Run "
                           "Setup.CheckRequiredTools() for instructions.")

    out_root = Path(work_path) / "BAMInfo"
    out_root.mkdir(parents=True, exist_ok=True)

    frames = []
    if "rna_path" in samplesheet.columns:
        has_rna = ~(samplesheet["rna_path"].isna()
                    | (samplesheet["rna_path"].astype(str).str.strip() == ""))
        cases = samplesheet.loc[has_rna].copy()
        if not cases.empty:
            cases["sample_type"] = "case"
            frames.append(cases)
    if rna_background is not None and not rna_background.empty:
        controls = rna_background.copy()
        controls["sample_type"] = "control"
        frames.append(controls)

    empty_result = pd.DataFrame(columns=["sampleID", "rna_path", "vcf_path",
                                         "strand", "pairedEnd", "SeqLevelStyle",
                                         "sample_type"])
    if not frames:
        print("No sample has an RNA file; skipping BAM processing.")
        return empty_result

    keep = ["sampleID", "rna_path", "vcf_path", "strand", "pairedEnd", "sample_type"]
    cohort = pd.concat([f[[c for c in keep if c in f.columns]] for f in frames],
                       ignore_index=True)
    for column in ("strand", "pairedEnd"):
        if column not in cohort.columns:
            cohort[column] = "auto"

    # The samplesheet always carries vcf_path - LoadSamplesheet requires it -
    # but rna_background never does, so concat leaves NaN on the control rows.
    # fillna before astype, or NaN becomes the string "nan" and RunASE goes
    # looking for a file called that.
    if "vcf_path" not in cohort.columns:
        cohort["vcf_path"] = ""
    cohort["vcf_path"] = cohort["vcf_path"].fillna("").astype(str)

    n_case = int((cohort["sample_type"] == "case").sum())
    n_control = int((cohort["sample_type"] == "control").sum())
    if n_case == 0:
        print("No case sample has an RNA file; the RNA branch will be skipped.\n"
              "  Background controls alone cannot produce patient results.")
        return empty_result

    max_workers, threads_per_sample = _rna.recommend_bam_parallelism(
        len(cohort), max_workers, threads_per_sample)

    print(f"Processing {len(cohort)} BAM file(s): {n_case} case, {n_control} control.\n"
          f"Results are cached under {out_root} (overwrite={overwrite}).\n"
          f"Using {max_workers} parallel worker(s), {threads_per_sample} thread(s) each.")

    # One bad sample must not abort the batch. Failures are judged at the end: a
    # failed control is dropped the way LoadRNABackground drops missing files,
    # but a failed case means that patient has no RNA results.
    def _run_one(row):
        sample_id = str(row["sampleID"])
        try:
            record, notes = _rna.process_one_sample(
                row, out_root, overwrite, threads_per_sample,
                max_reads, target_informative)
            return record, [(sample_id, n) for n in notes], None
        except Exception as exc:
            return None, [], (sample_id, row["sample_type"], str(exc))

    started = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_one, row) for _, row in cohort.iterrows()]
        outcomes = [f.result() for f in tqdm(as_completed(futures), total=len(futures),
                                             desc="Processing BAMs", unit="sample")]
    elapsed = time.time() - started

    records, notes, failures = [], [], []
    for record, sample_notes, failure in outcomes:
        notes.extend(sample_notes)
        (records if failure is None else failures).append(
            record if failure is None else failure)

    if failures:
        failed_controls = [s for s, kind, _ in failures if kind == "control"]
        failed_cases = [s for s, kind, _ in failures if kind == "case"]
        if failed_controls:
            print(f"\nWARNING: dropping {len(failed_controls)} background "
                  f"control(s) that could not be processed: {failed_controls}")
        if failed_cases:
            detail = "\n".join(f"  {s}: {msg}" for s, kind, msg in failures
                               if kind == "case")
            raise RuntimeError(f"{len(failed_cases)} case sample(s) could not be "
                               f"processed, so they would have no RNA "
                               f"evidence:\n{detail}")
    if not records:
        raise RuntimeError("No BAM file could be processed successfully.")

    result = pd.DataFrame(records)
    order = {sid: i for i, sid in enumerate(cohort["sampleID"])}
    result = (result.sort_values("sampleID", key=lambda s: s.map(order))
              .reset_index(drop=True))

    # vcf_path is joined back from the samplesheet rather than carried through
    # the per-sample record: that record is cached in bam_info.json and
    # describes the BAM, so repointing a sample at a new VCF without touching
    # its BAM would otherwise get a cache hit serving the old path.
    result = result.merge(cohort[["sampleID", "vcf_path"]], on="sampleID", how="left")
    result["vcf_path"] = result["vcf_path"].fillna("").astype(str)

    if notes:
        print(f"\n{len(notes)} sample(s) need a closer look:")
        for sample_id, note in notes:
            print(f"  [{sample_id}] {note}")

    n_cached = int(result.get("from_cache", pd.Series(False, index=result.index)).sum())
    n_built = int(result.get("index_built", pd.Series(False, index=result.index)).sum())

    def _tally(column):
        return ", ".join(f"{v}={c}" for v, c in result[column].value_counts().items())

    print(f"\nDone. {len(result)} BAM file(s) in {elapsed:.0f}s "
          f"({n_cached} from cache, {n_built} newly indexed).\n"
          f"  strand:    {_tally('strand')}\n"
          f"  pairedEnd: {_tally('pairedEnd')}\n"
          f"  contigs:   {_tally('SeqLevelStyle')}")

    styles = set(result["SeqLevelStyle"]) - {"unknown"}
    if len(styles) > 1:
        breakdown = result.groupby("SeqLevelStyle")["sampleID"].apply(list).to_dict()
        print(f"\nNOTE: the cohort mixes chromosome naming styles: {breakdown}\n"
              f"  FRASER can reconcile this, but only via the SeqLevelStyle "
              f"column in its colData; without it the styles stay disjoint and "
              f"the model fits on non-overlapping junctions with no error. "
              f"Reconciling also prunes to standard chromosomes.")

    strands = set(result["strand"])
    if "unstranded" in strands and strands & {"forward", "reverse"}:
        offenders = result.loc[result["strand"] == "unstranded", "sampleID"].tolist()
        print(f"\nWARNING: {len(offenders)} unstranded sample(s) sit in an "
              f"otherwise stranded cohort: {offenders}\n"
              f"  FRASER refuses this outright. Unstranded controls will have to "
              f"be dropped before the FRASER step.")

    missing_vcf = result.loc[(result["sample_type"] == "case")
                             & (result["vcf_path"] == ""), "sampleID"].tolist()
    if missing_vcf:
        print(f"\nNOTE: {len(missing_vcf)} case sample(s) have no vcf_path: "
              f"{missing_vcf}\n  RunASE needs the patient's own heterozygous "
              f"sites and will skip them; FRASER and OUTRIDER are unaffected.")

    return result[["sampleID", "rna_path", "vcf_path", "strand", "pairedEnd",
                   "SeqLevelStyle", "sample_type"]].reset_index(drop=True)


def RunFRASER(rna_cohort, work_path, references, overwrite=False,
              n_threads=None, max_attempts=6):
    """
    Run FRASER 2.0 across the whole RNA cohort, writing one table per case.

    Two stages. Split-read counting is one OS process per sample, because R
    never returns heap to the system and a long-lived worker accumulates the
    peak of every sample it has touched. Everything after runs in one process
    with counting serialised, so peak memory stays at a single sample's worth.

        FraserRes/_cohort/     sample_anno.tsv, run_fraser.R, fraser.log, logs/,
                               cache/, savedObjects/, fraser_results.tsv
        FraserRes/<sampleID>/  junctions.feather, one per case

    The cohort folder holds FRASER's count cache and is meant to be reused
    across batches: split-read counts are a property of one BAM alone, so a
    control counted once never needs counting again. That reuse depends entirely
    on work_path staying the same - a fresh work directory starts from nothing,
    which for forty controls at tens of minutes each is not a small difference.
    Non-split counts are redone whenever cohort membership changes, being
    counted against a splice-site set derived from the whole cohort; those go
    through featureCounts and are far cheaper.

    Args:
        rna_cohort: DataFrame from ProcessBAM.
        work_path: root work directory.
        references: object from Setup.ResolveReferenceInputs.
        overwrite: re-run even when finished results are present.
        n_threads: parallel counting processes, and cores for the fit. Derived
            from available memory and the largest BAM when None.
        max_attempts: retries for the second stage before giving up.

    Returns:
        rna_cohort with a 'fraser_path' column. Cohort-level outputs are not in
        the table because they are not per-sample: fraser_results.tsv and
        savedObjects/*/fds-object.RDS live under FraserRes/_cohort/.
    """
    # Namespace for the cohort-level cache and saved objects. Not a parameter:
    # it keys storage only and has no effect on the numbers, and nothing that
    # would make two analyses genuinely different is exposed yet.
    analysis_name = "my_work"

    if rna_cohort is None:
        print("RNA cohort is empty; skipping FRASER.")
        return pd.DataFrame(columns=["sampleID", "rna_path", "strand", "pairedEnd",
                                     "SeqLevelStyle", "sample_type", "fraser_path"])
    if rna_cohort.empty:
        print("RNA cohort is empty; skipping FRASER.")
        return _rna.attach_output_column(rna_cohort, "fraser_path", {})

    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript was not found on PATH. Run "
                           "Setup.CheckRequiredTools(need_rna=True) for instructions.")
    r_script = _rna.rscript_path("run_fraser.R")

    fraser_root = Path(work_path) / "FraserRes"
    cohort_dir = fraser_root / "_cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    results_tsv = cohort_dir / "fraser_results.tsv"

    print("Preparing the FRASER cohort ...")
    anno, cohort = _rna.prepare_fraser_cohort(rna_cohort)
    case_ids = cohort.loc[cohort["sample_type"] == "case", "sampleID"].astype(str).tolist()

    # `case_ids and` matters: all([]) is True, so a cohort with no cases would
    # otherwise report itself finished.
    expected = [fraser_root / sid / "junctions.feather" for sid in case_ids]
    if case_ids and all(p.exists() for p in expected) and not overwrite:
        print(f"FRASER results already present for all {len(case_ids)} case(s).\n"
              f"  Pass overwrite=True to re-run.")
        return _rna.attach_output_column(rna_cohort, "fraser_path",
                                         dict(zip(case_ids, expected)))

    anno_path = cohort_dir / "sample_anno.tsv"
    anno.to_csv(anno_path, sep="\t", index=False)

    # filterExpressionAndVariability stores its result under savedObjects and,
    # on a later run, rbinds the stored object onto the new one. That rbind
    # needs matching columns, so any membership change makes it fail with an
    # opaque .rbind.SummarizedExperiment error. Both count caches go too, being
    # keyed to a splice-site set derived from the whole cohort. Per-sample split
    # counts are membership-independent and stay - they are the expensive ones.
    roster = sorted(anno["sampleID"])
    roster_file = cohort_dir / "cohort_roster.json"
    previous = json.loads(roster_file.read_text()) if roster_file.exists() else None
    if previous is not None and previous != roster:
        print(f"  cohort membership changed "
              f"(+{len(set(roster) - set(previous))} / "
              f"-{len(set(previous) - set(roster))}); clearing cohort-dependent state")
        for stale in ("savedObjects", "cache/nonSplicedCounts", "cache/otherCounts"):
            shutil.rmtree(cohort_dir / stale, ignore_errors=True)
    roster_file.write_text(json.dumps(roster, indent=2))

    gene_ranges = Path(references.rarecollab.gene_ranges)
    if not gene_ranges.exists():
        print(f"  {gene_ranges.name} not in the reference bundle; generating it")
        gene_ranges = _rna.write_gene_ranges(references.rarecollab.gencode_annot,
                                            cohort_dir / "gene_ranges.tsv")

    shutil.copy2(r_script, cohort_dir / "run_fraser.R")

    if n_threads is None:
        # Peak RSS per counting process tracks the largest BAM, not the mean:
        # one unlucky batch of big samples is what kills a run. Measured at
        # roughly 2.5x the on-disk size.
        n_threads = max(1, min(_get_available_cpus(), 8))
        try:
            largest = max(Path(p).stat().st_size for p in cohort["rna_path"])
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
                        n_threads = max(1, min(n_threads,
                                               int(avail * 0.7 // (largest * 2.5))))
                        break
        except (OSError, ValueError):
            pass

    n_case = len(case_ids)
    print(f"Running FRASER on {len(cohort)} sample(s) ({n_case} case, "
          f"{len(cohort) - n_case} background), {n_threads} process(es)/thread(s).\n"
          f"  Live log: {cohort_dir / 'fraser.log'}")

    # ---- stage 1: split reads, isolated per sample ----
    failures = _rna.count_split_reads_parallel(anno, anno_path, cohort_dir,
                                               analysis_name,
                                               max_workers=n_threads,
                                               keep_scaffolds=False)
    if failures:
        failed = {s for s, _ in failures}
        failed_cases = sorted(failed & set(case_ids))
        if failed_cases:
            detail = "\n".join(f"  {s}: {d}" for s, d in failures)
            raise RuntimeError(f"Split-read counting failed for case sample(s) "
                               f"{failed_cases}, so they would have no RNA "
                               f"evidence:\n{detail}\n"
                               f"  Per-sample logs: {cohort_dir / 'logs'}")
        print(f"  dropping {len(failed)} control(s) that could not be counted: "
              f"{sorted(failed)}")
        anno = anno.loc[~anno["sampleID"].isin(failed)].reset_index(drop=True)
        cohort = cohort.loc[~cohort["sampleID"].isin(failed)].reset_index(drop=True)
        anno.to_csv(anno_path, sep="\t", index=False)
        # Keep the roster in step, or the next run sees a phantom membership
        # change and clears cohort caches for nothing.
        roster_file.write_text(json.dumps(sorted(anno["sampleID"]), indent=2))

    # ---- stage 2: everything else, retrying while it makes progress ----
    #
    # Retrying is worthwhile because FRASER caches counts per sample, so each
    # attempt resumes where the last stopped. Progress is the signal rather than
    # a fixed ladder: while an attempt completes new samples the concurrency is
    # evidently workable, so it repeats; only a run that achieves nothing
    # justifies backing off. An input that fails regardless then shows up within
    # two attempts instead of after hours of stepping down.
    log_path = cohort_dir / "fraser.log"
    threads, attempt, total_elapsed, rc = n_threads, 0, 0.0, 1

    while attempt < max_attempts:
        attempt += 1
        before = _rna.count_cached_samples(cohort_dir, analysis_name)
        cmd = ["Rscript", "--vanilla", str(r_script), str(anno_path),
               str(gene_ranges), str(cohort_dir), str(cohort_dir),
               str(threads), analysis_name]
        if attempt > 1:
            print(f"\nAttempt {attempt}/{max_attempts} with {threads} thread(s) "
                  f"({before}/{2 * len(cohort)} sample-phases cached)")

        rc, elapsed = _rna.invoke_r(cmd, log_path, len(cohort), cohort_dir,
                                    analysis_name, append=(attempt > 1))
        total_elapsed += elapsed
        if rc == 0:
            break

        gained = _rna.count_cached_samples(cohort_dir, analysis_name) - before
        if gained > 0:
            print(f"  attempt {attempt} failed after {elapsed/60:.1f} min but "
                  f"completed {gained} more sample-phase(s); retrying at "
                  f"{threads} thread(s)")
            continue
        if threads <= 1:
            break
        threads = max(1, threads // 2)
        print(f"  attempt {attempt} failed after {elapsed/60:.1f} min with no new "
              f"samples; dropping to {threads} thread(s)")

    if rc != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-25:])
        raise RuntimeError(f"FRASER failed after {attempt} attempt(s), "
                           f"{total_elapsed/60:.1f} min total.\n"
                           f"  Full log: {log_path}\n"
                           f"  To reproduce by hand:\n    {' '.join(cmd)}\n\n"
                           f"--- last lines of the log ---\n{tail}")

    # R writes one gzipped TSV per case so it never holds the whole cohort in
    # memory; converting to feather here keeps the R side free of an arrow
    # dependency and costs one sample's memory at a time.
    produced = {}
    for sid in case_ids:
        src = fraser_root / sid / "junctions.tsv.gz"
        if not src.exists():
            print(f"  WARNING: no junction table produced for {sid}")
            continue
        out = fraser_root / sid / "junctions.feather"
        pd.read_csv(src, sep="\t").to_feather(out)
        src.unlink()
        produced[sid] = out

    if not produced:
        raise RuntimeError(f"FRASER exited cleanly but produced no junction "
                           f"tables. See {log_path}.")

    n_sig = max(0, sum(1 for _ in open(results_tsv)) - 1) if results_tsv.exists() else 0
    print(f"\nFRASER finished in {total_elapsed/60:.1f} min.\n"
          f"  {len(produced)} case sample(s) written under {fraser_root}\n"
          f"  {n_sig} significant event(s) at padj<=0.1, |deltaJaccard|>=0.1\n"
          f"  cohort-level tables under {cohort_dir}")
    return _rna.attach_output_column(rna_cohort, "fraser_path", produced)


def RunOutrider(rna_cohort, work_path, references, overwrite=False,
                n_threads=None, fpkm_cutoff=1.0):
    """
    Run OUTRIDER across the RNA cohort, writing one table per case.

    Like FRASER a cohort method: the autoencoder learns what normal expression
    looks like across samples, so a lone sample has nothing to be an outlier
    against. Cases and controls are fitted together and sliced afterwards.

        OutriderRes/_cohort/     sample_anno.tsv, gene_annotation.saf,
                                 outrider.log, logs/, cache/, ods-object.RDS
        OutriderRes/<sampleID>/  expression.feather, one per case

    Counting is far cheaper than FRASER's: featureCounts streams the BAM in C
    instead of materialising alignments as R objects, so memory stays flat and
    concurrency can be higher. The per-sample cache is also permanent - gene
    counts depend on the BAM and the annotation only, never on cohort
    membership - so a control counted once is never counted again.

    Reads rna_path, strand, pairedEnd and sample_type, and depends on no other
    step, so it can run before or after RunFRASER.

    Args:
        rna_cohort: DataFrame from ProcessBAM or an earlier RNA step.
        work_path: root work directory.
        references: object from Setup.ResolveReferenceInputs.
        overwrite: re-run even when finished results are present.
        n_threads: total parallelism, split between concurrent samples and
            threads each. Derived from the core count when None.
        fpkm_cutoff: genes whose 95th percentile across samples falls below this
            are dropped before fitting. 1.0 is the OUTRIDER default.

    Returns:
        rna_cohort with an 'outrider_path' column, holding every gene that
        survived the expression filter rather than only the significant ones,
        since downstream weighs evidence rather than consuming a verdict.
    """
    if rna_cohort is None:
        print("RNA cohort is empty; skipping OUTRIDER.")
        return pd.DataFrame(columns=["sampleID", "rna_path", "strand", "pairedEnd",
                                     "SeqLevelStyle", "sample_type", "outrider_path"])
    if rna_cohort.empty:
        print("RNA cohort is empty; skipping OUTRIDER.")
        return _rna.attach_output_column(rna_cohort, "outrider_path", {})

    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript was not found on PATH. Run "
                           "Setup.CheckRequiredTools(need_rna=True) for instructions.")
    r_script = _rna.rscript_path("run_outrider.R")

    root = Path(work_path) / "OutriderRes"
    cohort_dir = root / "_cohort"
    cache_dir = cohort_dir / "cache" / "geneCounts"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    # OUTRIDER has no equivalent of FRASER's refusal to mix stranded with
    # unstranded: featureCounts applies strandSpecific per sample, so a mixed
    # cohort counts correctly. An unknown layout still cannot be counted.
    cohort = rna_cohort.copy()
    unknown = (cohort["strand"] == "unknown") | (cohort["pairedEnd"] == "unknown")
    if unknown.any():
        bad_cases = cohort.loc[unknown & (cohort["sample_type"] == "case"), "sampleID"]
        if len(bad_cases) > 0:
            raise RuntimeError(f"Library layout could not be determined for case "
                               f"sample(s): {bad_cases.tolist()}\n"
                               f"  Set 'strand' and 'pairedEnd' in the samplesheet.")
        print(f"  dropping {int(unknown.sum())} control(s) with unknown layout")
        cohort = cohort.loc[~unknown].copy()

    case_ids = cohort.loc[cohort["sample_type"] == "case", "sampleID"].astype(str).tolist()
    if not case_ids:
        raise RuntimeError("No usable case sample; nothing to analyse.")

    if len(cohort) < 50:
        print(f"  WARNING: only {len(cohort)} sample(s) in the cohort. DROP "
              f"recommends at least 50 for both aberrant expression and "
              f"splicing. Below that the latent dimension is capped by the "
              f"sample count, so variation the model cannot absorb - tissue, "
              f"batch, sex, RIN - stays in the residuals, and per-gene "
              f"dispersion from few observations runs conservative.")

    n_control = len(cohort) - len(case_ids)
    if n_control:
        print(f"  NOTE: OUTRIDER assumes the {n_control} background sample(s) are "
              f"independent of one another and of the cases. Relatives - trio "
              f"parents especially - act as biological replicates and can mask "
              f"the very effect being looked for, since the model learns the "
              f"shared variant as normal.")

    expected = [root / sid / "expression.feather" for sid in case_ids]
    if all(p.exists() for p in expected) and not overwrite:
        print(f"OUTRIDER results already present for all {len(case_ids)} case(s).\n"
              f"  Pass overwrite=True to re-run.")
        return _rna.attach_output_column(rna_cohort, "outrider_path",
                                         dict(zip(case_ids, expected)))

    anno = _rna.build_fraser_anno(cohort, with_seqlevelstyle=False)
    anno_path = cohort_dir / "sample_anno.tsv"
    anno.to_csv(anno_path, sep="\t", index=False)

    gene_ranges = Path(references.rarecollab.gene_ranges)
    if not gene_ranges.exists():
        gene_ranges = _rna.write_gene_ranges(references.rarecollab.gencode_annot,
                                             cohort_dir / "gene_ranges.tsv")
    saf_path, n_genes = _rna.write_gene_saf(gene_ranges,
                                            cohort_dir / "gene_annotation.saf")
    print(f"Annotation: {n_genes} gene(s) -> {saf_path.name}")

    shutil.copy2(r_script, cohort_dir / "run_outrider.R")

    if n_threads is None:
        n_threads = max(1, min(_get_available_cpus(), 16))
    # featureCounts holds only summary counters, so several samples at once is
    # safe here in a way it was not for FRASER; the split favours concurrent
    # samples, since each one's own threading scales poorly.
    max_workers = max(1, min(len(anno), n_threads // 2)) if n_threads > 1 else 1
    threads_each = max(1, n_threads // max_workers)

    print(f"Running OUTRIDER on {len(cohort)} sample(s) ({len(case_ids)} case, "
          f"{n_control} background).\n  Live log: {cohort_dir / 'outrider.log'}")

    failures = _rna.count_expression_parallel(anno, anno_path, saf_path, cache_dir,
                                              root, max_workers, threads_each)
    if failures:
        failed = {s for s, _ in failures}
        failed_cases = sorted(failed & set(case_ids))
        if failed_cases:
            detail = "\n".join(f"  {s}: {d}" for s, d in failures)
            raise RuntimeError(f"Gene counting failed for case sample(s) "
                               f"{failed_cases}:\n{detail}\n"
                               f"  Per-sample logs: {cohort_dir / 'logs'}")
        print(f"  dropping {len(failed)} control(s) that could not be counted: "
              f"{sorted(failed)}")
        anno = anno.loc[~anno["sampleID"].isin(failed)].reset_index(drop=True)
        cohort = cohort.loc[~cohort["sampleID"].isin(failed)].reset_index(drop=True)
        anno.to_csv(anno_path, sep="\t", index=False)

    log_path = cohort_dir / "outrider.log"
    cmd = ["Rscript", "--vanilla", str(r_script), str(anno_path), str(cache_dir),
           str(cohort_dir), str(n_threads), str(fpkm_cutoff)]
    started = time.time()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=cohort_dir)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if _rna.phase_label(line) or "Error" in line:
                print(f"    {line.rstrip()}")
        proc.wait()
    elapsed = time.time() - started

    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-25:])
        raise RuntimeError(f"OUTRIDER failed after {elapsed/60:.1f} min "
                           f"(exit {proc.returncode}).\n"
                           f"  Full log: {log_path}\n"
                           f"  To reproduce by hand:\n    {' '.join(cmd)}\n\n"
                           f"--- last lines of the log ---\n{tail}")

    produced = {}
    for sid in case_ids:
        src = root / sid / "expression.tsv.gz"
        if not src.exists():
            print(f"  WARNING: no expression table produced for {sid}")
            continue
        out = root / sid / "expression.feather"
        pd.read_csv(src, sep="\t").to_feather(out)
        src.unlink()
        produced[sid] = out

    if not produced:
        raise RuntimeError(f"OUTRIDER exited cleanly but produced no tables. "
                           f"See {log_path}.")

    print(f"\nOUTRIDER finished in {elapsed/60:.1f} min.\n"
          f"  {len(produced)} case sample(s) written under {root}")
    return _rna.attach_output_column(rna_cohort, "outrider_path", produced)


def RunASE(rna_cohort, work_path, fasta_references, overwrite=False,
           max_workers=None, min_depth=10, min_mapq=10, min_baseq=10,
           ratio_thr=0.10, fdr_alpha=0.05, timeout=4 * 3600):
    """
    Count reads per allele at each patient's heterozygous sites, and flag
    mono-allelic expression.

    Not a cohort method: the comparison is between two alleles within one
    sample, so there is no model fitted across samples and no use for background
    controls - without a matched VCF they have no heterozygous sites. Controls
    are skipped.

        ASERes/_cohort/logs/  one log per sample
        ASERes/<sampleID>/    ase_counts.tsv, ase.feather

    Reads rna_path, vcf_path and SeqLevelStyle, and depends on no other RNA
    step. Contig naming is settled per sample: GATK requires the reference, the
    BAM and the VCF to agree, and only the VCF is cheap to rewrite, so the BAM's
    naming picks the reference and a mismatched VCF is renamed into a copy under
    the sample's work folder. The BAM is never touched.

    Args:
        rna_cohort: DataFrame from ProcessBAM or an earlier RNA step.
        work_path: root work directory.
        fasta_references: object from Setup.BuildReferenceIndex.
        overwrite: re-run even where a result already exists.
        max_workers: samples at once. Derived from cores when None.
        min_depth: reads required at a site for it to be reported.
        min_mapq, min_baseq: GATK read and base quality floors.
        ratio_thr: how extreme an allelic ratio must be, at either end, to be a
            candidate for the MAE flag.
        fdr_alpha: BH threshold the same flag also has to clear.
        timeout: per-GATK-step ceiling, in seconds.

    Returns:
        rna_cohort with an 'ase_path' column, holding every site
        ASEReadCounter reported, with IS_MAE as one column rather than a filter.
    """
    if rna_cohort is None:
        print("RNA cohort is empty; skipping ASE.")
        return pd.DataFrame(columns=["sampleID", "rna_path", "vcf_path", "strand",
                                     "pairedEnd", "SeqLevelStyle", "sample_type",
                                     "ase_path"])
    if rna_cohort.empty:
        print("RNA cohort is empty; skipping ASE.")
        return _rna.attach_output_column(rna_cohort, "ase_path", {})

    for tool in ("gatk", "bcftools"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} was not found on PATH. Run "
                               f"Setup.CheckRequiredTools() for instructions.")
    for path in (fasta_references.fasta, fasta_references.fasta_ucsc):
        if not Path(path).exists():
            raise RuntimeError(f"Reference FASTA missing: {path}")

    out_root = Path(work_path) / "ASERes"
    out_root.mkdir(parents=True, exist_ok=True)

    # Only cases, and only those with a VCF: het sites come from the patient's
    # own DNA, so a sample without one has nothing to measure.
    cases = rna_cohort.loc[rna_cohort["sample_type"] == "case"].copy()
    has_vcf = ~(cases["vcf_path"].isna()
                | (cases["vcf_path"].astype(str).str.strip() == ""))
    skipped = cases.loc[~has_vcf, "sampleID"].tolist()
    cases = cases.loc[has_vcf]

    if skipped:
        print(f"  skipping {len(skipped)} case(s) with no vcf_path: {skipped}")
    if cases.empty:
        print("No case sample has both RNA and a VCF; skipping ASE.")
        return _rna.attach_output_column(rna_cohort, "ase_path", {})

    if max_workers is None:
        # GATK runs on the JVM with a fixed heap and streams the BAM, so memory
        # is flat and the ceiling is cores rather than RAM - unlike FRASER's
        # counting, which had to be sized against the largest BAM.
        max_workers = max(1, min(len(cases), _get_available_cpus() // 2, 8))

    print(f"Running ASE on {len(cases)} case sample(s), {max_workers} process(es) "
          f"at a time.\n  Per-sample logs: {out_root / '_cohort' / 'logs'}",
          flush=True)

    results, failures, notes, site_counts = {}, [], [], []
    bar = tqdm(total=len(cases), unit="sample", desc="ASE", ascii=True,
               bar_format="{desc:<38} {bar} {n_fmt}/{total_fmt} [{elapsed}]")
    stop = threading.Event()
    threading.Thread(target=lambda: [bar.refresh() for _ in
                                     iter(lambda: stop.wait(15), True)],
                     daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_rna.run_ase_one, row, out_root, fasta_references,
                                   overwrite, min_depth, min_mapq, min_baseq,
                                   ratio_thr, fdr_alpha, timeout)
                       for _, row in cases.iterrows()]
            for future in as_completed(futures):
                sid, ok, detail, sample_notes, n_sites, n_mae = future.result()
                notes.extend((sid, n) for n in sample_notes)
                if ok:
                    results[sid] = out_root / sid / "ase.feather"
                    site_counts.append((n_sites, n_mae))
                else:
                    failures.append((sid, detail))
                    bar.write(f"  FAILED {sid}: {detail}")
                bar.update(1)
    finally:
        stop.set()
        bar.close()

    if notes:
        print(f"\n{len(notes)} note(s):")
        for sid, note in notes:
            print(f"  [{sid}] {note}")
    if failures:
        detail = "\n".join(f"  {s}: {d}" for s, d in failures)
        print(f"\nWARNING: {len(failures)} case(s) produced no ASE result:\n"
              f"{detail}\n  Those patients keep an empty ase_path; their other "
              f"RNA evidence is unaffected.")
    if not results:
        raise RuntimeError(f"No sample produced an ASE result. See "
                           f"{out_root / '_cohort' / 'logs'}.")

    sites = [n for n, _ in site_counts]
    maes = [m for _, m in site_counts]
    print(f"\nASE finished. {len(results)} case sample(s) written under {out_root}\n"
          f"  sites per sample: median {int(np.median(sites)):,} "
          f"(range {min(sites):,}-{max(sites):,})\n"
          f"  MAE flagged:      median {int(np.median(maes))} "
          f"(range {min(maes)}-{max(maes)})")
    return _rna.attach_output_column(rna_cohort, "ase_path", results)


def PrepareRNAEvidence(rna_cohort, samplesheet, work_path, overwrite=False,
                       max_workers=None):
    """
    Reshape the RNA outputs for DiagnosticEngine and record them on the
    samplesheet.

        Diagnostic_results/RNA_MoE/Splicing/<sampleID>.feather
        Diagnostic_results/RNA_MoE/Expression/<sampleID>.feather
        Diagnostic_results/RNA_MoE/ASE/<sampleID>.feather

    Nothing is combined; each sample is handled on its own and peak memory is
    one sample's worth. The reshaped copies live apart from the analysis
    outputs, so FraserRes, OutriderRes and ASERes stay as the analyses left them.

    Partial cohorts are expected, not exceptional. A user who ran only ASE, or
    whose FRASER run failed for one patient, gets everything that does exist;
    the samplesheet column for a missing analysis is left empty, and
    DiagnosticEngine.Candidates already treats an absent path as absent evidence.

    Args:
        rna_cohort: DataFrame carrying whichever of fraser_path, outrider_path
            and ase_path exist.
        samplesheet: the DNA samplesheet to annotate.
        work_path: root work directory.
        overwrite: rewrite files that are already there.
        max_workers: samples reshaped at once. Derived from cores when None.

    Returns:
        samplesheet with rna_splicing_path, rna_expression_path and rna_ase_path
        added, also written to <work_path>/samplesheet_with_paths.csv so a later
        session can pick up without re-running anything.
    """
    kinds = _rna.rna_kinds()
    save_root = Path(work_path) / "Diagnostic_results" / "RNA_MoE"

    def _blank(reason):
        print(reason)
        out = samplesheet.copy()
        for _, _, column in kinds.values():
            out[column] = ""
        return out

    if rna_cohort is None or rna_cohort.empty:
        return _blank("RNA cohort is empty; no RNA evidence to prepare.")

    cases = rna_cohort.loc[rna_cohort["sample_type"] == "case"]
    if cases.empty:
        return _blank("No case sample in the RNA cohort; nothing to prepare.")

    available = [k for k, (col, _, _) in kinds.items() if col in cases.columns]
    if not available:
        return _blank("None of fraser_path / outrider_path / ase_path is present "
                      "on the RNA cohort; nothing to prepare.")

    print(f"Preparing RNA evidence for {len(cases)} case sample(s): "
          f"{', '.join(available)}", flush=True)

    if max_workers is None:
        max_workers = max(1, min(len(cases), _get_available_cpus(), 8))

    jobs = {str(row["sampleID"]): {kind: str(row.get(col, "") or "").strip()
                                   for kind, (col, _, _) in kinds.items()
                                   if kind in available}
            for _, row in cases.iterrows()}

    written = {kind: {} for kind in kinds}
    absent = {kind: [] for kind in kinds}
    failures = []

    bar = tqdm(total=len(jobs), unit="sample", desc="RNA evidence", ascii=True,
               bar_format="{desc:<38} {bar} {n_fmt}/{total_fmt} [{elapsed}]")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_rna.prepare_one, sid, sources, save_root, overwrite)
                       for sid, sources in jobs.items()]
            for future in as_completed(futures):
                sid, got, miss, error = future.result()
                for kind, path in got.items():
                    written[kind][sid] = path
                for kind in miss:
                    absent[kind].append(sid)
                if error:
                    failures.append((sid, error))
                    bar.write(f"  FAILED {sid}: {error}")
                bar.update(1)
    finally:
        bar.close()

    for kind, (_, subdir, _) in kinds.items():
        if kind not in available:
            print(f"  {subdir:<11} not run")
            continue
        note = f", {len(absent[kind])} without a result" if absent[kind] else ""
        print(f"  {subdir:<11} {len(written[kind])} sample(s){note}")
    if failures:
        print(f"\nWARNING: {len(failures)} sample(s) could not be reshaped; their "
              f"RNA evidence will be missing downstream.")

    # Joined on sampleID rather than by position: the samplesheet holds every
    # patient, including those with no RNA, and their columns must stay empty
    # rather than pick up a neighbour's path.
    out = samplesheet.copy()
    for kind, (_, _, column) in kinds.items():
        mapping = {k: str(v) for k, v in written[kind].items()}
        out[column] = out["sampleID"].astype(str).map(mapping).fillna("")

    samplesheet_path = Path(work_path) / "samplesheet_with_paths.csv"
    tmp = Path(work_path) / ".samplesheet_with_paths.csv.tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, samplesheet_path)
    print(f"Updated samplesheet: {samplesheet_path}")
    return out