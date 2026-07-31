#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import shutil
import subprocess
import tempfile
import time
import requests
from pathlib import Path

import pandas as pd

from ._lib._references import (
    AimReferences, PhrankReferences, OmimReferences,
    GnomadReferences, VepReferences, SingularityImages,
    FastaReferences, RareCollabReferences,
)
from ._lib._cpu import get_available_cpus as _get_available_cpus
        
def RecommendWorkerConfig(samplesheet):
    """
    Auto-recommend parallelism settings based on available CPUs and number of
    samples. Returns a dict you can splat into GENERATE_SINGLETON_FEATURES.
    """
    n_cpus = _get_available_cpus()
    n_samples = len(samplesheet)

    split_workers = min(n_samples, n_cpus, 12)
    vep_fork = 12
    vep_workers = max(1, min(n_samples, 2*n_cpus // vep_fork))
    hpo_workers = min(n_samples, n_cpus, 12)
    modules_workers = min(n_cpus, 24)

    phrank_workers = min(n_samples, n_cpus, 12)
    tier_workers = min(n_cpus, 24)
    join_phrank_workers = min(n_cpus, 24)
    merge_workers = min(n_samples, n_cpus, 12)
    vartogene_workers = min(n_samples, n_cpus, 12)

    #--------------------------------------------------
    candidates_workers = min(n_samples, n_cpus, 12)
    database_clinvar_filter_workers = min(n_samples, n_cpus, 12)

    return {
        "split_workers": split_workers,
        "vep_workers": vep_workers,
        "vep_fork": vep_fork,
        "vep_buffer_size": 5000,
        "hpo_workers": hpo_workers,
        "modules_workers": modules_workers,
        "phrank_workers": phrank_workers,
        "tier_workers": tier_workers,
        "join_phrank_workers": join_phrank_workers,
        "merge_workers": merge_workers,
        "vartogene_workers": vartogene_workers,
        "candidates_workers": candidates_workers,
        "database_clinvar_filter_workers": database_clinvar_filter_workers,
    }

def PrepareSingularityImages(ref_dir):
    """
    Pull/build Singularity images. The cache directory is persistent across
    batches; place it next to the reference bundle (e.g. ref_dir/singularity_cache).
    """
    if not CheckRequiredTools(mute = True):
        print("\nSingularity image preparation stopped because required tools are missing.")
        return

    singularity_bin = shutil.which("singularity")
    if singularity_bin:
        # singularity layout: <prefix>/bin/singularity  ->  <prefix>/var/singularity/mnt/session
        sing_prefix = Path(singularity_bin).resolve().parent.parent
        session_dir = sing_prefix / "var" / "singularity" / "mnt" / "session"
        session_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["container", "final", "overlay", "source"]:
            (sing_prefix / "var" / "singularity" / "mnt" / sub).mkdir(parents=True, exist_ok=True)

    cache_dir = Path(ref_dir) / "singularity_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # name -> (docker URI, local sif filename)
    images = {
        "vep": ("docker://ensemblorg/ensembl-vep:release_104.3", "vep.sif"),
    }

    results = {}

    print("Checking and preparing Singularity images ...")
    for name, (docker_uri, sif_name) in images.items():
        sif_path = cache_dir / sif_name

        if sif_path.exists():
            print(f"  [{name}] Found cached image: {sif_path}")
            results[name] = sif_path
            continue

        print(f"  [{name}] Pulling {docker_uri} -> {sif_path}")
        tmp_name = f"{sif_name}.partial"
        tmp_sif = cache_dir / tmp_name
        if tmp_sif.exists():
            tmp_sif.unlink()

        try:
            subprocess.run(
                ["singularity", "pull", "--name", tmp_name, docker_uri],
                check=True,
                cwd=str(cache_dir),
            )
        except subprocess.CalledProcessError as e:
            if tmp_sif.exists():
                tmp_sif.unlink()
            raise RuntimeError(
                f"Failed to pull Singularity image '{name}' from {docker_uri}.\n"
                f"  Make sure Docker Hub is reachable and you are not behind a strict firewall.\n"
                f"  Original error: {e}"
            )

        if not tmp_sif.exists():
            raise RuntimeError(
                f"Singularity pull for '{name}' reported success but produced no .sif file: {tmp_sif}"
            )

        tmp_sif.rename(sif_path)
        print(f"  [{name}] Done: {sif_path}")
        results[name] = sif_path

    print("\nAll Singularity images are ready.")
    return SingularityImages(vep=results["vep"])
    
def _validate_reference_inputs(references):
    required_paths = [
        references.chrmap_file,
        references.ensembl_to_location_file,
        references.ensembl_to_symbol_file,

        references.ref_annot_dir,
        references.ref_var_tier_dir,
        references.ref_merge_expand_dir,
        references.ref_mod5_diffusion_dir,

        references.phrank.dagfile,
        references.phrank.disease_annotation,
        references.phrank.gene_annotation,
        references.phrank.disease_gene,

        references.omim.omim_hgmd_phen,
        references.omim.omim_obo,
        references.omim.omim_genemap2,
        references.omim.omim_pheno,

        references.gnomad.genome_vcf,
        references.gnomad.genome_tbi,
        references.gnomad.exome_vcf,
        references.gnomad.exome_tbi,

        references.vep.cache_dir,
        references.vep.plugins_dir,
        references.vep.custom_gnomad,
        references.vep.custom_clinvar,
        references.vep.custom_hgmd,
        references.vep.plugin_revel,
        references.vep.plugin_spliceai_snv,
        references.vep.plugin_spliceai_indel,
        references.vep.plugin_cadd,
        references.vep.plugin_dbnsfp,

        references.rarecollab.gencode_annot,
        references.rarecollab.mane_transcript,
        references.rarecollab.moe_model,
        references.rarecollab.clinvar_feather, 
        references.rarecollab.omim_disease,
        references.rarecollab.hpo_lib,
        references.rarecollab.hpo_genes,
    ]

    missing = [path for path in required_paths if not path.exists()]

    if not references.vep.index_files:
        missing.append(references.vep.cache_dir / "*.tbi")

    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required AIM reference dependency paths:\n"
            f"{missing_text}"
        )

def ResolveReferenceInputs(ref_dir, ref_ver):
    ref_dir = Path(ref_dir)
    aim_bin_dir = Path(__file__).parent / "_aim_bin"

    if ref_ver not in {"hg19", "hg38"}:
        raise ValueError(f"Unsupported ref_ver: {ref_ver}. Expected 'hg19' or 'hg38'.")

    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference directory does not exist: {ref_dir}")

    if not ref_dir.is_dir():
        raise NotADirectoryError(f"Reference path is not a directory: {ref_dir}")

    ref_assembly = "grch37" if ref_ver == "hg19" else "grch38"

    if ref_ver == "hg19":
        vep_gnomad_name = "gnomad.genomes.r2.1.sites.grch37_noVEP.vcf.gz"
        vep_cadd_name = "hg19_whole_genome_SNVs.tsv.gz"
        vep_dbnsfp_name = "dbNSFP4.3a_grch37.gz"
        vep_revel_name = "new_tabbed_revel.tsv.gz"
    else:
        vep_gnomad_name = "gnomad.genomes.GRCh38.v3.1.2.sites.vcf.gz"
        vep_cadd_name = "hg38_whole_genome_SNV.tsv.gz"
        vep_dbnsfp_name = "dbNSFP4.1a_grch38.gz"
        vep_revel_name = "new_tabbed_revel_grch38.tsv.gz"

    vep_ref_dir = ref_dir / "vep" / ref_ver

    references = AimReferences(
        chrmap_file=ref_dir / "bcf_annotate" / "chrmap.txt",
        ensembl_to_location_file=ref_dir / "phrank" / ref_ver / f"{ref_assembly}_symbol_to_location.txt",
        ensembl_to_symbol_file=ref_dir / "phrank" / ref_ver / "ensembl_to_symbol.txt",

        ref_annot_dir=ref_dir / "annotate",
        ref_var_tier_dir=ref_dir / "var_tier",
        ref_merge_expand_dir=ref_dir / "merge_expand",
        ref_mod5_diffusion_dir=ref_dir / "mod5_diffusion",

        phrank=PhrankReferences(
            dagfile=ref_dir / "phrank" / ref_ver / "child_to_parent.txt",
            disease_annotation=ref_dir / "phrank" / ref_ver / "disease_to_pheno.txt",
            gene_annotation=ref_dir / "phrank" / ref_ver / "gene_to_phenotype.txt",
            disease_gene=ref_dir / "phrank" / ref_ver / "disease_to_gene.txt",
        ),

        omim=OmimReferences(
            omim_hgmd_phen=ref_dir / "omim_annotate" / ref_ver / "HGMD_phen.tsv",
            omim_obo=ref_dir / "omim_annotate" / "hp.obo",
            omim_genemap2=ref_dir / "omim_annotate" / ref_ver / "genemap2_v2022.rds",
            omim_pheno=ref_dir / "omim_annotate" / ref_ver / "HPO_OMIM.tsv",
        ),

        gnomad=GnomadReferences(
            genome_vcf=ref_dir / "filter_vep" / ref_ver / f"gnomad.{ref_ver}.blacklist.genomes.vcf.gz",
            genome_tbi=ref_dir / "filter_vep" / ref_ver / f"gnomad.{ref_ver}.blacklist.genomes.vcf.gz.tbi",
            exome_vcf=ref_dir / "filter_vep" / ref_ver / f"gnomad.{ref_ver}.blacklist.exomes.vcf.gz",
            exome_tbi=ref_dir / "filter_vep" / ref_ver / f"gnomad.{ref_ver}.blacklist.exomes.vcf.gz.tbi",
        ),

        vep=VepReferences(
            cache_dir=vep_ref_dir,
            plugins_dir=vep_ref_dir / "Plugins",
            custom_gnomad=vep_ref_dir / vep_gnomad_name,
            custom_clinvar=vep_ref_dir / "clinvar_20220730.vcf.gz",
            custom_hgmd=vep_ref_dir / f"HGMD_Pro_2022.2_{ref_ver}.vcf.gz",
            plugin_revel=vep_ref_dir / vep_revel_name,
            plugin_spliceai_snv=vep_ref_dir / f"spliceai_scores.masked.snv.{ref_ver}.vcf.gz",
            plugin_spliceai_indel=vep_ref_dir / f"spliceai_scores.masked.indel.{ref_ver}.vcf.gz",
            plugin_cadd=vep_ref_dir / vep_cadd_name,
            plugin_dbnsfp=vep_ref_dir / vep_dbnsfp_name,
            index_files=tuple(sorted(vep_ref_dir.glob("*.tbi"))),
        ),

        rarecollab=RareCollabReferences(
            gencode_annot=ref_dir / "rarecollab" / "gencode_annot.feather",
            gene_ranges=ref_dir / "rarecollab" / "gene_ranges.tsv",
            mane_transcript=ref_dir / "rarecollab" / "mane_transcript.feather",
            moe_model=ref_dir / "rarecollab" / "MoE_finalized.pt",
            clinvar_feather=ref_dir / "rarecollab" / "ClinVarVCF.feather", 
            omim_disease=ref_dir / "rarecollab" / "OMIM_Disease_Description.tsv",
            hpo_lib=ref_dir / "rarecollab" / "hp.obo",
            hpo_genes=ref_dir / "rarecollab" / "HPO_genes_to_phenotype.txt",
            clingen_dosage=ref_dir / "rarecollab" / "ClinGen_Dosage_Info.csv",
        ),

        aim_bin_dir=aim_bin_dir
    )

    _validate_reference_inputs(references)

    return references

def CheckRequiredTools(need_rna=False, mute=False):
    """
    Check the command-line tools (and, if need_rna=True, the R + FRASER stack)
    that RareCollab needs, then print a single combined verdict.

    DNA tools are always checked. RNA tools are only checked when need_rna=True,
    because DNA-only analysis is valid but RNA-only is not (RNA must accompany
    DNA).

    RNA side only checks FRASER (OUTRIDER comes in as its dependency), and
    REQUIRES FRASER >= 1.99.0, i.e. FRASER 2.0. Older FRASER (1.x) produces a
    different output schema (different metrics/columns) that downstream code
    does not support, so an older FRASER is treated as NOT usable.

    When mute=True, nothing is printed at all; rely on the return value.

    Returns True only if everything that was checked is available (and, for
    FRASER, new enough), else False.
    """
    MIN_FRASER = "1.99.0"   # first FRASER 2.0 release

    dna_tools = {
        "samtools":    "conda install -y -c conda-forge -c bioconda samtools",
        "gatk":        "conda install -y -c conda-forge -c bioconda gatk4",
        "wget":        "conda install -y -c conda-forge wget",
        "gunzip":      "conda install -y -c conda-forge gzip",
        "file":        "conda install -y -c conda-forge file",
        "bgzip":       "conda install -y -c conda-forge -c bioconda htslib",
        "tabix":       "conda install -y -c conda-forge -c bioconda htslib",
        "bcftools":    "conda install -y -c conda-forge -c bioconda bcftools",
        "singularity": "conda install -y -c conda-forge singularityce",
    }
    # FRASER 2.0; installing it also pulls OUTRIDER in as a dependency.
    fraser_install = "conda install -y -c conda-forge -c bioconda bioconductor-fraser=2.2.0"

    if not mute:
        print("Python executable:", sys.executable)
        print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
        print()

    # ================= DNA: command-line tools =================
    if not mute:
        print("=== DNA tools ===")
    dna_missing = {}
    for tool, cmd in dna_tools.items():
        path = shutil.which(tool)
        if path is None:
            dna_missing[tool] = cmd
            if not mute:
                print(f"{tool}: NOT FOUND")
        elif not mute:
            print(f"{tool}: {path}")
    dna_ok = not dna_missing
    if dna_ok and not mute:
        print("All DNA command-line tools are available.")

    # ================= RNA: R + FRASER (>= 2.0) =================
    rna_ok = None
    rna_missing = {}
    if need_rna:
        if not mute:
            print("\n=== RNA tools ===")
        rscript = shutil.which("Rscript")
        if rscript is None:
            if not mute:
                print("Rscript: NOT FOUND")
            rna_missing["R"] = "conda install -y -c conda-forge r-base=4.4"
            if not mute:
                print("FRASER: cannot verify (R missing)")
            rna_missing["FRASER"] = fraser_install
        else:
            if not mute:
                print(f"Rscript: {rscript}")
                # show R version too (FRASER 2.0 needs R >= 4.3)
                rv = subprocess.run(
                    [rscript, "-e", 'cat(as.character(getRversion()))'],
                    capture_output=True, text=True,
                )
                if rv.returncode == 0 and rv.stdout.strip():
                    print(f"R version: {rv.stdout.strip()}")

            # Ask R for FRASER status: MISSING / OLD:<ver> / OK:<ver>
            r_expr = (
                'if (!requireNamespace("FRASER", quietly=TRUE)) {'
                '  cat("MISSING")'
                '} else {'
                '  v <- packageVersion("FRASER");'
                f'  if (v >= numeric_version("{MIN_FRASER}")) cat("OK:", as.character(v), sep="")'
                '  else cat("OLD:", as.character(v), sep="")'
                '}'
            )
            out = subprocess.run([rscript, "-e", r_expr], capture_output=True, text=True)

            if out.returncode != 0:
                # Rscript itself errored - do NOT assume anything is installed
                if not mute:
                    print("FRASER: check FAILED (Rscript error)")
                    if out.stderr.strip():
                        print("  " + out.stderr.strip().splitlines()[-1])
                rna_missing["FRASER"] = fraser_install
            else:
                status = out.stdout.strip()
                if status.startswith("OK:"):
                    if not mute:
                        print(f"FRASER: {status[3:]}")
                elif status.startswith("OLD:"):
                    old_ver = status[4:]
                    if not mute:
                        print(f"FRASER: {old_ver} is too old (need >= {MIN_FRASER}, i.e. FRASER 2.0)")
                        print("  FRASER 1.x has a different output schema and is not supported.")
                    rna_missing["FRASER"] = fraser_install
                else:  # MISSING or anything unexpected
                    if not mute:
                        print("FRASER: NOT FOUND")
                    rna_missing["FRASER"] = fraser_install

        rna_ok = not rna_missing
        if rna_ok and not mute:
            print("All RNA tools are available.")

    # ================= install instructions for whatever is missing =================
    all_missing = {}
    all_missing.update({t: (c, "DNA") for t, c in dna_missing.items()})
    all_missing.update({t: (c, "RNA") for t, c in rna_missing.items()})
    if all_missing and not mute:
        print("\nSome required tools are missing.")
        print("Activate the same conda environment used by this Jupyter kernel:")
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            print(f"\n  conda activate {os.path.basename(conda_prefix)}")
        else:
            print("\n  conda activate <your_environment_name>")
        print("\nThen install the missing tools:")
        for tool, (cmd, kind) in all_missing.items():
            print(f"  # {tool} ({kind})")
            print(f"  {cmd}")
        print("\nAfter installation, restart the Jupyter kernel and run this check again.")

    # ================= combined verdict =================
    if not mute:
        print("\n=== Summary ===")
        if not need_rna:
            if dna_ok:
                print("DNA tools complete. DNA-only analysis is ready. (RNA not requested.)")
            else:
                print("DNA tools missing (see above). DNA is required for every analysis.")
        elif dna_ok and rna_ok:
            print("DNA and RNA tools both complete. Full DNA + RNA analysis is ready.")
        elif dna_ok and not rna_ok:
            print("DNA tools complete, but RNA tools are missing (see above).")
            print("You can still run DNA-only analysis now; install the RNA tools to enable RNA.")
        elif not dna_ok and rna_ok:
            print("RNA tools are present, but DNA tools are missing (see above).")
            print("RNA analysis requires DNA, so install the DNA tools before running anything.")
        else:
            print("Both DNA and RNA tools are missing (see above). Install both to proceed.")

    if not need_rna:
        return dna_ok
    return dna_ok and rna_ok

def BuildReferenceIndex(ref_ver, ref_dir):
    """
    Locate and validate the reference genome files, in both contig naming
    conventions.

    Two copies of the genome are needed, not one. GATK requires the reference,
    the BAM and the VCF to agree on contig names, and a BAM cannot be renamed
    cheaply - rewriting its header streams out a full copy, which for a cohort
    of a few hundred gigabytes is not an option. So the BAM's convention is
    authoritative and the reference is chosen to match it:

        fasta / fai / dict            Ensembl naming: 1, 2, X
        fasta_ucsc / fai_ucsc / ...   UCSC naming:    chr1, chr2, chrX

    Renaming one into the other would not work. Main chromosomes are the same
    sequence under both conventions, but scaffolds are not a prefix apart -
    Ensembl's KI270728.1 is UCSC's chr1_KI270728v1_random - so a mechanical
    rename produces contig names that exist in neither. Both copies therefore
    ship in the reference bundle.

    This function only checks; it does not download or build. The bundle is a
    maintained artefact, so a missing file means the bundle is incomplete or
    the path is wrong, and guessing at a repair would hide that.

    Args:
        ref_ver: 'hg19' or 'hg38'.
        ref_dir: root of the reference bundle.

    Returns:
        FastaReferences with all six paths.

    Raises:
        FileNotFoundError: listing every missing file at once, so the bundle
            can be fixed in one pass rather than one error at a time.
    """
    if ref_ver not in {"hg19", "hg38"}:
        raise ValueError(
            f"Unsupported ref_ver: {ref_ver}. Expected 'hg19' or 'hg38'."
        )

    ref_dir = Path(ref_dir)
    ensembl_dir = ref_dir / "ref_human_genome"
    ucsc_dir = ref_dir / "ref_human_genome_UCSC"

    paths = {
        "fasta":      ensembl_dir / f"final_{ref_ver}.fa",
        "fai":        ensembl_dir / f"final_{ref_ver}.fa.fai",
        "dict":       ensembl_dir / f"final_{ref_ver}.dict",
        "fasta_ucsc": ucsc_dir / f"final_{ref_ver}.fa",
        "fai_ucsc":   ucsc_dir / f"final_{ref_ver}.fa.fai",
        "dict_ucsc":  ucsc_dir / f"final_{ref_ver}.dict",
    }

    print("Checking reference genome files ...")

    missing = {name: p for name, p in paths.items() if not p.exists()}
    if missing:
        listed = "\n".join(f"  {name:12s} {p}" for name, p in missing.items())
        raise FileNotFoundError(
            f"{len(missing)} reference file(s) missing for {ref_ver}:\n{listed}\n\n"
            f"  Both naming conventions are required: GATK will not accept a "
            f"reference whose contig names differ from the BAM's, and the BAM "
            f"is the one thing too large to rewrite.\n"
            f"  The UCSC copy is the analysis set from\n"
            f"    https://hgdownload.soe.ucsc.edu/goldenPath/{ref_ver}/bigZips/"
            f"analysisSet/{ref_ver}.analysisSet.fa.gz\n"
            f"  indexed with `samtools faidx` and "
            f"`gatk CreateSequenceDictionary`."
        )

    # A .fai or .dict left over from an earlier FASTA is a real failure mode,
    # and GATK's complaint about it names neither file. Comparing contig counts
    # catches a mismatched pair without relying on modification times, which
    # copying a bundle around does not preserve reliably.
    for label, fai, dic in (("Ensembl", paths["fai"], paths["dict"]),
                            ("UCSC", paths["fai_ucsc"], paths["dict_ucsc"])):
        n_fai = sum(1 for _ in open(fai))
        n_dict = sum(1 for line in open(dic) if line.startswith("@SQ"))
        if n_fai != n_dict:
            raise ValueError(
                f"The {label} reference index and sequence dictionary disagree: "
                f"{fai.name} lists {n_fai} contig(s) but {dic.name} lists "
                f"{n_dict}.\n"
                f"  One of them was built from a different FASTA. Delete both "
                f"and regenerate with `samtools faidx` and "
                f"`gatk CreateSequenceDictionary`."
            )

    print("*Files Found*")
    print(f"  Ensembl naming: {paths['fasta']}")
    print(f"  UCSC naming:    {paths['fasta_ucsc']}")

    return FastaReferences(**paths)

def _ResolveRNAColumn(df, column):
    """
    Ensure `column` exists on df and holds only allowed, lowercased values.

    Column absent    -> created, filled with 'auto'.
    Blank / NA cell  -> 'auto', quietly (blank is a normal way to say "unknown").
    Anything else    -> 'auto' + a warning naming the offending rows.

    'auto' is resolved from the BAM later, during RNA preprocessing.
    """
    allowed_by_column = {
        "strand":    ("unstranded", "forward", "reverse", "auto"),
        "pairedEnd": ("yes", "no", "auto"),
    }
    allowed = allowed_by_column[column]

    if column not in df.columns:
        df[column] = "auto"
        print(f"No '{column}' column found; adding it with 'auto' for every sample.")
        return df

    values, bad = [], []
    for idx, raw in df[column].items():
        # NA check first: str(NaN) would give "nan" and warn on every blank row.
        text = "" if pd.isna(raw) else str(raw).strip().lower()
        if text == "":
            values.append("auto")
        elif text in allowed:
            values.append(text)
        else:
            bad.append(f"row {int(idx) + 2}: '{str(raw).strip()}'")
            values.append("auto")

    df[column] = values

    if bad:
        print(
            f"WARNING: invalid '{column}' value(s) -> {', '.join(bad)}\n"
            f"  These have been set to 'auto'. Allowed values: {' | '.join(allowed)}"
        )
    else:
        print(f"'{column}' column found; all values are valid.")

    return df


def LoadSamplesheet(csv_path, fulfill_empty_hpo=False):
    """
    Load and validate samplesheet.

    Required columns:
        - sampleID
        - vcf_path
        - hpo_path

    Optional column:
        - rna_path
            * column absent            -> proceed without RNA.
            * column present, all empty -> read in, then drop the column.
            * column present, some empty -> keep as-is; empty rows mean the
              sample has no RNA file. Non-empty paths are validated to exist.

    Optional columns, only handled when rna_path is present:
        - strand    : unstranded | forward | reverse | auto
        - pairedEnd : yes | no | auto
      Case-insensitive. Missing columns are created and filled with 'auto';
      blank or invalid values become 'auto' (invalid ones with a warning).
      'auto' is resolved from the BAM later, during RNA preprocessing.

    If hpo_path exists:
        - validate HPO format.

    If hpo_path does not exist:
        - fulfill_empty_hpo=False: raise FileNotFoundError.
        - fulfill_empty_hpo=True: create the file and write HP:0000001.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Samplesheet not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Samplesheet is empty.")

    # Check column names
    required_cols = {"sampleID", "vcf_path", "hpo_path"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Samplesheet is missing required columns: {sorted(missing_cols)}")
    print("All required columns are found.")

    # Check empty sampleID
    empty_sample_mask = (df["sampleID"].isna() | (df["sampleID"].astype(str).str.strip() == ""))
    if empty_sample_mask.any():
        bad_rows = (df.index[empty_sample_mask] + 2).tolist()
        raise ValueError(f"Samplesheet contains empty sampleID values at rows: {bad_rows}")

    # Check sampleID filename safety
    sample_id_pattern = re.compile(r"^[A-Za-z0-9._-]+$")
    invalid_sample_mask = ~df["sampleID"].astype(str).str.strip().str.match(sample_id_pattern)
    if invalid_sample_mask.any():
        bad_entries = [{"row": int(idx + 2), "sampleID": str(df.loc[idx, "sampleID"]),} for idx in df.index[invalid_sample_mask]]
        raise ValueError(
            "Samplesheet contains invalid sampleID values.\n"
            "sampleID will be used as a directory and filename, so it may only contain:\n"
            "  letters A-Z/a-z, numbers 0-9, dot '.', underscore '_', and hyphen '-'.\n"
            f"Invalid entries: {bad_entries}"
        )

    print("All sampleID values are filename-safe.")

    # Check duplicated sampleID
    duplicated = df.loc[df["sampleID"].duplicated(), "sampleID"].tolist()
    if duplicated:
        raise ValueError(f"Samplesheet contains duplicated sampleID values: {duplicated}")
    print('No empty or duplicated sample ID')

    # ---- Optional rna_path column ----
    has_rna_col = "rna_path" in df.columns
    if has_rna_col:
        rna_empty_mask = df["rna_path"].isna() | (df["rna_path"].astype(str).str.strip() == "")
        if rna_empty_mask.all():
            print("rna_path column present but all values are empty; dropping the column.")
            df = df.drop(columns=["rna_path"])
            has_rna_col = False
        else:
            n_with_rna = int((~rna_empty_mask).sum())
            print(f"rna_path column present; {n_with_rna}/{len(df)} sample(s) have an RNA path, the rest left empty.")
    else:
        print("No rna_path column found; proceeding without RNA.")

    # ---- Optional strand / pairedEnd columns (RNA only) ----
# ---- Optional strand / pairedEnd columns (RNA only) ----
    if has_rna_col:
        df = _ResolveRNAColumn(df, "strand")
        df = _ResolveRNAColumn(df, "pairedEnd")

    # HPO ID format: HP: + 7 digits
    hpo_pattern = re.compile(r"^HP:\d{7}$")
    for idx, row in df.iterrows():
        row_num = idx + 2
        sample_id = str(row["sampleID"]).strip()

        vcf_raw = row["vcf_path"]
        hpo_raw = row["hpo_path"]

        # ---- Check VCF path ----
        if pd.isna(vcf_raw) or str(vcf_raw).strip() == "":
            raise ValueError(f"Missing vcf_path for sample {sample_id} at row {row_num}")
        vcf_path = Path(str(vcf_raw).strip())

        if not vcf_path.exists():
            raise FileNotFoundError(f"VCF file not found for sample {sample_id}: {vcf_path}")

        # ---- Check RNA path (optional) ----
        # Placed BEFORE HPO handling on purpose: the HPO block can `continue`,
        # which would otherwise skip RNA validation.
        if has_rna_col:
            rna_raw = row["rna_path"]
            if pd.isna(rna_raw) or str(rna_raw).strip() == "":
                pass  # this sample has no RNA file; leave it empty
            else:
                rna_path = Path(str(rna_raw).strip())
                if not rna_path.exists():
                    raise FileNotFoundError(f"RNA file not found for sample {sample_id}: {rna_path}")

        # ---- Check HPO path string ----
        if pd.isna(hpo_raw) or str(hpo_raw).strip() == "":
            raise ValueError(f"Missing hpo_path for sample {sample_id} at row {row_num}")

        hpo_path = Path(str(hpo_raw).strip())

        # ---- HPO file handling ----
        if not hpo_path.exists():
            if fulfill_empty_hpo:
                print(f"HPO file not found for sample {sample_id}, creating default HPO file with HP:0000001: {hpo_path}")
                hpo_path.parent.mkdir(parents=True, exist_ok=True)
                with open(hpo_path, "w") as f:
                    f.write("HP:0000001\n")
                continue
            else:
                raise FileNotFoundError(f"HPO file not found for sample {sample_id}: {hpo_path}")

        # ---- Only validate HPO files that already exist ----
        invalid_lines = []
        valid_count = 0

        with open(hpo_path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                term = line.strip()

                # Ignore blank lines
                if term == "":
                    continue

                if hpo_pattern.match(term):
                    valid_count += 1
                else:
                    invalid_lines.append((line_num, term))

        if invalid_lines:
            error_msg = "\n".join(
                [f"  line {line_num}: {term}" for line_num, term in invalid_lines]
            )

            raise ValueError(
                f"Invalid HPO format in file for sample {sample_id}: {hpo_path}\n"
                f"Expected format: HP: followed by 7 digits, e.g. HP:0000957\n"
                f"Invalid lines:\n{error_msg}"
            )

        if valid_count == 0:
            if fulfill_empty_hpo:
                print(f"HPO file is empty for sample {sample_id}, creating default HPO file with HP:0000001: {hpo_path}")
                with open(hpo_path, "w") as f:
                    f.write("HP:0000001\n")
            else:
                raise ValueError(
                    f"HPO file is empty or contains only blank lines for sample "
                    f"{sample_id}: {hpo_path}"
                )

    print('All VCF files are found.')
    print('All HPO files are found and validated.')
    if has_rna_col:
        print('rna_path column kept; provided RNA files validated, empty rows left as-is.')
        print("strand / pairedEnd resolved; 'auto' entries will be detected from the BAM.")

    return df

def LoadRNABackground(csv_path, existing_sample_ids=None):
    """
    Load and validate a supplementary RNA background samplesheet (controls).

    Background samples are added to the FRASER/OUTRIDER cohort so the outlier
    models have enough samples to fit when the user's own cohort is small.

    Required columns:
        - sampleID : filename-safe label, only needs to be unique and not clash
                     with the main samplesheet. The name itself is not meaningful
                     for background/control samples.
        - rna_path : path to an RNA file (e.g. BAM). Rows whose file does not
                     exist are dropped (not an error) - only real files are kept.

    Optional columns (same rules as the main samplesheet):
        - strand    : unstranded | forward | reverse | auto
        - pairedEnd : yes | no | auto
      Case-insensitive. Missing columns are created and filled with 'auto';
      blank or invalid values become 'auto' (invalid ones with a warning).

    Any other column is ignored.

    Note: this function does NOT check for .bai index files. Index existence is
    handled later in the pipeline (generate one if missing, reuse if present).

    Args:
        csv_path: path to the background CSV.
        existing_sample_ids: optional iterable of sampleIDs already in the main
            cohort. Background sampleIDs must not collide with them.

    Returns:
        DataFrame with columns ['sampleID', 'rna_path', 'strand', 'pairedEnd'],
        only rows whose file exists.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"RNA background samplesheet not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("RNA background samplesheet is empty.")

    # ---- Required columns ----
    required_cols = {"sampleID", "rna_path"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"RNA background samplesheet is missing required columns: {sorted(missing_cols)}"
        )
    print("All required columns are found.")

    # Keep the required columns plus the optional RNA library-layout ones
    keep_cols = ["sampleID", "rna_path"] + [
        c for c in ("strand", "pairedEnd") if c in df.columns
    ]
    ignored = [c for c in df.columns if c not in keep_cols]
    if ignored:
        print(f"Ignoring extra column(s): {ignored}")
    df = df[keep_cols].copy()

    # ---- sampleID: empty ----
    empty_sample_mask = df["sampleID"].isna() | (df["sampleID"].astype(str).str.strip() == "")
    if empty_sample_mask.any():
        bad_rows = (df.index[empty_sample_mask] + 2).tolist()
        raise ValueError(f"RNA background contains empty sampleID values at rows: {bad_rows}")
    df["sampleID"] = df["sampleID"].astype(str).str.strip()

    # ---- sampleID: filename safety (still used as a sample name) ----
    sample_id_pattern = re.compile(r"^[A-Za-z0-9._-]+$")
    invalid_sample_mask = ~df["sampleID"].str.match(sample_id_pattern)
    if invalid_sample_mask.any():
        bad_entries = [
            {"row": int(idx + 2), "sampleID": df.loc[idx, "sampleID"]}
            for idx in df.index[invalid_sample_mask]
        ]
        raise ValueError(
            "RNA background contains invalid sampleID values.\n"
            "sampleID will be used as a sample name/filename, so it may only contain:\n"
            "  letters A-Z/a-z, numbers 0-9, dot '.', underscore '_', and hyphen '-'.\n"
            f"Invalid entries: {bad_entries}"
        )
    print("All sampleID values are filename-safe.")

    # ---- sampleID: duplicates within the background ----
    duplicated = df.loc[df["sampleID"].duplicated(), "sampleID"].tolist()
    if duplicated:
        raise ValueError(f"RNA background contains duplicated sampleID values: {duplicated}")

    # ---- sampleID: collision with the main cohort (the one constraint that matters) ----
    if existing_sample_ids is not None:
        collisions = sorted(set(df["sampleID"]) & set(map(str, existing_sample_ids)))
        if collisions:
            raise ValueError(
                "RNA background sampleID(s) collide with the main cohort; sample names "
                "must be unique across the combined cohort:\n"
                f"  {collisions}"
            )
    print("No empty, duplicated, or colliding sample IDs.")

    # ---- rna_path: empty rows are dropped ----
    rna_empty_mask = df["rna_path"].isna() | (df["rna_path"].astype(str).str.strip() == "")
    if rna_empty_mask.any():
        dropped_empty = df.loc[rna_empty_mask, "sampleID"].tolist()
        print(f"Dropping {len(dropped_empty)} row(s) with empty rna_path: {dropped_empty}")
        df = df.loc[~rna_empty_mask].copy()

    df["rna_path"] = df["rna_path"].astype(str).str.strip()

    # ---- rna_path: drop rows whose file does not exist (not an error) ----
    exists_mask = df["rna_path"].apply(lambda p: Path(p).exists())
    if (~exists_mask).any():
        dropped_missing = df.loc[~exists_mask, ["sampleID", "rna_path"]]
        print(f"Dropping {int((~exists_mask).sum())} row(s) whose RNA file was not found:")
        print(dropped_missing.to_string(index=False))
        df = df.loc[exists_mask].copy()

    if df.empty:
        raise ValueError("No usable background samples remain after dropping missing files.")

    # ---- Optional strand / pairedEnd columns ----
    # Resolved after the row drops so warnings only mention rows that survive.
    df = _ResolveRNAColumn(df, "strand")
    df = _ResolveRNAColumn(df, "pairedEnd")

    print(f"{len(df)} background RNA file(s) found and kept.")
    return df[["sampleID", "rna_path", "strand", "pairedEnd"]].reset_index(drop=True)

def LaunchLLMServer(
    partition="", nodelist=None, port=12321, num_parallel=2,
    model_name="gpt-oss:20b", job_name="ollama_server", log_dir="",
    timeout_seconds=300, mem="64G", cpus_per_task=4, gpus=1,
):
    """
    Launch an Ollama LLM server via SLURM + Singularity, wait until ready,
    and return its connection information.

    Parameters
    ----------
    mem:
        SLURM system-memory request, for example "64G".
        This is CPU RAM, not GPU VRAM.
    cpus_per_task:
        Number of CPU cores allocated to the Ollama server.
    gpus:
        Number of GPUs requested from SLURM.

    Returns
    -------
    dict
        model_name, ollama_url, num_parallel, job_id, and node.
    """
    nodelist_line = f"#SBATCH --nodelist={nodelist}" if nodelist else ""

    log_path = Path(log_dir) / f"{job_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{nodelist_line}
#SBATCH --output={log_path}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{gpus}

set -euo pipefail

module load singularity

singularity exec --nv \\
    --env OLLAMA_HOST=0.0.0.0:{port} \\
    --env OLLAMA_NUM_PARALLEL={num_parallel} \\
    docker://ollama/ollama:0.18.0 \\
    bash -c '
        ollama serve &
        server_pid=$!

        sleep 10
        ollama pull "{model_name}"

        wait "$server_pid"
    '
"""

    # Generate temporary SLURM script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        script_path = f.name

    # Submit job
    try:
        result = subprocess.run(["sbatch", "--parsable", script_path],
                                check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"sbatch failed:\n{exc.stderr}") from exc

    job_id = result.stdout.strip().split(";")[0]
    print(f"Submitted SLURM job id: {job_id}, mem={mem}, gpus={gpus}, "
          f"waiting for it to start ...")

    # Wait for job to enter RUNNING state
    start = time.time()
    node = None

    while time.time() - start < timeout_seconds:
        q = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%T|%N|%R"],
                           check=True, text=True, capture_output=True)
        output = q.stdout.strip()
        elapsed = int(time.time() - start)

        if output:
            state, node_field, reason = output.split("|", 2)
            if state == "RUNNING":
                node = node_field
                print(f"Job RUNNING on node: {node}")
                break
            if state in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                         "OUT_OF_MEMORY"}:
                raise RuntimeError(f"SLURM job {job_id} entered state={state}; "
                                   f"reason={reason}. Log: {log_path}")
            print(f"  [{elapsed}s] state={state}, reason={reason}, waiting ...",
                  flush=True)
        else:
            # Job may already have exited and disappeared from squeue.
            status = subprocess.run(
                ["sacct", "-j", job_id, "--noheader", "--parsable2",
                 "--format=State,ExitCode"],
                check=False, text=True, capture_output=True)
            status_text = status.stdout.strip()
            if status_text:
                raise RuntimeError(f"Job {job_id} left squeue before becoming "
                                   f"ready. Status:\n{status_text}\n"
                                   f"Log: {log_path}")
            print(f"  [{elapsed}s] job not currently visible in squeue",
                  flush=True)

        time.sleep(3)
    else:
        subprocess.run(["scancel", job_id], check=False)
        raise TimeoutError(f"Job {job_id} did not start within "
                           f"{timeout_seconds} seconds and was cancelled. "
                           f"Check SLURM reason and log: {log_path}")

    # Wait for Ollama API and requested model
    ollama_url = f"http://{node}:{port}"
    print(f"Waiting for Ollama API at {ollama_url} ...")

    while time.time() - start < timeout_seconds:
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m.get("name", "") for m in response.json().get("models", [])]
                if model_name in models or any(model_name in n for n in models):
                    print(f"LLM server ready: {ollama_url}; model={model_name}")
                    return {"model_name": model_name, "ollama_url": ollama_url,
                            "num_parallel": num_parallel, "job_id": job_id,
                            "node": node, "mem": mem, "gpus": gpus}
                print(f"Ollama is running, but {model_name} is not loaded yet. "
                      f"Available models: {models}", flush=True)
        except requests.exceptions.RequestException:
            pass

        time.sleep(5)

    subprocess.run(["scancel", job_id], check=False)
    raise TimeoutError(f"Ollama did not become ready within {timeout_seconds} "
                       f"seconds. Job {job_id} was cancelled. Log: {log_path}")


def StopLLMServer(job_id):
    """Cancel an LLM server job launched by LaunchLLMServer."""
    import subprocess
    subprocess.run(["scancel", str(job_id)], check=True)
    print(f"Cancelled job {job_id}")

def LLMConfig(model_name="gpt-oss:20b",
              ollama_url="http://127.0.0.1:12321",
              num_parallel=1,
              temperature=0.7,
              ):
    """
    Build a single-LLM config dict for one or all agents.
    
    Args:
        model_name: Ollama model identifier (e.g., 'gpt-oss:20b').
        ollama_url: Ollama server endpoint, including scheme and port.
        temperature: LLM sampling temperature.
    
    Returns:
        dict with keys 'model_name', 'ollama_url', 'temperature'.
    """
    return {
        "model_name": model_name,
        "ollama_url": ollama_url,
        "temperature": temperature,
        "num_parallel": num_parallel,
    }

def StopLLMServer(job_id):
    """Cancel an LLM server job launched by LaunchLLMServer."""
    import subprocess
    subprocess.run(["scancel", str(job_id)], check=True)
    print(f"Cancelled job {job_id}")