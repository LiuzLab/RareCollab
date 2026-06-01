import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from ._lib._references import (
    AimReferences, PhrankReferences, OmimReferences,
    GnomadReferences, VepReferences, SingularityImages,
    FastaReferences, RareCollabReferences,
)

def _get_available_cpus():
    """
    Detect how many CPU cores the current process can use.
    Falls back gracefully on platforms without sched_getaffinity.
    """
    try:
        # Linux: most accurate (respects cgroups, taskset, SLURM, etc.)
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # macOS/Windows: returns hardware count, may overestimate in containers
        return os.cpu_count() or 1
        
def RecommendWorkerConfig(samplesheet):
    """
    Auto-recommend parallelism settings based on available CPUs and number of
    samples. Returns a dict you can splat into GENERATE_SINGLETON_FEATURES.
    """
    n_cpus = _get_available_cpus()
    n_samples = len(samplesheet)

    split_workers = min(n_samples, n_cpus, 12)
    vep_fork = 12
    vep_workers = max(1, min(n_samples, n_cpus // vep_fork))
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
            mane_transcript=ref_dir / "rarecollab" / "mane_transcript.feather",
            moe_model=ref_dir / "rarecollab" / "MoE_finalized.pt",
            clinvar_feather=ref_dir / "rarecollab" / "ClinVarVCF.feather", 
            omim_disease=ref_dir / "rarecollab" / "OMIM_Disease_Description.tsv",
            hpo_lib=ref_dir / "rarecollab" / "hp.obo",
            hpo_genes=ref_dir / "rarecollab" / "HPO_genes_to_phenotype.txt",
        ),

        aim_bin_dir=aim_bin_dir
    )

    _validate_reference_inputs(references)

    return references

def CheckRequiredTools(mute = False):
    required_tools = {
        "samtools": "conda install -y -c conda-forge -c bioconda samtools",
        "gatk": "conda install -y -c conda-forge -c bioconda gatk4",
        "wget": "conda install -y -c conda-forge wget",
        "gunzip": "conda install -y -c conda-forge gzip",
        "file": "conda install -y -c conda-forge file",
        "bgzip": "conda install -y -c conda-forge -c bioconda htslib",
        "tabix": "conda install -y -c conda-forge -c bioconda htslib",
        "bcftools": "conda install -y -c conda-forge -c bioconda bcftools",
        "singularity": '''conda install -y -c conda-forge "singularity>=3.7"'''
    }

    if not mute:
        print("Python executable:", sys.executable)
        print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
        print()

    missing_tools = {}

    for tool, install_command in required_tools.items():
        tool_path = shutil.which(tool)

        if tool_path is None:
            missing_tools[tool] = install_command
            print(f"{tool}: NOT FOUND")
        else:
            if not mute:
                print(f"{tool}: {tool_path}")

    if not missing_tools:
        if not mute:
            print("\nAll required command-line tools are available.")
        return True

    print("\nSome required command-line tools are missing.")
    print("Please open your Terminal and activate the same conda environment used by this Jupyter kernel:")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env_name = os.path.basename(conda_prefix)
        print(f"\n  conda activate {env_name}")
    else:
        print("\n  conda activate <your_environment_name>")

    print("\nThen install the missing tools:")
    for tool, install_command in missing_tools.items():
        print(f"  # {tool}")
        print(f"  {install_command}")

    print("\nAfter installation, restart the Jupyter kernel and run this check again.")

    return False

def BuildReferenceIndex(ref_ver, ref_dir):
    """
    Build/check FASTA index files. The cache directory is persistent across
    batches; place it next to the reference bundle (e.g. ref_dir/ref_human_genome).
    """
    if not CheckRequiredTools(mute = True):
        print("\nReference index construction stopped because required tools are missing.")
        return

    if ref_ver not in {"hg19", "hg38"}:
        raise ValueError(f"Unsupported ref_ver: {ref_ver}. Expected 'hg19' or 'hg38'.")

    ref_index_dir = Path(ref_dir) / "ref_human_genome"
    ref_index_dir.mkdir(parents=True, exist_ok=True)

    final_fasta = ref_index_dir / f"final_{ref_ver}.fa"
    final_fai = ref_index_dir / f"final_{ref_ver}.fa.fai"
    final_dict = ref_index_dir / f"final_{ref_ver}.dict"

    print('Checking fasta files ...')
    if final_fasta.exists() and final_fai.exists() and final_dict.exists():
        print('*Files Found*')
        return FastaReferences(
            fasta=final_fasta,
            fai=final_fai,
            dict=final_dict,
        )

    raw_fasta_gz = ref_index_dir / f"{ref_ver}.fa.gz"
    raw_fasta = ref_index_dir / f"{ref_ver}.fa"
    num_prefix_fasta = ref_index_dir / f"num_prefix_{ref_ver}.fa"

    fasta_url = (
        f"http://hgdownload.soe.ucsc.edu/goldenPath/"
        f"{ref_ver}/bigZips/{ref_ver}.fa.gz"
    )

    print('Files Not Found in cache dir')
    print('Downloading fasta file ...')
    subprocess.run(
        f"wget --quiet -O {raw_fasta_gz} {fasta_url}",
        shell=True,
        check=True,
    )

    print('Unzipping fasta file ...')
    subprocess.run(
        f"gunzip -c {raw_fasta_gz} > {raw_fasta}",
        shell=True,
        check=True,
    )

    print('Processing fasta file 1/2...')
    subprocess.run(
        f"sed 's/>chr/>/g' {raw_fasta} > {num_prefix_fasta}",
        shell=True,
        check=True,
    )

    print('Processing fasta file 2/2...')
    chromosomes = " ".join(
        [str(i) for i in range(1, 23)] + ["X", "Y", "M"]
    )

    print('Genrating .fai file ...')
    subprocess.run(
        f"samtools faidx {num_prefix_fasta} {chromosomes} > {final_fasta}",
        shell=True,
        check=True,
    )

    subprocess.run(
        f"samtools faidx {final_fasta}",
        shell=True,
        check=True,
    )

    print('Gatk Creating Sequence Dictionary ...')
    subprocess.run(
        f"gatk CreateSequenceDictionary -R {final_fasta}",
        shell=True,
        check=True,
    )

    if not final_fasta.exists():
        raise FileNotFoundError(f"Failed to create FASTA: {final_fasta}")
    if not final_fai.exists():
        raise FileNotFoundError(f"Failed to create FASTA index: {final_fai}")
    if not final_dict.exists():
        raise FileNotFoundError(f"Failed to create sequence dictionary: {final_dict}")

    return FastaReferences(
        fasta=final_fasta,
        fai=final_fai,
        dict=final_dict,
    )

def LoadSamplesheet(csv_path, fulfill_empty_hpo=False):
    """
    Load and validate samplesheet.

    Required columns:
        - sampleID
        - vcf_path
        - hpo_path

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

    return df

def LaunchLLMServer(
    partition="a100q",
    nodelist=None,                  # 可选指定 node
    port=12321,
    num_parallel=2,
    model_name="gpt-oss:20b",
    job_name="ollama_server",
    log_dir="/home/jiashengw/RareCollab",
    timeout_seconds=300,
):
    """
    Launch an Ollama LLM server via SLURM + Singularity, wait until ready,
    and return its connection URL.
    
    The server runs as a persistent background SLURM job; cancel it later
    with `scancel <job_id>` or `StopLLMServer(job_id)`.
    
    Returns:
        {"ollama_url": "...", "job_id": "...", "node": "..."}
    """
    import tempfile, subprocess, time, requests
    from pathlib import Path
    
    # 1. Generate SLURM script
    nodelist_line = f"#SBATCH --nodelist={nodelist}" if nodelist else ""
    log_path = Path(log_dir) / f"{job_name}.log"
    
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{nodelist_line}
#SBATCH --output={log_path}

module load singularity

singularity exec --nv \\
--env OLLAMA_HOST=0.0.0.0:{port} \\
--env OLLAMA_NUM_PARALLEL={num_parallel} \\
docker://ollama/ollama \\
bash -c "ollama serve & sleep 10 && ollama pull {model_name} && wait"
"""
    
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False,
    ) as f:
        f.write(script)
        script_path = f.name
    
    # 2. Submit job
    result = subprocess.run(
        ["sbatch", "--parsable", script_path],
        check=True, text=True, capture_output=True,
    )
    job_id = result.stdout.strip().split(";")[0]
    print(f"Submitted SLURM job id: {job_id}, waiting for it to start ...")
    
    # 3. Wait for RUNNING + get node
    start = time.time()
    node = None
    while time.time() - start < timeout_seconds:
        q = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T %N"],
            check=True, text=True, capture_output=True,
        )
        if q.stdout.strip():
            parts = q.stdout.strip().split(None, 1)
            state = parts[0]
            if state == "RUNNING" and len(parts) == 2:
                node = parts[1]
                print(f"Job RUNNING on node: {node}")
                break
            else:
                elapsed = int(time.time() - start)
                print(f"  [{elapsed}s] state={state}, waiting ...", flush=True)
        else:
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] job not in squeue", flush=True)
        time.sleep(3)
    else:
        # Timeout — cancel the orphan job to avoid wasted GPU time
        try:
            subprocess.run(["scancel", job_id], check=False)
            print(f"Cancelled orphan job {job_id}")
        except Exception:
            pass
        raise TimeoutError(
            f"Job {job_id} did not start within {timeout_seconds}s. "
            f"Check `{log_path}` for errors."
        )
    
    # 4. Wait for ollama API ready
    ollama_url = f"http://{node}:{port}"
    print(f"Waiting for ollama API at {ollama_url} ...")
    while time.time() - start < timeout_seconds:
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if model_name in models or any(model_name in m for m in models):
                    print(f"✅ LLM server ready: {ollama_url}, model {model_name} loaded")
                    return {
                        "model_name": model_name,
                        "ollama_url": ollama_url,
                        "num_parallel": num_parallel,
                        "job_id": job_id,
                        "node": node,
                    }
                print(f"  ollama running but model {model_name} not yet loaded, "
                      f"available: {models}")
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    
    # Timeout on API ready — cancel job too
    try:
        subprocess.run(["scancel", job_id], check=False)
        print(f"Cancelled job {job_id} (API never ready)")
    except Exception:
        pass
    raise TimeoutError(
        f"LLM server did not become ready within {timeout_seconds}s. "
        f"Check `{log_path}` for errors."
    )


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