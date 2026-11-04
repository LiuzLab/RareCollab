<div align="center">
  <img width="766" height="219" alt="RareCollab" src="https://github.com/user-attachments/assets/77a103c9-752a-4529-8f75-914c740f2d4c" />
</div>

# Introduction

**RareCollab** is a Python package for rare diseases powered by **Ollama**. It integrates multimodal patient data, including **DNA**, **RNA**, and **phenotype** information, to support candidate variant prioritization and diagnostic interpretation.

Our paper is available on **arXiv**: <https://arxiv.org/abs/2602.04058>

## Before Running RareCollab

Before running RareCollab, please prepare the following files and directories:

1. **Download the RareCollab data dependencies.** No AWS account is needed — `--no-sign-request` makes the request anonymous. Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) first if you do not have it, then:

   ```bash
   # Change to the directory where you want the data
   cd <desired/folder/path>

   # Download
   aws s3 sync s3://rarecollab-data-dependencies-1.0-public/ . --no-sign-request
   ```

   The directory you download into is the `ref_dir` passed to
   `RareCollab.Setup.ResolveReferenceInputs` in Step 1.

2. **Create** an NCBI API key using your email address from [NCBI API Keys](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/). This step is optional, but it can speed up searches using the NCBI and Entrez APIs.

# Usage

## Installation

Run the code below to install RareCollab:

```python
%pip install git+https://github.com/LiuzLab/RareCollab.git
```

---

## Step 1. Setup

### 1.1 Check required command-line tools

```python
import RareCollab

RareCollab.Setup.CheckRequiredTools()
```

Follow the on-screen instructions to install any missing tools into your target environment.

> [!TIP]
> When you see the message `All required command-line tools are available.`, all external dependencies have been correctly installed and you can proceed to the next step.

### 1.2 Configure the paths

```python
# Path to the reference data dependencies you downloaded
ref_dir = '/path/RareCollab-data-dependencies-1.0'

# Path to your working folder (will be created if it does not exist)
work_dir = '/path/work'

#Path to save the final results (will be created if it does not exist)
output_path = '/path/output'

# Reference genome build: 'hg38' or 'hg19'
ref_ver = 'hg38'
```

### 1.3 Prepare the references and container images

This step locates the reference files, builds the required indexes, and sets up the Singularity images that the pipeline will use:

```python
# Locate and validate the reference input files for the chosen genome build
references = RareCollab.Setup.ResolveReferenceInputs(ref_dir=ref_dir, ref_ver=ref_ver)

# Build the FASTA index files required for downstream analysis
fasta_references = RareCollab.Setup.BuildReferenceIndex(ref_dir=ref_dir, ref_ver=ref_ver)

# Pull and prepare the Singularity images for the required tools
singularity_images = RareCollab.Setup.PrepareSingularityImages(ref_dir=ref_dir)
```

Each call returns a handle (`references`, `fasta_references`, `singularity_images`) that later steps will use, so run all three before continuing.

### 1.4 Load the samplesheet

```python
# Load the samplesheet and validate its configuration
samplesheet = RareCollab.Setup.LoadSamplesheet(csv_path='/path/samplesheet.csv', fulfill_empty_hpo=False)
```

The samplesheet is a **CSV** with three required columns: `sampleID`, `vcf_path`, `hpo_path`.

<details>
<summary>Column requirements</summary>

| Column | Description |
| --- | --- |
| `sampleID` | A unique identifier for each sample. Must not be duplicated. Letters, digits, hyphens (`-`), and underscores (`_`) are safe to use. To avoid potential parsing issues, it is best to avoid other special characters (such as spaces, `/`, `\`, `*`, or `#`). |
| `vcf_path` | The absolute path to the sample's VCF file. |
| `hpo_path` | The absolute path to the sample's HPO file. |

</details>

<details>
<summary>HPO file format and <code>fulfill_empty_hpo</code></summary>

Each HPO file is a plain-text (`.txt`) file containing a list of HPO terms, one per line, in the form `HP:XXXXXXX`.

When `fulfill_empty_hpo=True`, if no HPO file exists at the specified `hpo_path`, a default HPO file will be created automatically at that location.

</details>

<details>
<summary>Optional RNA columns</summary>

RNA analysis is entirely optional — DNA-only analysis is fully supported. To include RNA evidence, add `rna_path`, `strand` and `pairedEnd` to the samplesheet.

| Column | Description |
| --- | --- |
| `rna_path` | The absolute path to the sample's RNA-seq BAM file. Leave the cell empty for samples without RNA; the column may also be omitted entirely to run DNA only. |
| `strand` | The library's strandedness. One of `unstranded` (not strand-specific), `forward` (read 1 in the same orientation as the transcript, fr-secondstrand), `reverse` (read 1 opposite to the transcript, fr-firststrand/dUTP — **the most common protocol for human RNA-seq**, including TruSeq Stranded mRNA), or `auto` (determine it from the BAM). |
| `pairedEnd` | Whether the library is paired-end. One of `yes`, `no`, or `auto` (determine it from the BAM). |

Both `strand` and `pairedEnd` are case-insensitive. An empty cell is treated as `auto`, and so is an absent column — the pipeline adds it for you. A value that is not recognised is replaced with `auto` and reported as a warning rather than raising, so a typo in an optional column does not stop a run partway through.

The two columns are resolved differently when your entry and the file disagree. `pairedEnd` is measured from the BAM regardless of what you enter, because it is read directly from the alignment flags and takes no extra work — if your entry disagrees with the file, the BAM wins and the difference is reported. `strand` behaves the other way round: an explicit value is always respected, and detection is only used to flag a possible discrepancy.

</details>

<details>
<summary>Demo files in <code>demo/</code></summary>

The repository ships four examples you can use as a reference for the expected format:

- a demo **HPO file**
- a demo **samplesheet for DNA-only analysis**
- a demo **samplesheet including RNA**
- a demo **RNA background samplesheet**, for supplying additional RNA samples as controls when your own cohort is too small (see Section X.X)

</details>

---

### 1.5 Configure the worker settings

```python
# Automatically recommend parallelization settings based on the samplesheet
config = RareCollab.Setup.RecommendWorkerConfig(samplesheet)
```

<details>
<summary>Adjusting <code>config</code></summary>

**At your own risk.** You can adjust the parallelization settings in `config` if needed, but overriding the recommended values may lead to excessive resource usage or unstable runs.

</details>

---

## Step 2. Generate Features

### Step 2.1. DNA Features from VCF Files
This step turns each sample's VCF into the features used by the downstream analysis. It runs in two stages — first processing the VCFs, then generating the features — and each stage updates `samplesheet` with the results:

```python
# Stage 1: Process the VCF files (split, normalize, etc.)
samplesheet = RareCollab.Features.ProcessVCF(
    samplesheet,
    config=config,
    work_dir=work_dir,
    references=references,
    fasta_references=fasta_references,
    overwrite=False,
)

# Stage 2: Generate features from the processed VCFs
samplesheet = RareCollab.Features.GenerateFeatures(
    samplesheet,
    work_dir=work_dir,
    references=references,
    fasta_references=fasta_references,
    singularity_images=singularity_images,
    ref_ver=ref_ver,
    config=config,
    overwrite=False,
)
```

Run both calls in order, since `GenerateFeatures` depends on the output of `ProcessVCF`.

<details>
<summary>Resuming an interrupted run</summary>

With `overwrite=False`, samples whose results already exist in `work_dir` are skipped, so you can safely re-run this step to resume an interrupted run. Set `overwrite=True` to force every sample to be reprocessed from scratch.

</details>

### Step 2.2. RNA Features from BAM Files (Optional)

If RNA-seq data is available, RareCollab can add splicing, expression and
allele-specific expression evidence to the diagnosis. This whole step is
optional: skip it and the pipeline runs on DNA alone.

#### 2.2.1 Background RNA samples

FRASER and OUTRIDER fit a single model across the entire cohort — they detect
how one sample deviates from the others, so a sample has nothing to be an
outlier against on its own. DROP recommends at least **50 samples** of the same
tissue and library protocol. Rare-disease batches are usually far smaller, so
additional RNA samples can be supplied purely as background to make the models
fittable.

```python
rna_background = RareCollab.Setup.LoadRNABackground(
    csv_path='/path/.../BackgroundRNA.csv',
    existing_sample_ids=samplesheet['sampleID'],
)
```

If your own cohort is already large enough, skip this and pass `None`:

```python
rna_background = None
```

<details>
<summary>Background samplesheet format</summary>

The background samplesheet requires `sampleID` and `rna_path`; `strand` and
`pairedEnd` are optional and follow the same rules as in the main samplesheet.
Rows whose BAM does not exist are dropped with a note rather than raising, and
background sample IDs must not collide with those in the main samplesheet.

The returned table is an ordinary DataFrame, so a background sample that turns
out to be unusable can be removed before it reaches the models:

```python
rna_background = rna_background[
    rna_background["sampleID"] != "<sample to drop>"
].reset_index(drop=True)
```

</details>

> [!TIP]
> Background samples should match the cases in tissue and library protocol; otherwise the outliers detected mostly reflect those differences.

#### 2.2.2 Characterise the BAM files

```python
rna_cohort = RareCollab.Features.ProcessBAM(
    samplesheet=samplesheet,
    rna_background=rna_background,
    work_path=work_dir,
    fasta_references=fasta_references,
    overwrite=False,
)
```

This builds a `.bai` index where one is missing, determines each library's
strandedness and paired-end status, notes whether the BAM uses `chr1`- or
`1`-style contig names, and combines cases and background into a single cohort
table. Results are cached per sample, so later runs reuse them unless the BAM
itself has changed.

The returned `rna_cohort` is the table every analysis below consumes.

<details>
<summary>Why <code>fasta_references</code> matters here</summary>

Passing `fasta_references` turns on a normalisation pass over the case BAMs.
GATK sorts intervals into the reference's contig order and then walks the BAM
forward only, so a BAM ordered differently loses every contig that sits earlier
in its own ordering than the one before it — a lexicographically ordered BAM
drops chr10 through chr22, exits cleanly, and says nothing. Contigs the
reference lacks are a second problem, aborting the run hours in.

Any case BAM whose contigs disagree with the reference is rewritten with
`gatk ReorderSam` into `BAMInfo/<sampleID>/<sampleID>.normalised.bam`, and
`rna_path` is repointed at the copy. A BAM that already matches is left alone.
Background samples are not normalised: FRASER and OUTRIDER do not care about
contig order.

Omit `fasta_references` to skip the pass entirely.

</details>

#### 2.2.3 Run the analyses

Run ASE first. It compares the two alleles **within** one patient at that
patient's own heterozygous sites, so it needs no background, does not depend on
which samples it is analysed alongside, and is computed once per patient
whatever else happens afterwards.

```python
# Allele-specific expression
rna_cohort = RareCollab.Features.RunASE(
    rna_samples=rna_cohort,
    work_path=work_dir,
    fasta_references=fasta_references,
    overwrite=False,
)
```

FRASER and OUTRIDER are cohort methods, and measure each case against the
background:

```python
# Aberrant splicing
rna_cohort = RareCollab.Features.RunFRASER(
    rna_cohort=rna_cohort,
    work_path=work_dir,
    references=references,
    overwrite=False,
)

# Expression outliers
rna_cohort = RareCollab.Features.RunOutrider(
    rna_cohort=rna_cohort,
    work_path=work_dir,
    references=references,
    overwrite=False,
)
```

Each call returns `rna_cohort` with one result-path column appended, so a single
variable carries through the whole step. No sample is ever dropped from the
table, so a failure in one analysis does not remove that patient from the next.
Run only the analyses you want — each is independent of the others.

<details>
<summary>Controlling parallelism</summary>

`RunFRASER` counts reads per sample and then fits a model across the cohort, and
the two stages scale differently:

| Argument | Stage |
| --- | --- |
| `count_workers` | Samples counted at once. Bound by memory — each worker holds one BAM's worth. |
| `fit_threads` | Threads for the cohort fit. Bound by cores. |
| `n_threads` | Sets both at once, when you do not want to separate them. |

`RunOutrider` takes `n_threads` alone. Leave all of them unset to let
RareCollab derive values from the machine; the defaults are deliberately
conservative, so a large machine is usually worth a explicit value.

Read counting is cached per sample and shared across analyses, so a sample
already counted is never counted again — including by a later cohort that
reuses the same background.

</details>

#### 2.2.4 Attach the RNA evidence to the samplesheet

```python
rna_evidence = RareCollab.Features.PrepareRNAEvidence(
    rna_cohort=rna_cohort,
    samplesheet=samplesheet,
    work_path=work_dir,
)

samplesheet = RareCollab.Features.MergeRNAEvidence(
    samplesheet, rna_evidence, work_path=work_dir,
)
```

`PrepareRNAEvidence` reshapes each analysis output into the form the diagnostic
engine reads and records the paths; `MergeRNAEvidence` puts those rows back into
the full samplesheet and writes `samplesheet_with_paths.csv`. Whatever exists is
used: if only one or two of the three analyses were run, or one failed for a
single patient, the remaining evidence still comes through and the missing
columns are simply left empty. Patients with no RNA at all pass through
untouched.

<details>
<summary>Running several case/background combinations</summary>

A patient measured against two different background cohorts gets two different
FRASER and OUTRIDER answers, and both are worth keeping — so each combination is
run under its own `analysis_name`, and the results are merged at the end.

ASE is unaffected: it is per-patient, so it is still run **once**, before any
splitting, exactly as above.

Split the cohort table into one case plus the shared background per analysis:

```python
import pandas as pd

controls = rna_cohort[rna_cohort["sample_type"] == "control"]

cohort_a = pd.concat(
    [rna_cohort[rna_cohort["sampleID"] == "<case A>"], controls],
    ignore_index=True)
cohort_b = pd.concat(
    [rna_cohort[rna_cohort["sampleID"] == "<case B>"], controls],
    ignore_index=True)
```

Run the cohort methods on each, naming the analysis:

```python
for cohort, name in ((cohort_a, "case_a"), (cohort_b, "case_b")):
    cohort = RareCollab.Features.RunFRASER(
        rna_cohort=cohort, work_path=work_dir, references=references,
        analysis_name=name, count_workers=12, fit_threads=32)
    cohort = RareCollab.Features.RunOutrider(
        rna_cohort=cohort, work_path=work_dir, references=references,
        analysis_name=name, n_threads=32)
```

Prepare each separately, then merge them all in one call:

```python
evidence_a = RareCollab.Features.PrepareRNAEvidence(
    rna_cohort=cohort_a, samplesheet=samplesheet,
    work_path=work_dir, analysis_name="case_a")

evidence_b = RareCollab.Features.PrepareRNAEvidence(
    rna_cohort=cohort_b, samplesheet=samplesheet,
    work_path=work_dir, analysis_name="case_b")

samplesheet = RareCollab.Features.MergeRNAEvidence(
    samplesheet, evidence_a, evidence_b, work_path=work_dir)
```

The merged samplesheet holds **one row per (patient, analysis)**. A patient who
appears in two analyses occupies two rows carrying two sets of RNA paths;
patients with no RNA keep their single row. Everything downstream reads
`analysis_name` off the row, so the whole samplesheet goes through the rest of
the pipeline in one pass rather than once per analysis.

`analysis_name` defaults to `my_work`, which writes into the same directories
earlier versions used — a cohort that is never split never has to know the
concept exists.

</details>

---

## Step 3. Score variants and select candidates

With DNA features generated — and RNA evidence attached, if any — the diagnostic
engine scores every variant and narrows the list down to the candidates worth an
expert opinion.

```python
# Run the Mixture-of-Experts (MoE) diagnostic engine
samplesheet = RareCollab.DiagnosticEngine.MoE(
    samplesheet=samplesheet,
    work_dir=work_dir,
    references=references,
)

# Generate the candidate gene/variant list
samplesheet = RareCollab.DiagnosticEngine.Candidates(
    samplesheet=samplesheet,
    work_dir=work_dir,
    config=config,
    overwrite=False,
)
```

---
## Step 4. Run RareCollab Agents

### 4.1 Launch the LLM Server

The agents talk to an **Ollama** server over HTTP. Start one by whichever route
fits your machine, then capture its connection details into `llm_config` — the
rest of the pipeline only ever sees `llm_config`.

<details>
<summary>Option A — on a SLURM cluster</summary>

```python
server = RareCollab.Setup.LaunchLLMServer(
    partition="partition",
    nodelist="node",
    mem="64G",
    port=12321,
    num_parallel=2,
    model_name="gpt-oss:20b",
    job_name="ollama_server",
    log_dir=work_dir,
    cpus_per_task=1,
    gpus=1,
    timeout_seconds=300,
)
```

<details>
<summary>Parameters</summary>

| Parameter | Type | Description |
| --- | --- | --- |
| `partition` | *str* | The SLURM partition to launch the server on. |
| `nodelist` | *str* | A specific node to run on. Leave it unset to let SLURM pick any node in the partition — pinning to one node means queueing behind whoever is already using it. |
| `port` | *int* | The port the server listens on. Change it if the default is already taken on that node. |
| `num_parallel` | *int* | How many requests the model serves concurrently. **This is the throughput dial**: it sets `OLLAMA_NUM_PARALLEL` on the server and is also returned in `llm_config`, where the agents use it as their worker count. Each extra slot costs a little GPU memory for its KV cache, so on a 40–80 GB card a 20B model comfortably supports 8 or more. |
| `model_name` | *str* | The model to serve (e.g. `gpt-oss:20b`). Pulled on first launch, which for a 20B model takes several minutes. |
| `job_name` | *str* | SLURM job name, also used for the log filename. |
| `log_dir` | *str* | Where to write `<job_name>.log`. The server's own output goes here, and it is the first place to look if startup fails. |
| `timeout_seconds` | *int* | How long to wait for the job to start **and** for the model to become ready. Raise it on a busy queue, or when the model has not been pulled on that node before. |
| `mem` | *str* | SLURM system-memory request, for example `"64G"`. This is **CPU RAM, not GPU VRAM** — model weights live in GPU memory, which is fixed by the card. |
| `cpus_per_task` | *int* | CPU cores allocated to the server process. |
| `gpus` | *int* | GPUs requested via `--gres=gpu:N`. One is enough for a 20B model at 4-bit (~13 GB); more only helps when the weights do not fit on a single card, and does nothing for throughput. |

</details>

`LaunchLLMServer` prints its SLURM job ID at startup, for example
`Submitted SLURM job id: xxxxxxxx`. Keep it — you need it to reconnect to the
server later, and to stop it.

</details>

<details>
<summary>Option B — on a machine with its own GPU</summary>

No SLURM needed. This starts Ollama on the machine running the notebook, pulls
the model if it is missing, and waits until it answers:

```python
server = RareCollab.Setup.LaunchLocalLLMServer(
    model_name="gpt-oss:20b",
    num_parallel=2,
    log_dir=work_dir,
)
```

A server already listening on the port is reused rather than replaced. The
server is started detached, so it survives a kernel restart.

</details>

<details>
<summary>Option C — a server that is already running</summary>

If someone else maintains the server, or you started one by hand, point at it
directly. `CheckLLMServer` confirms it answers and holds the model, which is
worth doing before a run rather than discovering it one failed request at a
time, hours in:

```python
RareCollab.Setup.CheckLLMServer("http://<host>:<port>", "gpt-oss:20b")

server = {
    "model_name": "gpt-oss:20b",
    "ollama_url": "http://<host>:<port>",
    "num_parallel": 2,
}
```

</details>

#### Build `llm_config`

Whichever option you used, `server` has the same shape:

```python
llm_config = RareCollab.Setup.LLMConfig(
    model_name=server["model_name"],
    ollama_url=server["ollama_url"],
    num_parallel=server["num_parallel"],
    temperature=0.7,
)
```

<details>
<summary>Parameters — <code>LLMConfig</code></summary>

| Parameter | Type | Description |
| --- | --- | --- |
| `model_name` | *str* | The served model name. Pass through from `server["model_name"]`. |
| `ollama_url` | *str* | The server's URL. Pass through from `server["ollama_url"]`. |
| `num_parallel` | *int* | Number of parallel requests. Pass through from `server["num_parallel"]`. |
| `temperature` | *float* | Sampling temperature. A value of `0.7` is recommended. |

The agents call Ollama's native `/api/generate` endpoint, so the server has to
be Ollama. An OpenAI-compatible endpoint — vLLM, a hosted API — will not work
without changes to the agent code.

</details>

<details>
<summary>Reconnecting after a kernel restart</summary>

Restarting the notebook loses the `server` variable while the job itself keeps
serving. Rebuild the handle from the SLURM job ID instead of relaunching:

```python
server = RareCollab.Setup.ReconnectLLMServer(
    job_id=<SLURM job id>,
    port=12321,
    model_name="gpt-oss:20b",
    num_parallel=2,
)
```

The node is read back from `squeue`, so a requeued job is still found.
`num_parallel` cannot be read back from a running server — pass the value it
was launched with. A larger number here does not make the server faster; it
makes the agents queue, which looks like slowness with a clean log.

</details>

#### Shut the server down

```python
RareCollab.Setup.StopLLMServer(server)
```

Pass the `server` dict from any of the options above. A server that was already
running when RareCollab found it is left alone — adopting one does not make it
RareCollab's to kill.

### 4.2 Run the Diagnostic Agents (Serial)

If you have only a single LLM available, run the diagnostic agents **serially**
as shown below. Each agent updates `samplesheet` and passes it to the next one,
so they must be run in order.

We recommend providing an NCBI email and API key — they're used by the database
and literature agents. If you don't have them, set both to `None`.

```python
# NCBI credentials — used by the database and literature agents.
# If you don't have them, set both to None (queries may then be rate-limited).
NCBI_EMAIL = "your_ncbi@email.com"   # or None
NCBI_KEY   = "your-api-key"          # or None

# Database agent: query external databases for each candidate (uses NCBI)
samplesheet = RareCollab.DatabaseAgent.RunAgent(
    samplesheet=samplesheet,
    work_dir=work_dir,
    references=references,
    llm_config=llm_config,
    ncbi_email=NCBI_EMAIL,
    ncbi_api_key=NCBI_KEY,
    config=config,
    overwrite=False,
)

# In-silico agent: run in-silico prediction/analysis on the candidates
samplesheet = RareCollab.InSilicoAgent.RunAgent(
    samplesheet=samplesheet,
    work_dir=work_dir,
    llm_config=llm_config,
    overwrite=False,
)

# Phenotype agent: preprocess the phenotype (HPO) data
samplesheet = RareCollab.PhenotypeAgent.Preprocessing(
    samplesheet=samplesheet,
    work_dir=work_dir,
    references=references,
    overwrite=False,
)

# Phenotype agent: analysis based on HPO terms
samplesheet = RareCollab.PhenotypeAgent.RunAgent_HPO(
    samplesheet=samplesheet,
    work_dir=work_dir,
    llm_config=llm_config,
    overwrite=False,
)

# Phenotype agent: analysis against OMIM
samplesheet = RareCollab.PhenotypeAgent.RunAgent_OMIM(
    samplesheet=samplesheet,
    work_dir=work_dir,
    llm_config=llm_config,
    overwrite=False,
)

# Phenotype agent: analysis from the literature (uses NCBI)
samplesheet = RareCollab.PhenotypeAgent.RunAgent_Literature(
    samplesheet=samplesheet,
    work_dir=work_dir,
    llm_config=llm_config,
    ncbi_email=NCBI_EMAIL,
    ncbi_api_key=NCBI_KEY,
    overwrite=False,
)
```

#### RNA agent

If you ran Step 2.2, add the RNA agent. It reads each patient's splicing,
expression and allele-specific expression evidence and gives a gene-level and a
variant-level reading of it.

```python
samplesheet = RareCollab.RNAAgent.RunAgent(
    samplesheet=samplesheet,
    work_dir=work_dir,
    llm_config=llm_config,
    references=references,
    ref_ver=ref_ver,
    overwrite=False,
)
```

---

## Step 5. Integrate the Results

The final step merges the outputs from all the diagnostic agents into a single integrated result and writes it to `output_path`:

```python
# Merge all agent outputs into the final integrated result
samplesheet = RareCollab.Integration.Review(
    samplesheet=samplesheet,
    work_dir=work_dir,
    fasta_references=fasta_references,
    output_path=output_path,
    overwrite=False,
)
```

After this step completes, your final integrated results are available at `output_path`.

## Output Format

The integrated results are written as **CSV files, one per sample** (split by `identifier` / sampleID). Within each CSV, candidate variants are **already ranked**

<details>
<summary>Output column reference</summary>

The columns in each output CSV are described below.

#### Variant & Gene Identity

| Column | Description |
|---|---|
| `varId` | Variant ID, formatted as `<chrom>_<pos>_<ref>_<alt>`. The `chr` prefix is stripped, and X and Y are written as `23` and `24`. |
| `identifier` | Sample ID. |
| `geneSymbol` | Gene symbol. |
| `HGVSc` | Coding-level (cDNA) variant nomenclature in HGVS format. |
| `HGVSc_core` | Final retained cDNA HGVS string. |
| `HGVSp` | Protein-level variant nomenclature in HGVS format. |
| `transcript_id` | Ensembl transcript ID. |
| `geneSymbol_VarId` | Combined gene-symbol + variant-ID key. |
| `Chromosome` | Chromosome. |
| `Pos` | Variant position. |

#### Inheritance & Variant Class

| Column | Description |
|---|---|
| `dominant` | 1 = dominant, 0 = not dominant. |
| `recessive` | 1 = recessive, 0 = not recessive. |
| `zyg` | Zygosity: 1 = heterozygous, 2 = homozygous, 0 = missing or uncalled. Stored as a float, so it reads as `1.0` / `2.0` / `0.0`. |
| `cons_frameshift_variant` | 1 = frameshift variant, 0 = not. |
| `frame_shift` | Whether the variant causes a frameshift. |
| `transcript_score` | Score for the transcript, computed as a weighted sum based on MANE status and the presence of HGVSp/HGVSc. |

#### Gene Constraint Metrics (gnomAD)

| Column | Description |
|---|---|
| `gnomadGenePLI` | gnomAD pLI score (probability of loss-of-function intolerance). |
| `gnomadGeneOELof` | gnomAD observed/expected ratio for loss-of-function variants. |
| `gnomadGeneOELofUpper` | Upper bound of the gnomAD LOEUF (O/E LoF confidence interval). |

#### Diagnostic Engine (Mixture-of-Experts) Scores

The overall score comes from a Mixture-of-Experts (MoE) model. Each expert module produces a score and a rank.

| Column | Description |
|---|---|
| `overall_logit` | Final-layer logit from the MoE model (interconvertible with `overall_prob`). |
| `overall_prob` | Overall probability (the sigmoid-transformed `overall_logit`). |
| `score_Database` / `rank_Database` | Score / rank from the Database module. |
| `score_Genetics` / `rank_Genetics` | Score / rank from the Genetics module. |
| `score_InSilico` / `rank_InSilico` | Score / rank from the InSilico module. |
| `score_Overview` / `rank_Overview` | Score / rank from the Overview module. |
| `score_Phenotype` / `rank_Phenotype` | Score / rank from the Phenotype module. |
| `Diagnostic_Engine_Rank` | Final rank from the MoE model. |

#### RNA — OUTRIDER (expression outliers)

All `Outrider_*` columns require the RNA module; they are empty if RNA was not run.

| Column | Description |
|---|---|
| `Outrider_pValue` | OUTRIDER p-value. |
| `Outrider_padjust` | OUTRIDER adjusted p-value. |
| `Outrider_zScore` | OUTRIDER z-score (from its fitted distribution). |
| `Outrider_l2f` | Log2 fold change. |
| `Outrider_rawcounts` | Raw read counts. |
| `Outrider_RawZscore` | Z-score computed directly from raw values (differs from `Outrider_zScore` because OUTRIDER's model z-score is not based on a plain normal distribution). |

#### RNA — FRASER 2.0 (splicing outliers)

All `Fraser_*` columns come from FRASER 2.0 and require the RNA module; empty if RNA was not run. Names follow FRASER's own terminology.

| Column | Description |
|---|---|
| `Fraser_GenePvalue` | Gene-level p-value. |
| `Fraser_pvaluesBetaBinomial_jaccard` | Beta-binomial p-value for the Jaccard metric. |
| `Fraser_psi5` / `Fraser_psi3` | PSI values for 5′ / 3′ splice-site usage. |
| `Fraser_rawOtherCounts_psi5` / `Fraser_rawOtherCounts_psi3` | Raw counts of "other" reads for the psi5 / psi3 metrics. |
| `Fraser_rawCountsJnonsplit` | Raw non-split read counts at the junction. |
| `Fraser_jaccard` | Jaccard splicing metric. |
| `Fraser_rawOtherCounts_jaccard` | Raw "other" counts for the Jaccard metric. |
| `Fraser_delta_jaccard` / `Fraser_delta_psi5` / `Fraser_delta_psi3` | Deviation (delta) from expected value for each metric. |
| `Fraser_predictedMeans_jaccard` | Model-predicted mean for the Jaccard metric. |
| `Fraser_junction_start` / `Fraser_junction_end` | Splice junction start / end coordinates. Empty unless the variant physically falls inside an outlier junction, so they are empty for variants selected on expression or ASE evidence alone. |

#### RNA — Allele-Specific Expression (ASE)

`ASE_*` columns come from the ASE analysis; empty if it was not run.

| Column | Description |
|---|---|
| `ASE_REF` / `ASE_ALT` | Reference / alternate allele. |
| `ASE_REF_COUNT` / `ASE_ALT_COUNT` | Read counts supporting the reference / alternate allele. |
| `ASE_ALT_RATIO` | Fraction of reads supporting the alternate allele. |
| `ASE_PVAL` | P-value for allelic imbalance. |
| `IS_MAE` | Whether the variant shows mono-allelic expression. |

#### RNA — Read Support Counts

All `ref_*` / `alt_*` columns are RNA-derived; 0 when the RNA module was not run.

| Column | Description |
|---|---|
| `ref_count_max` / `ref_count_mean` / `ref_count_min` | Max / mean / min reference-allele read counts. |
| `alt_count_max` / `alt_count_mean` / `alt_count_min` | Max / mean / min alternate-allele read counts. |

#### Candidate Selection & Compound Heterozygosity

| Column | Description |
|---|---|
| `is_compound_het_group` | Whether the variant belongs to a compound-het group. |
| `evidence_rules` | Reason(s) the variant was selected as a disease candidate. |
| `find_compound_het` | Whether the variant is part of a compound het. |
| `partner` | The paired variant, if this is a compound het. |
| `compound_het_group_id` | Group ID for the compound-het group, if applicable. |

#### LLM Agent Judgments

These columns hold conclusions from the LLM agents. An **empty** value means the candidate did not meet the prerequisites for that judgment (e.g. missing database info). When populated, conclusions may be one of: supporting, opposing, neutral, relevant, not relevant, insufficient information, etc.

| Column | Description |
|---|---|
| `Database_Reasoning` / `Database_Conclusion` | Reasoning / conclusion from the Database agent. |
| `Database_Zygosity` | Zygosity call from the Database agent. |
| `HPO_Reasoning` / `HPO_Conclusion` | Reasoning / conclusion from the HPO (phenotype) agent. |
| `OMIM_Reasoning` / `OMIM_Conclusion` | Reasoning / conclusion from the OMIM agent. |
| `Literature_Reasoning` / `Literature_Conclusion` | Reasoning / conclusion from the Literature agent. |
| `Insilico_Reasoning` / `Insilico_Conclusion` | Reasoning / conclusion from the in-silico prediction agent. |
| `RNA_GeneLevel_Reasoning` / `RNA_GeneLevel_Event` / `RNA_GeneLevel_Conclusion` | Gene-level reasoning / event / conclusion from the RNA agent. |
| `RNA_VarLevel_Reasoning` / `RNA_VarLevel_Event` / `RNA_VarLevel_Conclusion` | Variant-level reasoning / event / conclusion from the RNA agent. |

#### Flags & Tiering

| Column | Description |
|---|---|
| `has_repeat_motif` | Whether the variant sits in a tandem repeat. Computed from the reference genome sequence, so it is populated whether or not RNA was run. |
| `rna_confirms_call_flag` | An RNA double-check of the DNA call. `True` by default — including when no RNA was run — so it means the call was **not contradicted**, rather than confirmed. Set to `False` only on positive evidence against the call: a mean reference-read count above 20 together with a minimum alternate-read count of 0. Low coverage never demotes. |
| `new_diseasegene_flag` | Whether this is a novel disease gene. |
| `omim_flag` | Whether the gene is an OMIM gene. |
| `strong_nom_flag` | Whether this is a strong candidate. |
| `Tier` | Grouping by evidence strength: 1 = most relevant, 2 = intermediate, 3 = least relevant / least evidence. |
| `new_rank` | Updated per-gene rank. |

#### Pairwise Comparison Module

All `PairWise*` / `pairwise_*` columns come from the pairwise (head-to-head) comparison module.

| Column | Description |
|---|---|
| `PairWiseScore` | Pairwise comparison score. |
| `PairWiseWins` / `PairWiseLosses` / `PairWiseTies` | Number of head-to-head wins / losses / ties. |
| `PairWiseEntityRank` | Final rank from the pairwise comparison. |
| `PairWiseWithinEntityRank` | For a compound het, whether this variant is the higher-ranked (1) or lower-ranked (2) member. |
| `pairwise_entity_id` | Entity ID in the comparison module. |
| `pairwise_entity_type` | Entity type in the comparison module. |

</details>
