#!/usr/bin/env python
# coding: utf-8

import re
import os
import json
import time
import pronto
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from Bio import Entrez, Medline
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

#↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓
#↓-↓-Preprocessing-↓-↓
def p_to_label(p: float) -> str:
    if p == 1.0: return "Obligate"
    if p == 0.0: return "Excluded"
    if 0.80 <= p < 1.0: return "Very frequent"
    if 0.30 <= p < 0.80: return "Frequent"
    if 0.05 <= p < 0.30: return "Occasional"
    if 0.01 <= p < 0.05: return "Very rare"
    return "Very rare"

def to_num_den(freq: str):
    TERM_TO_P = {
        "HP:0040280": 1.00,   # Obligate
        "HP:0040281": 0.90,   # Very frequent
        "HP:0040282": 0.55,   # Frequent
        "HP:0040283": 0.17,   # Occasional
        "HP:0040284": 0.025,  # Very rare
        "HP:0040285": 0.00,   # Excluded
    }
    if freq == "-":
        return 0.0, 0.0, 0
    if freq.startswith("HP:"):
        den = 6.0
        num = TERM_TO_P[freq] * den
        return num, den, 1
    if freq.endswith("%"):
        den = 6.0
        num = float(freq[:-1])/100 * den
        return num, den, 1
    m, n = freq.split("/")
    return float(int(m)), float(int(n)), 1

def build_gene2hpo(df: pd.DataFrame) -> dict[str, str]:
    tmp = df[["gene_symbol", "hpo_id", "hpo_name", "frequency"]].copy()
    parsed = tmp["frequency"].apply(to_num_den)
    tmp["num"] = parsed.map(lambda t: t[0])
    tmp["den"] = parsed.map(lambda t: t[1])
    tmp["has"] = parsed.map(lambda t: t[2])

    grp = tmp.groupby(["gene_symbol", "hpo_id"], as_index=False).agg(
        hpo_name=("hpo_name", "first"),
        num=("num", "sum"),
        den=("den", "sum"),
        has=("has", "max"),
    )

    grp["entry"] = grp.apply(lambda r: f"{r.hpo_name} (Frequency not specified)" if (r.has == 0 or r.den == 0) else f"{r.hpo_name} ({p_to_label(r.num / r.den)})", axis=1)

    return grp.groupby("gene_symbol")["entry"].apply(lambda s: ", ".join(s)).to_dict()

def _parse_patient_hpo_terms(samplesheet, hpo_obo_path, hpo_dir, overwrite):
    """
    Read each sample's hpo_path from samplesheet, parse HPO IDs into terms.
    """
    save_file = hpo_dir / "Patients_HPO_Term.json"
    if save_file.exists() and not overwrite:
        with open(save_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("  Decoding HPO ontology ...")
    HPO_dict = pronto.Ontology(str(hpo_obo_path))

    patient_hpo_dict = defaultdict(list)
    for row in samplesheet.itertuples(index=True):
        sample_id = row.sampleID
        hpo_path = Path(row.hpo_path)

        if not hpo_path.exists():
            print(f"  HPO file not found for {sample_id}: {hpo_path} — using HP:0000001 default")
            patient_hpo_dict[sample_id] = [('HP:0000001', 'All', 0)]
            continue

        with hpo_path.open("r", encoding="utf-8") as f:
            hpo_ids = [line.strip() for line in f if line.strip()]

        if (len(hpo_ids) == 1 and hpo_ids[0] == 'HP:0000001') or len(hpo_ids) < 1:
            patient_hpo_dict[sample_id] = [('HP:0000001', 'All', 0)]
        else:
            for hpo_id in hpo_ids:
                if hpo_id in HPO_dict:
                    patient_hpo_dict[sample_id].append((hpo_id, HPO_dict[hpo_id].name, 1))

        if len(patient_hpo_dict[sample_id]) < 1:
            patient_hpo_dict[sample_id] = [('HP:0000001', 'All', 0)]
            print(f"  No valid HPO terms for {sample_id} — using HP:0000001 default")

    # Atomic write
    tmp = hpo_dir / f".{save_file.name}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(patient_hpo_dict), f)
    os.replace(tmp, save_file)

    return patient_hpo_dict

def Preprocessing(samplesheet, work_dir, references, overwrite=False):
    """
    Prepare phenotype reference data for downstream phenotype agents.
    
    Pipeline:
      1. Merge candidate gene tables across samples.
      2. Parse each patient's HPO terms via samplesheet hpo_path column.
      3. Build HPO gene-to-phenotype map.
      4. Build OMIM gene-to-clinical-features map.
    
    Outputs in work_dir/Agents/Phenotype/:
        PatientCandGene/MergedFile.feather
        HPO/Patient_HPO_Dict.json
        HPO/HPO_MAP.json
        HPO/OMIM_MAP.json
    
    Args:
        samplesheet: Must contain sampleID, candidates_path, hpo_path columns.
        work_dir: Pipeline work directory.
        references: AimReferences (uses .rarecollab.hpo_lib, .rarecollab.hpo_genes,
            .rarecollab.omim_disease).
        overwrite: Re-build outputs even if they already exist.
    """
    for col in ("candidates_path", "hpo_path"):
        if col not in samplesheet.columns:
            raise ValueError(
                f"samplesheet missing '{col}' column. "
                f"Run upstream pipeline steps first."
            )

    work_dir = Path(work_dir)
    root_path = work_dir / "Agents" / "Phenotype"
    hpo_dir = root_path / "HPO"
    cand_gene_dir = root_path / "PatientCandGene"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    cand_gene_dir.mkdir(parents=True, exist_ok=True)

    # ===== Step 1: Merge candidate gene tables across samples =====
    print("1. Merging patient candidate gene files ...")
    merged_path = cand_gene_dir / "MergedFile.feather"
    if merged_path.exists() and not overwrite:
        print("  Merged data detected, loading.")
    else:
        merged_data = []
        for row in tqdm(samplesheet.itertuples(index=True),
                        total=len(samplesheet), desc="Merging"):
            df = pd.read_feather(row.candidates_path, columns=['identifier', 'geneSymbol'])
            df = df.drop_duplicates().reset_index(drop=True)
            merged_data.append(df)
        merged_data = pd.concat(merged_data, ignore_index=True, copy=False)

        # Atomic write
        tmp = cand_gene_dir / f".{merged_path.name}.tmp"
        merged_data.to_feather(tmp)
        os.replace(tmp, merged_path)

    # ===== Step 2: Parse patient HPO terms =====
    print("2. Preparing patient HPO terms ...")
    patient_hpo_dict_path = hpo_dir / "Patient_HPO_Dict.json"
    if patient_hpo_dict_path.exists() and not overwrite:
        print("  Patient HPO dict detected, skipping.")
    else:
        patient_dict = _parse_patient_hpo_terms(
            samplesheet, references.rarecollab.hpo_lib, hpo_dir, overwrite,
        )
        # Build sample_id → "term1, term2, ..." string
        patient_hpo_term_dict = {
            sample_id: ", ".join(t[1] for t in tuples)
            for sample_id, tuples in patient_dict.items()
        }

        tmp = hpo_dir / f".{patient_hpo_dict_path.name}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(patient_hpo_term_dict, f)
        os.replace(tmp, patient_hpo_dict_path)

    # ===== Step 3: Build HPO gene-to-phenotype map =====
    print("3. Processing HPO gene mapping ...")
    hpo_map_path = hpo_dir / "HPO_MAP.json"
    if hpo_map_path.exists() and not overwrite:
        print("  HPO MAP detected, skipping.")
    else:
        df = pd.read_csv(references.rarecollab.hpo_genes, sep="\t", dtype=str)
        hpo_map = build_gene2hpo(df)

        tmp = hpo_dir / f".{hpo_map_path.name}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(hpo_map, f)
        os.replace(tmp, hpo_map_path)

    # ===== Step 4: Build OMIM gene-to-clinical-features map =====
    print("4. Processing OMIM documents ...")
    omim_map_path = hpo_dir / "OMIM_MAP.json"
    if omim_map_path.exists() and not overwrite:
        print("  OMIM MAP detected, skipping.")
    else:
        df = pd.read_table(references.rarecollab.omim_disease)
        df = df[['gene_symbol', 'Phenotypes', 'Clinical Features']].copy()
        df = df.dropna(subset=["Clinical Features"]).reset_index(drop=True)
        omim_gene_map = (
            df.assign(
                _cf=lambda d: d["Clinical Features"].str.replace("\n\n", "\n", regex=False),
                _txt=lambda d: (
                    d["gene_symbol"] + " causes " + d["Phenotypes"].astype(str)
                    + ".\nClinical Features: " + d["_cf"]
                ).str.strip(),
            )
            .groupby("gene_symbol")["_txt"]
            .apply(list)
            .to_dict()
        )

        tmp = hpo_dir / f".{omim_map_path.name}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(omim_gene_map, f)
        os.replace(tmp, omim_map_path)

    # ===== Update samplesheet =====
    samplesheet = samplesheet.copy()
    samplesheet["phenotype_agent_root"] = str(root_path)

    samplesheet_path = work_dir / "samplesheet_with_paths.csv"
    tmp = work_dir / ".samplesheet_with_paths.csv.tmp"
    samplesheet.to_csv(tmp, index=False)
    os.replace(tmp, samplesheet_path)
    print(f"Updated samplesheet: {samplesheet_path}")

    print("--Phenotype Preprocessing DONE--\n")
    return samplesheet
#↑-↑-Preprocessing-↑-↑
#↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑

#↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓
#↓-↓-↓-HPO Agent-↓-↓-↓
def build_prompt_HPO_agent(gene: str, patient_terms: str, hpo_gene_des: str) -> str:
    patient_block = patient_terms if patient_terms else "No patient phenotype terms provided."
    hpo_block = (hpo_gene_des if (hpo_gene_des and hpo_gene_des.strip()) else f"No documented phenotype from HPO for {gene}.")

    prompt = f"""You are a clinical genomics assistant. Your goal is to assess how well a candidate gene's known phenotypes match a patient's phenotype profile.

### Inputs
- Gene: {gene}

**Patient Phenotype (HPO terms):**
{patient_block}

**Gene Phenotype — HPO-derived (unique terms; Frequency Term when available):**
{hpo_block}

### Instructions
Definition of Vaild Frequency Term:
- Obligate: Always present, i.e. in 100% of the cases.
- Very frequent: Present in 80-99% of the cases.
- Frequent: Present in 30-79% of the cases.
- Occasional: Present in 5-29% of the cases.
- Very rare: Present in 1-4% of the cases.
- Excluded: Present in 0% of the cases.

#### A. General constraints
1) Use ONLY the information above. Do not invent additional phenotypes or disease mechanisms beyond what is provided.
2) Weigh both:
   - Positive matches: overlapping or very similar features, especially when they appear frequently in the gene phenotypes.
   - Unexplained key patient features: important patient findings that are not represented in the gene phenotypes.
   Treat unexplained features as “unexplained” rather than automatic contradictions, unless the gene phenotypes text explicitly indicate the opposite pattern.
3) Keep reasoning concise but specific: explicitly mention overlapping and unexplained features.

#### B. Think in phenotype axes / systems
4) Organize your reasoning by **major phenotype axes / systems**, rather than just counting raw terms. Typical axes include (but are not limited to):
   - Neuromuscular / motor axis (hypotonia, muscle weakness, myopathy, motor delay, fatigability, contractures, gait disturbance, etc.)
   - Central nervous system / brain structure axis (microcephaly, macrocephaly, brain malformations, seizures, ataxia, encephalopathy, etc.)
   - Cognitive / developmental / behavioral axis (global developmental delay, intellectual disability, autism, behavioral abnormalities, regression, etc.)
   - Craniofacial / dysmorphic axis (facial dysmorphism, micrognathia/prognathia, ear anomalies, palate anomalies, etc.)
   - Ocular axis (optic atrophy, retinal disease, cataracts, ophthalmoplegia, nystagmus, etc.)
   - Auditory axis (hearing loss or related ear findings).
   - Skeletal / limb / contracture axis (limb malformations, scoliosis/kyphosis, joint contractures, talipes, bone anomalies, fractures, etc.)
   - Growth axis (failure to thrive, short stature, overgrowth, abnormal weight gain/loss, slender/stocky build, etc.)
   - Cardiac axis (cardiomyopathy, arrhythmia, conduction defects, structural heart disease, etc.)
   - Respiratory axis (respiratory insufficiency, hypoventilation, recurrent infections, diaphragm weakness, etc.)
   - Gastrointestinal / feeding / nutrition axis (feeding difficulties, poor appetite, vomiting, malabsorption, tube feeding, etc.)
   - Endocrine / metabolic axis.
   - Genitourinary / genital axis.
   - Hematologic / immune / skin-hair axis.
   - Any other axis clearly suggested by the provided terms.

5) For each relevant axis:
   - Summarize key patient features on that axis.
   - Summarize overlapping gene features on that axis (mention frequency if present).
   - State whether this axis is **strongly explained**, **partially explained**, or **not explained** by the gene.

#### C. Label definitions
6) Use the following precise rules for the overall verdict:

- **Stand-Alone Strong Evidence**
  - Based on the provided phenotypes alone (HPO-derived terms), the gene explains the patient's phenotype in a **highly specific, convincing, diagnosis-like way**.
  - There is **strong, specific overlap across the main disease axes**, including multiple distinctive matching features (not just generic findings), and minimal important unexplained features.
  - No strong contradictions are evident based on the provided phenotypes.

- **Good Fit**
  - The gene phenotypes explain **most of the patient's key features across the main disease axes**.
  - There is strong, specific overlap on at least one major axis, and other major axes are either compatible or only missing a few features.
  - No strong contradictions are evident based on the provided phenotypes.

- **Partial Fit**
  - The gene phenotypes clearly and specifically explain **at least one major disease axis** of the patient, with **multiple specific overlapping features** on that axis (not just one or two very generic findings).
  - There may be important patient features or axes that remain unexplained by the gene phenotypes; these could be due to another gene, modifier, or unrelated causes.
  - Partial Fit should be used when there is a **coherent, disease-like pattern of overlap** on at least one axis, even if other axes are unexplained.

- **Not Fit**
  - There is **no major axis** with a clear, specific pattern of overlap between patient and gene phenotypes.
  - Any overlap is limited to **very nonspecific features** (for example, only “developmental delay”, “feeding difficulties”, or “hypotonia” without additional specific matching features), and no coherent disease pattern emerges.
  - Or the gene's typical phenotype pattern (based on the provided terms) is clearly incompatible with the patient's main problems, according to the information given.

7) When you are uncertain:
   - If the match is so specific that it feels like a convincing standalone diagnosis from phenotype alone, choose **Stand-Alone Strong Evidence**.
   - If there is a clear, specific pattern of overlap on at least one major axis with multiple matching features, but not fully convincing as a standalone diagnosis, lean toward **Good Fit** or **Partial Fit** depending on how many major axes are explained.
   - If overlap is only through 1-2 very generic findings without a coherent pattern, lean toward **Not Fit**.

#### D. Output formatting rules:
8) Output MUST be valid JSON. No markdown. No code fences. No extra keys. Use ASCII characters only. Use straight apostrophe ' and straight double quote " only.
   Return exactly one JSON object with keys "Reasoning" and "Conclusion".

   - "Reasoning" use bullet points grouped by major axis (e.g. "Neuromuscular axis:", "CNS/brain axis:", etc.), mapping key patient features to gene features, and stating for each axis whether it is strongly explained / partially explained / not explained.
   - "Conclusion" must be exactly one of: "Not Fit", "Partial Fit", "Good Fit", "Stand-Alone Strong Evidence".

   Output schema (must match exactly):{{"Reasoning":"<string>","Conclusion":"<Not Fit|Partial Fit|Good Fit|Stand-Alone Strong Evidence>"}}
   If you cannot comply, output exactly:{{"Reasoning":"comply_error","Conclusion":"Not Fit"}}
"""

    return prompt

def hpo_llm_one(model_name, ollama_url, temperature,
                llm_dir, sample_id, gene,
                patient_terms, hpo_gene_des,
                overwrite=False):
    """
    Call LLM for one (sample, gene) HPO evaluation. Atomic write of txt + json.
    Includes 3-retry JSON parsing because reasoning may be long/bullet-formatted.
    """
    output_dir = llm_dir / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_txt = output_dir / f"{gene}.txt"
    output_json = output_dir / f"{gene}.json"

    if output_json.exists() and not overwrite:
        return 1

    if patient_terms == "All":
        patient_terms = ""

    prompt_text = build_prompt_HPO_agent(
        gene=gene, patient_terms=patient_terms, hpo_gene_des=hpo_gene_des,
    )
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }

    # Long reasoning + bullets often breaks JSON; retry 3x
    obj = None
    last_output = None
    for attempt in range(3):
        r = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        last_output = data.get("response", "").strip()
        try:
            obj = json.loads(last_output)
            break
        except json.JSONDecodeError:
            continue

    if obj is None:
        raise RuntimeError(
            f"Failed to parse JSON after 3 tries for {sample_id}/{gene}. "
            f"Last output: {last_output[:200]}..."
        )

    conclusion = obj.get('Conclusion', 'Not Fit')
    reasoning_line = obj.get('Reasoning', 'comply_error')
    final_text = f"{reasoning_line}\nConclusion: {conclusion}\n"

    # Atomic write
    tmp_txt = output_txt.parent / f".{output_txt.name}.tmp"
    tmp_txt.write_text(final_text, encoding="utf-8")
    os.replace(tmp_txt, output_txt)

    tmp_json = output_json.parent / f".{output_json.name}.tmp"
    obj.setdefault('Reasoning', 'comply_error')
    obj.setdefault('Conclusion', 'Not Fit')
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp_json, output_json)

    return 1


def RunAgent_HPO(samplesheet, work_dir, llm_config, overwrite=False):
    """
    Run Phenotype HPO Agent on candidate genes for each sample.

    Evaluates how well each candidate gene's HPO-annotated phenotypes match
    the patient's HPO terms. Output:
        work_dir/Agents/Phenotype/AgentEvaluation/HPO_Agent/<sample>/<gene>.{txt,json}

    Requires PhenotypeAgent.Preprocessing to have been run first.

    Args:
        samplesheet: Pipeline samplesheet (used for phenotype_agent_root + sanity).
        work_dir: Pipeline work directory.
        llm_config: Dict with keys model_name, ollama_url, temperature, num_parallel.
        overwrite: Re-run even if outputs exist.
    """
    work_dir = Path(work_dir)
    root_path = work_dir / "Agents" / "Phenotype"

    # Load preprocessing outputs
    merged_path = root_path / "PatientCandGene" / "MergedFile.feather"
    patient_hpo_dict_path = root_path / "HPO" / "Patient_HPO_Dict.json"
    hpo_map_path = root_path / "HPO" / "HPO_MAP.json"

    for p, name in [
        (merged_path, "MergedFile.feather"),
        (patient_hpo_dict_path, "Patient_HPO_Dict.json"),
        (hpo_map_path, "HPO_MAP.json"),
    ]:
        if not p.exists():
            raise ValueError(
                f"{name} not found at {p}. "
                f"Run PhenotypeAgent.Preprocessing first."
            )

    print("Loading preprocessed phenotype data ...")
    merged_data = pd.read_feather(merged_path)
    with patient_hpo_dict_path.open("r", encoding="utf-8") as f:
        patient_hpo_term_dict = json.load(f)
    with hpo_map_path.open("r", encoding="utf-8") as f:
        hpo_map = json.load(f)

    LLM_WORKERS = llm_config.get("num_parallel", 1)
    llm_dir = root_path / "AgentEvaluation" / "HPO_Agent"
    llm_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running HPO Agent (model={llm_config['model_name']}, workers={LLM_WORKERS}) ...")

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        futures = [
            ex.submit(
                hpo_llm_one,
                llm_config["model_name"],
                llm_config["ollama_url"],
                llm_config["temperature"],
                llm_dir,
                row.identifier, row.geneSymbol,
                patient_hpo_term_dict.get(row.identifier, ""),
                hpo_map.get(row.geneSymbol, ""),
                overwrite,
            )
            for row in merged_data.itertuples(index=False)
        ]

        ok = fail = 0
        with tqdm(total=len(futures), desc="HPO Agent") as pbar:
            for fut in as_completed(futures):
                try:
                    fut.result()
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] HPO Agent: {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(Evaluated=ok, Failed=fail)

    print("--Phenotype HPO Agent DONE--\n")
    return samplesheet

#↑-↑-↑-HPO Agent-↑-↑-↑
#↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑

#↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-
#↓-↓-↓-OMIM Agent-↓-↓-↓
def build_prompt_OMIM_agent(gene: str, patient_hpo: str, omim_entries: str) -> str:
    patient_block = patient_hpo if (patient_hpo and patient_hpo.strip()) else "No patient phenotype terms provided."
    # Note: by construction, omim_entries should be non-empty here.
    omim_block = "\n\n".join([f"{i+1}.\n{s}" for i, s in enumerate(omim_entries)])
    prompt = f"""You are a clinical genomics assistant. Your task is to assess whether this gene's OMIM Clinical Features match the patient's phenotype.

Inputs
- Gene: {gene}

Patient phenotype (HPO terms):
{patient_block}

OMIM evidence (gene-linked; ONLY Clinical Features):
- Each numbered item below describes a disease/phenotype that is associated with this gene.
- The sentence "{gene} causes ..." explicitly indicates the gene-disease association for that item.

{omim_block}

Rules (must follow)
1) Use ONLY the text above. Do not add external gene knowledge.
2) Treat the patient phenotype list as the ONLY patient evidence available.
3) Do not require axis-based organization. Keep reasoning short and concrete.
4) Consider both:
   - Matches: specific overlaps between patient terms and OMIM clinical features.
   - Missing: major OMIM features not present in the patient terms.

Conclusion labels (choose exactly ONE)
- Stand-Alone Strong Evidence:
  OMIM clinical features provide a highly specific, diagnosis-like match to the patient; multiple distinctive overlaps; minimal major missing features.
- Good Fit:
  Patient matches most major OMIM clinical features; strong overlap overall; missing features are few or not central.
- Partial Fit:
  Clear, specific overlap exists for an important subset of OMIM features, but substantial major OMIM features are missing from the patient terms.
- Not Fit:
  No coherent specific overlap pattern; overlaps (if any) are only very generic and not compelling.
- Impossible:
  Use ONLY when OMIM clinical features explicitly state a phenotype is REQUIRED/INVARIABLE/ALWAYS PRESENT (i.e., a must-have core feature),
  and that must-have feature is NOT present (or clearly not represented) in the patient phenotype terms.
  Do NOT use Impossible just because something is “common” or “typical”; it must be stated as required/always present in the provided OMIM text.

Output formatting rules (STRICT):
Output MUST be valid JSON. No markdown. No code fences. No extra keys.
Use ASCII characters only. Use straight apostrophe ' and straight double quote " only.

Return exactly one JSON object with keys "Reasoning" and "Conclusion".
'Reasoning' should contain 2-6 bullet points (one line for each bullet) starting with '-', then in the middle explaining matches, missing major features, and any explicit must-have mismatch if present.
'Conclusion' must be exactly one of 'Impossible', 'Not Fit', 'Partial Fit', 'Good Fit', 'Stand-Alone Strong Evidence'

Output schema (must match exactly):{{"Reasoning":"<string>","Conclusion":"<Impossible|Not Fit|Partial Fit|Good Fit|Stand-Alone Strong Evidence>"}}
If you cannot comply, output exactly:{{"Reasoning":"comply_error","Conclusion":"Impossible"}}
"""
    return prompt

def omim_llm_one(model_name, ollama_url, temperature,
                 llm_dir, sample_id, gene,
                 patient_terms, omim_entries,
                 overwrite=False):
    """
    Call LLM for one (sample, gene) OMIM evaluation. Atomic write of txt + json.
    Returns:
        1 on success
        2 on skip (no OMIM entries for this gene)
    """
    output_dir = llm_dir / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_txt = output_dir / f"{gene}.txt"
    output_json = output_dir / f"{gene}.json"

    if output_json.exists() and not overwrite:
        return 1

    if not omim_entries:
        return 2   # signal "skipped, no OMIM data for gene"

    if patient_terms == "All":
        patient_terms = ""

    prompt_text = build_prompt_OMIM_agent(gene, patient_terms, omim_entries)
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }

    # Long reasoning + bullets often breaks JSON; retry 3x
    obj = None
    last_output = None
    for attempt in range(3):
        r = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        last_output = data.get("response", "").strip()
        try:
            obj = json.loads(last_output)
            break
        except json.JSONDecodeError:
            continue

    if obj is None:
        raise RuntimeError(
            f"Failed to parse JSON after 3 tries for {sample_id}/{gene}. "
            f"Last output: {last_output[:200]}..."
        )

    conclusion = obj.get('Conclusion', 'Not Fit')
    reasoning_line = obj.get('Reasoning', 'comply_error')
    final_text = f"{reasoning_line}\nConclusion: {conclusion}\n"

    # Atomic write
    tmp_txt = output_txt.parent / f".{output_txt.name}.tmp"
    tmp_txt.write_text(final_text, encoding="utf-8")
    os.replace(tmp_txt, output_txt)

    tmp_json = output_json.parent / f".{output_json.name}.tmp"
    obj.setdefault('Reasoning', 'comply_error')
    obj.setdefault('Conclusion', 'Not Fit')
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp_json, output_json)

    return 1


def RunAgent_OMIM(samplesheet, work_dir, llm_config, overwrite=False):
    """
    Run Phenotype OMIM Agent on candidate genes for each sample.

    Evaluates how well each candidate gene's OMIM-documented clinical features
    match the patient's HPO terms. Output:
        work_dir/Agents/Phenotype/AgentEvaluation/OMIM_Agent/<sample>/<gene>.{txt,json}

    Genes with no OMIM entries are skipped (counted at the end).

    Requires PhenotypeAgent.Preprocessing to have been run first.

    Args:
        samplesheet: Pipeline samplesheet.
        work_dir: Pipeline work directory.
        llm_config: Dict with keys model_name, ollama_url, temperature, num_parallel.
        overwrite: Re-run even if outputs exist.
    """
    work_dir = Path(work_dir)
    root_path = work_dir / "Agents" / "Phenotype"

    # Load preprocessing outputs
    merged_path = root_path / "PatientCandGene" / "MergedFile.feather"
    patient_hpo_dict_path = root_path / "HPO" / "Patient_HPO_Dict.json"
    omim_map_path = root_path / "HPO" / "OMIM_MAP.json"

    for p, name in [
        (merged_path, "MergedFile.feather"),
        (patient_hpo_dict_path, "Patient_HPO_Dict.json"),
        (omim_map_path, "OMIM_MAP.json"),
    ]:
        if not p.exists():
            raise ValueError(
                f"{name} not found at {p}. "
                f"Run PhenotypeAgent.Preprocessing first."
            )

    print("Loading preprocessed phenotype data ...")
    merged_data = pd.read_feather(merged_path)
    with patient_hpo_dict_path.open("r", encoding="utf-8") as f:
        patient_hpo_term_dict = json.load(f)
    with omim_map_path.open("r", encoding="utf-8") as f:
        OMIM_Gene_MAP = json.load(f)

    LLM_WORKERS = llm_config.get("num_parallel", 1)
    llm_dir = root_path / "AgentEvaluation" / "OMIM_Agent"
    llm_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running OMIM Agent (model={llm_config['model_name']}, workers={LLM_WORKERS}) ...")

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        futures = [
            ex.submit(
                omim_llm_one,
                llm_config["model_name"],
                llm_config["ollama_url"],
                llm_config["temperature"],
                llm_dir,
                row.identifier, row.geneSymbol,
                patient_hpo_term_dict.get(row.identifier, ""),
                OMIM_Gene_MAP.get(row.geneSymbol, []),
                overwrite,
            )
            for row in merged_data.itertuples(index=False)
        ]

        ok = skipped = fail = 0
        with tqdm(total=len(futures), desc="OMIM Agent") as pbar:
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    if ret == 1:
                        ok += 1
                    elif ret == 2:
                        skipped += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] OMIM Agent: {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(Evaluated=ok, Skipped=skipped, Failed=fail)

    if skipped > 0:
        print(f"{skipped} (sample, gene) pairs skipped — no OMIM clinical features.")

    print("--Phenotype OMIM Agent DONE--\n")
    return samplesheet

#↑-↑-↑-OMIM Agent-↑-↑-↑
#↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-


#↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-
#↓-↓-↓-Literature Agent-↓-↓-↓
def build_pubmed_term(gene: str) -> str:
    gene_q = f"\"{gene}\"[Title/Abstract]"
    kw_q = "(" + " OR ".join(
        [f"{kw}[Title/Abstract]" for kw in ["variant", "variants", "mutation",
                                            "mutations", "diagnosis", "mendelian",
                                            "cause", "causes"]]
    ) + ")"
    return f"({gene_q}) AND {kw_q}"

def esearch_pmids(term: str, retmax: int = 400, max_retries: int = 5):
    last_e = None
    for attempt in range(max_retries):
        try:
            with Entrez.esearch(db="pubmed", term=term, retmax=retmax, sort="relevance") as handle:
                res = Entrez.read(handle)
            return res.get("IdList", [])
        except Exception as e:
            last_e = e
            sleep_s = min(60, 2 ** attempt)
            time.sleep(sleep_s)
    print(f"[esearch] ERROR after max retries: {last_e} term={term}")
    return []

def _make_gene_boundary_regex(gene: str) -> re.Pattern:
    """
    Strictly match the gene name: letters/numbers/underlines etc are not allowed
    eg: 'SNUPN' match exactly '... SNUPN ...'; NOT match 'SNUPN1' or 'pre-SNUPN'.
    """
    g = re.escape(gene)
    return re.compile(rf"(?<![A-Za-z0-9_\-]){g}(?![A-Za-z0-9_\-])")

def local_strict_filter(records: list[dict[str, str]], gene: str, max_keep: int = 50):
    """
    Filter unrelated gene names in Title/Abstract (ignore case)
    return max_keep items
    """
    gpat = _make_gene_boundary_regex(gene)
    KW_REGEX = re.compile(r"\b(?:variant|variants|mutation|mutations|diagnosis|mendelian|cause|causes)\b", re.I)
    out = []
    for r in records:
        pmid = str(r.get("PMID", "")).strip()
        title = (r.get("TI") or "").strip()
        abstr = (r.get("AB") or "").strip()
        text = f"{title} {abstr}"
        if not pmid or not title:
            continue
        if not gpat.search(text):
            continue
        if not KW_REGEX.search(text):
            continue
        out.append({"PMID": pmid, "Title": title})
        if len(out) >= max_keep:
            break
    return out

def efetch_medline(pmids: list[str], chunk_size: int = 200, max_retries: int = 4):
    records = []

    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i : i + chunk_size]
        last_e = None
        for attempt in range(max_retries):
            try:
                with Entrez.efetch(
                    db="pubmed",
                    id=",".join(chunk),
                    rettype="medline",
                    retmode="text",
                ) as handle:
                    records.extend(Medline.parse(handle))
                break 
            except Exception as e:
                last_e = e
                sleep_s = min(60, 2 ** attempt)
                time.sleep(sleep_s)
        else:
            print(
                f"[efetch] ERROR after retries: {last_e} "
                f"pmids[{i}:{min(i + chunk_size, len(pmids))}] n={len(chunk)}"
                )

    return records

def prompt_for_classification(gene: str, items: list[dict[str, str]]) -> str:
    """
    Return LLM prompy with "- PMID :: Title".
    """
    header = (
        "You are a precise literature triager for rare disease.\n"
        f"Gene: {gene}\n"
        "Given a list of items (PMID, Title), OUTPUT ONLY the items that are explicitly about rare disease contexts and classify EACH into exactly one of:\n"
        "- phenotype association (links gene to disease/phenotype without mechanistic experiments),\n"
        "- functional study (mechanistic or model experiments about variant impact or gene function),\n"
        "- patient cohort (multi-patient genetic study),\n"
        "- case report (single patient or small family report),\n"
        "- review (review article).\n"
        "If uncertain, choose the most plausible single category. Do not include anything unrelated to rare disease.\n"
        "Return as TSV with exactly 3 columns and no extra commentary:\n"
        "PMID\\tTitle\\tCategory\n"
        "\nItems:\n"
    )

    lines = []
    for it in items:
        lines.append(f"- {it['PMID']}: {it['Title']}")
    lines.append("\nNow produce ONLY the TSV with header line.")
    return header + "\n".join(lines)

def fetch_pubmed_records(pmids: list[str],EUTILS_EFETCH, NCBI_EMAIL, NCBI_KEY) -> dict[str, dict[str, str]]:
    NCBI_SLEEP_SEC = 0.34
    pmids = [p for p in pmids if p and p.isdigit()]
    if not pmids:
        return {}
    out = {}
    PMID_BATCH_SIZE = 50
    for i in range(0, len(pmids), PMID_BATCH_SIZE):
        batch = pmids[i:i+PMID_BATCH_SIZE]
        recs = fetch_pubmed_batch(batch,EUTILS_EFETCH, NCBI_EMAIL, NCBI_KEY)
        out.update(recs)
        if NCBI_SLEEP_SEC:
            time.sleep(NCBI_SLEEP_SEC)
    return out

def fetch_pubmed_batch(pmids: list[str], EUTILS_EFETCH, NCBI_EMAIL, NCBI_KEY) -> dict[str, dict[str, str]]:
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
        "tool": "literature_agent",
    }
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_KEY:
        params["api_key"] = NCBI_KEY

    r = requests.get(EUTILS_EFETCH, params=params, timeout=60)
    r.raise_for_status()
    out = {}
    root = ET.fromstring(r.text)
    # XML: PubmedArticleSet/PubmedArticle/MedlineCitation/PMID; Article/ArticleTitle; Article/Abstract/AbstractText
    for art in root.findall(".//PubmedArticle"):
        pmid_node = art.find("./MedlineCitation/PMID")
        pmid = pmid_node.text.strip() if (pmid_node is not None and pmid_node.text) else None
        # Abstract can have multiple AbstractText nodes; concatenate
        abs_parts = []
        for at in art.findall("./MedlineCitation/Article/Abstract/AbstractText"):
            t = "".join(at.itertext()).strip() if at is not None else ""
            if t:
                abs_parts.append(t)
        abstract = " ".join(abs_parts).strip()

        if pmid:
            out[pmid] = {"title": "", "abstract": abstract}
    return out

def build_prompt_Literature_agent(gene: str, phenotype_text: str, lit_items: list[dict[str, str]]) -> str:
    """
    Build prompt:
    - Patient phenotype (HPO labels 串)
    - Gene symbol
    - Literature (PMID + Title + Abstract)
    """
    phenostr = phenotype_text if phenotype_text else "(not provided)"
    gene_line = f"Gene: {gene}"

    #literatures:
    if lit_items:
        lit_lines = []
        for it in lit_items:
            lit_lines.append( f"""- PMID: {it['pmid']} Title: {it.get('title','')} Abstract: {it.get('abstract','')}""")
        lit_block = "\n".join(lit_lines)
    else:
        lit_block = "(no qualifying papers found by category filter)"

    prompt = f"""You are a genetics literature reviewer assisting with phenotype–gene matching.

Patient phenotype (HPO labels):
{phenostr}

{gene_line}

Relevant literature (PubMed abstracts with PMIDs):
{lit_block}

Task:
Based only on the patient phenotype and the gene-associated information described in the abstracts above, assess how well the gene's **reported** phenotype matches the patient's phenotype profile.

Important constraints:
- Use ONLY the information in the text above. Do not add phenotypes or mechanisms from outside knowledge.
- Abstracts often describe only the most typical or partial phenotype for a gene. 
  → Absence of a feature in the abstracts should usually be treated as “unknown”, not as evidence against the gene.
- Give more weight to:
  * Clear overlaps in organ system / disease axis (e.g. neuromuscular disease, epileptic encephalopathy, cardiomyopathy, renal disease).
  * Specific shared findings (e.g. “congenital hypotonia”, “optic atrophy”, “hypertrophic cardiomyopathy”, “early-onset ataxia”).
- Use mismatch only when the abstracts explicitly describe a phenotype pattern that is clearly incompatible with the patient's main problems
  (for example: purely adult-onset isolated cardiomyopathy in the literature vs. isolated congenital brain malformation in the patient).

Label definitions:
- Good Fit:
  - The abstracts describe a phenotype pattern that strongly overlaps the patient's main problems
    (same key organ systems and age of onset, and several specific shared features),
  - and there are no clear conflicts based on what is written.
- Partial Fit:
  - The abstracts support a plausible overlap with the patient:
    * same main organ system or syndrome family, OR
    * several shared but not exhaustive features,
  - but the description is incomplete, more general, or only partially matches the patient.
  - This is the appropriate label when the literature suggests the gene could reasonably explain at least part of the patient's phenotype, but important details are missing.
- Not Fit:
  - The abstracts either:
    (a) describe a phenotype pattern that is clearly different from the patient's main problems (based on the text), OR
    (b) contain no meaningful phenotype information relevant to the patient (no overlap beyond extremely generic words like “patient”, “disease”, “mutation” without phenotypic content).

When uncertain between Partial Fit and Not Fit:
- If there is at least one plausible overlap at the level of organ system or phenotype cluster, lean toward **Partial Fit**.
- If there is no meaningful overlap in phenotype or organ system, lean toward **Not Fit**.

Return exactly one line JSON object with keys "Reasoning" and "Conclusion":
'Reasoning' should be a brief and concrete rationale grounded in phenotype terms and the abstracts; explicitly mention overlaps / or clear differences, and cite PMIDs in parentheses where relevant, e.g. (PMID:12345678).
'Conclusion' must be exactly one of 'Not Fit', 'Partial Fit', 'Good Fit'.

Output schema (must match exactly):{{"Reasoning":"<string>","Conclusion":"<Not Fit|Partial Fit|Good Fit>"}}
If you cannot comply, output exactly:{{"Reasoning":"comply_error","Conclusion":"Not Fit"}}
"""

    return prompt


def literature_fetch_one(gene, gene_pubmed_dir, collected_lit_dir, related_lit_dir,
                        model_name, ollama_url, temperature,
                        literature_categories, eutils_efetch_url,
                        ncbi_email, ncbi_key,
                        overwrite=False):
    """
    Phase 1 worker: for one gene, search PubMed, filter, LLM-classify literature,
    fetch abstracts of relevant papers.
    
    Outputs:
        gene_pubmed_dir/<gene>.json     PMID list from esearch
        collected_lit_dir/<gene>.tsv     LLM-classified literature
        related_lit_dir/<gene>.json      Final abstracts (only if any relevant)
    
    Returns:
        1 on success
        2 on no-pubmed-results (skipped)
        3 on no-rare-disease-literature (skipped)
    """
    search_res_path = gene_pubmed_dir / f"{gene}.json"
    classified_path = collected_lit_dir / f"{gene}.tsv"
    abstract_path = related_lit_dir / f"{gene}.json"
    
    # Cache hit: if final output exists, skip
    if abstract_path.exists() and not overwrite:
        return 1
    
    # Step 1.1: PubMed esearch (cached per gene)
    if search_res_path.exists() and not overwrite:
        with search_res_path.open("r", encoding="utf-8") as f:
            pmids = json.load(f)
    else:
        search_term = build_pubmed_term(gene)
        pmids = esearch_pmids(search_term, retmax=400, max_retries=5)
        tmp = gene_pubmed_dir / f".{search_res_path.name}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(pmids, f)
        os.replace(tmp, search_res_path)
    
    if not pmids:
        return 2  # No PubMed results
    
    # Step 1.2: LLM-classify literature (cached per gene)
    if classified_path.exists() and not overwrite:
        obj = pd.read_csv(classified_path, dtype='str')
    else:
        recs = efetch_medline(pmids=pmids)
        items = local_strict_filter(recs, gene, max_keep=50)
        prompt_text = prompt_for_classification(gene, items)
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        
        obj = None
        last_output = None
        for attempt in range(3):
            r = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            last_output = data.get("response", "").strip()
            llm_output_fixed = last_output.replace("\\t", "\t")
            try:
                obj = pd.read_csv(StringIO(llm_output_fixed), sep="\t", dtype=str)
                break
            except Exception:
                continue
        
        if obj is None:
            raise RuntimeError(
                f"Failed to parse TSV after 3 tries for gene {gene}. "
                f"Last output: {last_output[:200]}..."
            )
        
        tmp = collected_lit_dir / f".{classified_path.name}.tmp"
        obj.to_csv(tmp, index=False)
        os.replace(tmp, classified_path)
    
    # Step 1.3: Filter to rare-disease-relevant categories
    if len(obj) == 0:
        return 3  # No relevant literature
    
    obj["Category"] = obj["Category"].str.lower().str.strip()
    obj = obj[obj["Category"].isin(literature_categories)]
    filtered_pmids = list(set(obj['PMID']))
    if len(filtered_pmids) == 0:
        return 3
    
    # Step 1.4: Fetch abstracts
    try:
        pmid2rec = fetch_pubmed_records(
            filtered_pmids,
            EUTILS_EFETCH=eutils_efetch_url,
            NCBI_EMAIL=ncbi_email,
            NCBI_KEY=ncbi_key,
        )
    except Exception as e:
        print(f"\n[WARN] PubMed fetch failed for {gene}: {e}")
        pmid2rec = {}
    
    for pmid, item in pmid2rec.items():
        matching_titles = obj.loc[obj['PMID'] == pmid, 'Title']
        if len(matching_titles) > 0:
            pmid2rec[pmid]['title'] = matching_titles.iloc[0]
    
    tmp = related_lit_dir / f".{abstract_path.name}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(pmid2rec, f)
    os.replace(tmp, abstract_path)
    
    return 1


def literature_llm_one(model_name, ollama_url, temperature,
                       llm_dir, related_lit_dir,
                       sample_id, gene, patient_terms,
                       overwrite=False):
    """
    Phase 2 worker: for one (sample, gene), evaluate gene's literature
    against patient HPO terms via LLM.
    
    Returns:
        1 on success
        2 on skip (no abstract for this gene)
    """
    abstract_path = related_lit_dir / f"{gene}.json"
    if not abstract_path.exists():
        return 2
    
    output_dir = llm_dir / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_txt = output_dir / f"{gene}.txt"
    output_json = output_dir / f"{gene}.json"
    
    if output_json.exists() and not overwrite:
        return 1
    
    with abstract_path.open("r", encoding="utf-8") as f:
        gene_abstract = json.load(f)
    
    lit_items = [
        {"pmid": pmid, "title": content.get("title", ""), "abstract": content.get("abstract", "")}
        for pmid, content in gene_abstract.items()
    ]
    
    prompt_text = build_prompt_Literature_agent(
        gene=gene, phenotype_text=patient_terms, lit_items=lit_items,
    )
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    
    obj = None
    last_output = None
    for attempt in range(3):
        r = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        last_output = data.get("response", "").strip()
        try:
            obj = json.loads(last_output)
            break
        except json.JSONDecodeError:
            continue
    
    if obj is None:
        raise RuntimeError(
            f"Failed to parse JSON after 3 tries for {sample_id}/{gene}. "
            f"Last output: {last_output[:200]}..."
        )
    
    conclusion = obj.get('Conclusion', 'Not Fit')
    reasoning_line = obj.get('Reasoning', 'comply_error')
    final_text = f"{reasoning_line}\nConclusion: {conclusion}\n"
    
    # Atomic write
    tmp_txt = output_txt.parent / f".{output_txt.name}.tmp"
    tmp_txt.write_text(final_text, encoding="utf-8")
    os.replace(tmp_txt, output_txt)
    
    tmp_json = output_json.parent / f".{output_json.name}.tmp"
    obj.setdefault('Reasoning', 'comply_error')
    obj.setdefault('Conclusion', 'Not Fit')
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp_json, output_json)
    
    return 1


def RunAgent_Literature(samplesheet, work_dir, llm_config,
                        ncbi_email=None, ncbi_api_key=None,
                        overwrite=False):
    """
    Run Phenotype Literature Agent.
    
    Two-phase pipeline:
      Phase 1 (per-gene, gene-level):
        For each candidate gene with no HPO_MAP entry:
          - Search PubMed for rare-disease literature
          - LLM-classify the results
          - Fetch abstracts for relevant categories
      Phase 2 (per-(sample, gene)):
        For each (sample, gene) with abstracts:
          - LLM evaluates fit between patient HPO terms and gene's literature
    
    Outputs in work_dir/Agents/Phenotype/:
        GenePubmed/<gene>.json         PubMed PMIDs per gene
        GeneLiterature/<gene>.tsv      LLM-classified literature per gene
        GeneAbstract/<gene>.json       Filtered relevant abstracts per gene
        AgentEvaluation/Literature_Agent/<sample>/<gene>.{txt,json}
    
    Requires PhenotypeAgent.Preprocessing to have been run first.
    
    Args:
        samplesheet: Pipeline samplesheet.
        work_dir: Pipeline work directory.
        llm_config: Dict with keys model_name, ollama_url, temperature, num_parallel.
        ncbi_email: NCBI account email (recommended, raises rate limit).
        ncbi_api_key: NCBI API key (recommended, raises rate limit).
        overwrite: Re-run even if outputs exist.
    """
    work_dir = Path(work_dir)
    root_path = work_dir / "Agents" / "Phenotype"
    
    # Load preprocessing outputs
    merged_path = root_path / "PatientCandGene" / "MergedFile.feather"
    gene_map_path = root_path / "HPO" / "HPO_MAP.json"
    patient_hpo_dict_path = root_path / "HPO" / "Patient_HPO_Dict.json"
    
    for p, name in [
        (merged_path, "MergedFile.feather"),
        (gene_map_path, "HPO_MAP.json"),
        (patient_hpo_dict_path, "Patient_HPO_Dict.json"),
    ]:
        if not p.exists():
            raise ValueError(
                f"{name} not found at {p}. "
                f"Run PhenotypeAgent.Preprocessing first."
            )
    
    print("Loading preprocessed phenotype data ...")
    merged_data = pd.read_feather(merged_path)
    with gene_map_path.open("r", encoding="utf-8") as f:
        gene_map = json.load(f)
    with patient_hpo_dict_path.open("r", encoding="utf-8") as f:
        patient_hpo = json.load(f)
    
    # Genes without HPO_MAP entries — these need literature search
    all_genes = set(merged_data['geneSymbol'])
    genelist = sorted(all_genes - set(gene_map.keys()))
    print(f"{len(genelist)} candidate genes need literature search.")
    
    # Setup Entrez
    if ncbi_email:
        Entrez.email = ncbi_email
    if ncbi_api_key:
        Entrez.api_key = ncbi_api_key
    
    EUTILS_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    LITERATURE_CATEGORIES = {
        "functional study", "review", "case report",
        "patient cohort", "phenotype association",
    }
    
    # Output dirs
    gene_pubmed_dir = root_path / "GenePubmed"
    collected_lit_dir = root_path / "GeneLiterature"
    related_lit_dir = root_path / "GeneAbstract"
    llm_dir = root_path / "AgentEvaluation" / "Literature_Agent"
    for d in [gene_pubmed_dir, collected_lit_dir, related_lit_dir, llm_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    PHASE1_WORKERS = min(llm_config.get("num_parallel", 1), 2)
    PHASE2_WORKERS = llm_config.get("num_parallel", 1)
    
    # ===== Phase 1: Collect literature per gene =====
    print(f"Phase 1: Collecting literature ({PHASE1_WORKERS} workers) ...")
    with ThreadPoolExecutor(max_workers=PHASE1_WORKERS) as ex:
        futures_map = {
            ex.submit(
                literature_fetch_one,
                gene, gene_pubmed_dir, collected_lit_dir, related_lit_dir,
                llm_config["model_name"], llm_config["ollama_url"], llm_config["temperature"],
                LITERATURE_CATEGORIES, EUTILS_EFETCH,
                ncbi_email, ncbi_api_key,
                overwrite,
            ): gene
            for gene in genelist
        }
        
        ok = no_pubmed = no_relevant = fail = 0
        with tqdm(total=len(futures_map), desc="Collecting Literature") as pbar:
            for fut in as_completed(futures_map):
                gene = futures_map[fut]
                try:
                    ret = fut.result()
                    if ret == 1:
                        ok += 1
                    elif ret == 2:
                        no_pubmed += 1
                    elif ret == 3:
                        no_relevant += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] Literature fetch for {gene}: {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(ok=ok, no_pubmed=no_pubmed, no_rel=no_relevant, fail=fail)
    
    print(f"  Phase 1 done: {ok} genes with abstracts, "
          f"{no_pubmed} no PubMed hits, {no_relevant} no relevant categories, {fail} failed.")
    
    # ===== Phase 2: Per-(sample, gene) LLM evaluation =====
    eval_data = merged_data[merged_data['geneSymbol'].isin(set(genelist))].reset_index(drop=True)
    print(f"Phase 2: Evaluating {len(eval_data)} (sample, gene) pairs ({PHASE2_WORKERS} workers) ...")
    
    with ThreadPoolExecutor(max_workers=PHASE2_WORKERS) as ex:
        futures = [
            ex.submit(
                literature_llm_one,
                llm_config["model_name"], llm_config["ollama_url"], llm_config["temperature"],
                llm_dir, related_lit_dir,
                row.identifier, row.geneSymbol,
                patient_hpo.get(row.identifier, ""),
                overwrite,
            )
            for row in eval_data.itertuples(index=False)
        ]
        
        ok = skipped = fail = 0
        with tqdm(total=len(futures), desc="Literature Agent") as pbar:
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    if ret == 1:
                        ok += 1
                    elif ret == 2:
                        skipped += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] Literature LLM: {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(Evaluated=ok, Skipped=skipped, Failed=fail)
    
    if skipped > 0:
        print(f"{skipped} (sample, gene) pairs skipped — no relevant literature.")
    
    print("--Phenotype Literature Agent DONE--\n")
    return samplesheet

#↑-↑-↑-Literature Agent-↑-↑-↑
#↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-

