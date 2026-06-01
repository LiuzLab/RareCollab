#!/usr/bin/env python
# coding: utf-8

import time
import requests
import os
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import json

def Call_NCBI_API(url, params, max_try = 3, timeout = 60):
    for attempt in range(max_try + 1):
        r = requests.get(url, params = params, timeout = timeout)
        #print(r.status_code)
        if r.status_code < 400: #Success
            return ET.fromstring(r.text).find(".//ClinicalAssertionList")
        if r.status_code == 429 or 500 <= r.status_code < 600: #Too Many Requests Or some other issues;
            sleep_s = min(60, (2 ** attempt))
            time.sleep(sleep_s)
        else:
            r.raise_for_status()
            return None
        
def _element_to_lines(el: ET.Element, depth: int = 0):
    lines = []
    name = el.tag
    attrs = {k: str(v) for k, v in el.attrib.items()}
    attr_str = (" [" + ", ".join(f'{k}="{v}"' for k, v in attrs.items()) + "]") if attrs else ""
    txt = el.text.strip() if el.text is not None else ""
    indent = "  " * depth
    #print(txt == "")
    if txt != "":
        lines.append(f"{indent}{name}{attr_str}: {txt}")
    elif len(list(el)) < 1:
        lines.append(f"{indent}{name}{attr_str}")
    else:
        lines.append(f"{indent}{name}{attr_str}:")
    for ch in list(el):
        ch_txt = ch.text.strip() if ch.text is not None else ""
        has_attr = any(v is not None for v in ch.attrib.values())
        has_grand = len(list(ch)) > 0
        if ch_txt is None and not has_attr and not has_grand:
            continue
        lines.extend(_element_to_lines(ch, depth + 1))
    return lines

def build_prompt_en(var_key: str, HGVSc_core:str, preview_blocks: list[str]) -> str:
  submissions_str = "\n---\n".join(preview_blocks).strip()
  prompt = f"""You are a clinical genetics assistant.

Task:
Given ClinVar submissions for a variant, do TWO things:
1) Judge whether the submissions support the variant's pathogenicity (Pathogenic/Likely pathogenic) vs not (Benign/Likely benign).
2) Infer the REQUIRED / IMPLIED zygosity for disease causality AS DESCRIBED in the submissions (do NOT guess beyond text).

Candidate variant identifiers:
- Variant key (chr_pos_ref_alt): {var_key}
- HGVSc_core: {HGVSc_core}

ClinVar submissions (verbatim, human-readable XML-to-text rendering of each submitter's record):
===
{submissions_str}
===

How to evaluate pathogenicity (be strict and concrete):
- Consider: submitter classifications (P/LP/VUS/B/LB), conflicts, review status / assertion criteria, evidence statements (case/segregation/functional), and recency.

Conclusion label definitions (choose ONE):
- Against:
  Overall evidence in submissions leans Benign/Likely benign, or does not support pathogenicity (e.g., mostly B/LB, or strong B/LB with no credible counter-evidence).
- Neutral:
  Not enough information to judge pathogenicity (e.g., sparse submissions, unclear evidence, only generic statements, or VUS with no meaningful supporting detail).
- Supporting:
  Evidence leans Pathogenic/Likely pathogenic, but not overwhelming.
  Examples:
  * Conflicting classifications but the balance of credible evidence leans P/LP.
  * Classified as VUS but includes concrete details that plausibly support P/LP (case/segregation/functional hints).
- Convincing:
  Strong evidence for Pathogenic/Likely pathogenic.
  Examples:
  * Multiple credible submissions consistently P/LP with assertion criteria and/or strong detailed evidence.
  * Clear, trustworthy detailed evidence strongly indicates pathogenicity even if not every submitter is perfectly aligned.

How to infer zygosity (do NOT guess beyond text):
- Choose from: homozygous | compound heterozygous | heterozygous | no information
- Use:
  * "homozygous" if submissions explicitly mention homozygous patients/requirement or clearly AR with homozygous cases.
  * "compound heterozygous" if submissions explicitly mention compound het / biallelic with different alleles / in trans.
  * "heterozygous" if submissions explicitly mention heterozygous affected individuals or AD mechanism tied to het cases.
  * "no information" if submissions do not clearly imply any of the above.

Output requirements:
- Must include an explicit Conclusion line with exactly one of: Against | Neutral | Supporting | Convincing
- Must include an explicit Zygosity line with exactly one of: homozygous | compound heterozygous | heterozygous | no information
- Reasoning should be brief (1-4 sentences).

Output MUST be valid JSON on a single line. No markdown. No code fences. No extra keys.
 Schema (keys must match exactly):
 "reasoning":"<string>","conclusion":"<Against|Neutral|Supporting|Convincing>","zygosity":"<homozygous|compound_heterozygous|heterozygous|no_information>"

 Rules:
 - Use exactly one of the allowed enum values.
 - If zygosity is not explicitly stated or cannot be inferred, use "no_information".
 - If you are unsure about conclusion, use "Neutral".
 - Do not include newlines in JSON string values.
 - If you cannot comply, output exactly: "reasoning":"parse_error","conclusion":"Neutral","zygosity":"no_information"
"""
  return prompt

def database_process_one(varId, vid, clinvar_submissions_dir, params,
                         url, max_try, truncate, overwrite=False):
    """Fetch ClinVar submission XML for one variant. Atomic write."""
    output_path = clinvar_submissions_dir / f"{varId}.pkl"
    if output_path.exists() and not overwrite:
        return 1
    
    params = dict(params)  # don't mutate caller's dict
    params['id'] = str(vid)
    
    cal = Call_NCBI_API(url=url, params=params, max_try=max_try)
    if cal is None:
        return 0
    
    blocks = []
    for cas in cal.findall("./ClinicalAssertion"):
        lines = _element_to_lines(cas, depth=0)
        block = "\n".join(lines)
        if truncate and len(block) > truncate:
            block = block[:truncate] + "..."
        blocks.append(block)
    
    tmp = clinvar_submissions_dir / f".{varId}.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(blocks, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, output_path)
    return 1

def call_ollama_one(model_name, ollama_url, temperature,
                    llm_res_dir, clinvar_submissions_dir,
                    varId, HGVSc_core, overwrite=False):
    """Call LLM for one variant. Atomic write of both txt and json outputs."""
    output_txt = llm_res_dir / "txt" / f"{varId}.txt"
    output_json = llm_res_dir / "json" / f"{varId}.json"
    if output_json.exists() and not overwrite:
        return 1
    
    pkl_path = clinvar_submissions_dir / f"{varId}.pkl"
    if not pkl_path.exists():
        # Upstream NCBI fetch missed this variant; skip
        return 0
    
    with open(pkl_path, "rb") as f:
        block = pickle.load(f)
    
    prompt_text = build_prompt_en(var_key=varId, HGVSc_core=HGVSc_core, preview_blocks=block)
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    r = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    llm_output = data.get("response", "").strip()
    obj = json.loads(llm_output)
    
    conclusion = obj.get('conclusion', 'Neutral')
    zygosity = obj.get('zygosity', 'no information')
    reasoning_line = obj.get('reasoning',
        'Parsed fields using relaxed rules; missing/invalid fields defaulted conservatively.')
    final_text = f"{reasoning_line}\nConclusion: {conclusion}\nZygosity: {zygosity}\n"
    
    # Atomic write txt
    tmp_txt = output_txt.parent / f".{output_txt.name}.tmp"
    tmp_txt.write_text(final_text, encoding="utf-8")
    os.replace(tmp_txt, output_txt)
    
    # Atomic write json
    tmp_json = output_json.parent / f".{output_json.name}.tmp"
    obj.setdefault('reasoning', 'parse_error')
    obj.setdefault('conclusion', 'Neutral')
    obj.setdefault('zygosity', 'no information')
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp_json, output_json)
    
    return 1

def clinvar_process_one(sample_id, candidates_path, output_path, ClinVar, overwrite=False):
    """Filter one sample's candidates by ClinVar membership. Atomic write."""
    if output_path.exists() and not overwrite:
        return 1
    
    data = pd.read_feather(candidates_path, columns=['varId', 'HGVSc_core'])
    data = data.merge(ClinVar, on='varId', how='inner')
    
    tmp = output_path.parent / f".{output_path.name}.tmp"
    data.to_feather(tmp)
    os.replace(tmp, output_path)
    return 1

def RunAgent(samplesheet, work_dir, references, llm_config,
             ncbi_email=None, ncbi_api_key=None,
             config=None, overwrite=False):
    """
    Run DNA Database Agent: filter candidates with ClinVar submissions,
    fetch ClinVar evidence from NCBI, and evaluate using LLM.
    
    Pipeline:
      1. Filter candidates by ClinVar membership (per sample)
      2. Merge ClinVar-relevant variants across samples (dedup)
      3. Fetch ClinVar submission XML from NCBI API
      4. Evaluate each variant's evidence with LLM (ollama)
    
    Args:
        samplesheet: Must contain sampleID + candidates_path columns.
        work_dir: Pipeline work directory.
        references: AimReferences (uses references.rarecollab.clinvar_feather).
        llm_config: Dict with keys model_name, ollama_url, temperature.
        ncbi_email: NCBI account email (optional, increases rate limit).
        ncbi_api_key: NCBI API key (optional, increases rate limit).
        config: Optional worker config dict.
        overwrite: Re-run even if outputs exist.
    """
    default_config = {"database_clinvar_filter_workers": 1}
    cfg = {**default_config, **(config or {})}
    
    work_dir = Path(work_dir)
    
    if "candidates_path" not in samplesheet.columns:
        raise ValueError(
            "samplesheet missing 'candidates_path' column. "
            "Run DiagnosticEngine.Candidates first."
        )
    
    # Output dirs
    output_root = work_dir / "Agents" / "Database"
    clinvar_filtered_dir = output_root / "ClinVarFiltered"
    clinvar_submissions_dir = output_root / "ClinVarVariants"
    llm_res_dir = output_root / "AgentEvaluation"
    
    for d in [clinvar_filtered_dir, clinvar_submissions_dir,
              llm_res_dir / "txt", llm_res_dir / "json"]:
        d.mkdir(parents=True, exist_ok=True)
    
    # NCBI request params
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    params = {
        "db": "clinvar", "rettype": "vcv", "retmode": "xml",
        "is_variationid": "true",
        "tool": "clinvar_submissions_fetcher",
    }
    if ncbi_email:
        params["email"] = ncbi_email
    if ncbi_api_key:
        params["api_key"] = ncbi_api_key
    
    NCBI_WORKERS = 2  # NCBI rate limit
    LLM_WORKERS = llm_config.get("num_parallel", 1)   # match OLLAMA_NUM_PARALLEL
    MAX_TRY = 3
    TRUNCATE = 3000
    
    # ===== Step 1: Filter candidates by ClinVar =====
    print("1. Filtering candidates by ClinVar membership ...")
    ClinVar = pd.read_feather(references.rarecollab.clinvar_feather)
    
    output_paths = {}
    futures_map = {}
    
    with ThreadPoolExecutor(max_workers=cfg["database_clinvar_filter_workers"]) as ex:
        for row in samplesheet.itertuples(index=True):
            sample_id = row.sampleID
            candidates_path = Path(row.candidates_path)
            output_path = clinvar_filtered_dir / f"{sample_id}.feather"
            
            fut = ex.submit(
                clinvar_process_one, sample_id, candidates_path,
                output_path, ClinVar, overwrite,
            )
            futures_map[fut] = (row.Index, sample_id, output_path)
        
        ok = fail = 0
        with tqdm(total=len(futures_map), desc="Scanning ClinVar Submission") as pbar:
            for fut in as_completed(futures_map):
                row_idx, sample_id, out_path = futures_map[fut]
                try:
                    fut.result()
                    output_paths[row_idx] = str(out_path)
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] ClinVar filter failed for {sample_id}: "
                          f"{type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(Processed=ok, Fail=fail)
    
    if fail > 0:
        raise RuntimeError(f"ClinVar filter failed for {fail} sample(s).")
    
    # ===== Step 2: Merge variants across samples =====
    print("2. Merging ClinVar-relevant variants across samples ...")
    merged = pd.concat(
        [pd.read_feather(p) for p in output_paths.values()],
        ignore_index=True, copy=False,
    )
    merged = merged.drop_duplicates(subset=['varId'], keep='first').reset_index(drop=True)
    input_tuples = list(merged[["varId", "VariationID", "HGVSc_core"]].itertuples(index=False, name=None))
    print(f"  Unique ClinVar-annotated variants: {len(input_tuples)}")
    
    # ===== Step 3: Fetch ClinVar submissions from NCBI =====
    print("3. Fetching ClinVar submission XML from NCBI ...")
    with ThreadPoolExecutor(max_workers=NCBI_WORKERS) as ex:
        futures = [
            ex.submit(database_process_one,
                      varId, vid, clinvar_submissions_dir,
                      params, url, MAX_TRY, TRUNCATE, overwrite)
            for (varId, vid, _) in input_tuples
        ]
        ok = fail = 0
        with tqdm(total=len(futures), desc="Calling ClinVar - NCBI") as pbar:
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    ok += int(ret == 1)
                    fail += int(ret == 0)
                except Exception:
                    fail += 1
                pbar.update(1)
                pbar.set_postfix(Retrieved=ok, Failed=fail)
    
    print(f"  {ok}/{len(input_tuples)} ClinVar submissions downloaded.")
    
    # ===== Step 4: Evaluate with LLM =====
    print(f"4. Evaluating evidence with LLM (model={llm_config['model_name']}) ...")
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        futures = [
            ex.submit(call_ollama_one,
                      llm_config["model_name"],
                      llm_config["ollama_url"],
                      llm_config["temperature"],
                      llm_res_dir, clinvar_submissions_dir,
                      varId, HGVSc_core, overwrite)
            for (varId, _, HGVSc_core) in input_tuples
        ]
        ok = fail = 0
        with tqdm(total=len(futures), desc="Evaluating Evidence") as pbar:
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    ok += int(ret == 1)
                    fail += int(ret == 0)
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] LLM eval: {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(Evaluated=ok, Failed=fail)
    
    # ===== Update samplesheet =====
    samplesheet = samplesheet.copy()
    samplesheet["clinvar_filtered_path"] = None
    for row_idx, path in output_paths.items():
        samplesheet.loc[row_idx, "clinvar_filtered_path"] = path
    
    # Database agent root path (for downstream: var-level outputs glob from this)
    samplesheet["database_agent_root"] = str(output_root)
    
    # Save samplesheet (atomic)
    samplesheet_path = work_dir / "samplesheet_with_paths.csv"
    tmp = work_dir / ".samplesheet_with_paths.csv.tmp"
    samplesheet.to_csv(tmp, index=False)
    os.replace(tmp, samplesheet_path)
    print(f"Updated samplesheet: {samplesheet_path}")
    
    print(f"--Database Agent DONE--\n")
    return samplesheet