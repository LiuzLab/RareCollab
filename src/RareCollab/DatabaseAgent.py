#!/usr/bin/env python
# coding: utf-8

import time
import requests
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
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

def database_process_one(varId, vid, clinvar_submission_path, params, url, max_try, truncate) -> None:
    save_file = Path(clinvar_submission_path + "/" + varId + ".pkl")
    if save_file.exists():
        return 1
    params['id'] = str(vid)
    #Call NCBI API:
    cal = Call_NCBI_API(url = url, params = params, max_try = max_try)
    if cal is None:
        print(f"Variant - {varId} - with id:{vid} NOT found on NCBI - Skip")
        return 0
    else:
        #Curate the information:
        blocks = []
        for cas in cal.findall("./ClinicalAssertion"):
            lines = _element_to_lines(cas, depth=0)
            block = "\n".join(lines)
            if truncate and len(block) > truncate:
                block = block[:truncate] + "..."
            blocks.append(block)
        with open(save_file, "wb") as f:
            pickle.dump(blocks, f, protocol=pickle.HIGHEST_PROTOCOL)
        return 1

def call_ollama_one(MODEL_NAME, OLLAMA_URL, TEMPERATURE, LLM_res_path, clinvar_submission_path, varId, HGVSc_core):
    output_txt_path = LLM_res_path + '/txt/' + varId + '.txt'
    output_json_path = LLM_res_path + '/json/' + varId + '.json'
    if Path(output_json_path).exists():
        return 1
    with open(f"{clinvar_submission_path}/{varId}.pkl", "rb") as f: block = pickle.load(f)
    prompt_text = build_prompt_en(var_key = varId, HGVSc_core = HGVSc_core, preview_blocks = block)
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": float(TEMPERATURE)},
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    llm_output = data.get("response", "").strip()
    obj = json.loads(llm_output)
    conclusion = obj['conclusion']if 'conclusion' in obj else 'Neutral'
    zygosity = obj['zygosity']if 'zygosity' in obj else 'no information'
    reasoning_line = obj['reasoning']if 'reasoning' in obj else 'Parsed fields using relaxed rules; missing/invalid fields defaulted conservatively.'
    final_text = f"{reasoning_line}\nConclusion: {conclusion}\nZygosity: {zygosity}\n"
    Path(output_txt_path).write_text(final_text, encoding="utf-8")
    with Path(output_json_path).open("w", encoding="utf-8") as f: json.dump(obj, f)
    return 1

def clinvar_process_one(sample_id, sample_path, output_path, ClinVar) -> None:
    save_file = Path(output_path + '/' + sample_id + '.feather')
    if save_file.exists():
        return 1
    data = pd.read_feather(sample_path, columns = ['varId','HGVSc_core'])
    data = data.merge(ClinVar, on = 'varId', how = 'inner')
    data.to_feather(save_file)
    return 1

def RunAgent(work_path, ClinVar_path, NCBI_EMAIL, NCBI_KEY, MODEL_NAME, OLLAMA_URL, TEMPERATURE, max_workers = 5):
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    TOOL = "clinvar_submissions_fetcher"
    truncate = 3000
    max_try = 3
    params = {"db": "clinvar",
            "rettype": "vcv",
            "retmode": "xml",
            "is_variationid":"true",
            "tool": TOOL, 
            "email": NCBI_EMAIL,
            "api_key": NCBI_KEY}
    
    output_path = work_path + "/Agents/Database/"
    ClinVarVariants_path = output_path + 'ClinVarVariants/'
    ClinVarFiltered_path = output_path + 'ClinVarFiltered/'
    LLM_res_path = output_path + 'AgentEvaluation/'
    
    #Creat root path:
    print(f"Creating work dir ~{Path(output_path)}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(ClinVarVariants_path).mkdir(parents=True, exist_ok=True)
    Path(ClinVarFiltered_path).mkdir(parents=True, exist_ok=True)
    Path(LLM_res_path).mkdir(parents=True, exist_ok=True)
    Path(LLM_res_path + 'txt').mkdir(parents=True, exist_ok=True)
    Path(LLM_res_path + 'json').mkdir(parents=True, exist_ok=True)

    #1.Filter Candidates with ClinVar submissions:
    print(f"Loading ClinVar Documents ...")
    ClinVar = pd.read_feather(ClinVar_path)
    input_path = work_path + '/Diagnostic_results/Candidates/'
    if not Path(input_path).exists():
        raise ValueError(f"Input file NOT detected, please run 'DiagnosticEngine.Candidates' first and check the work_path ... ")
    SampleID_Path = {p.name.split('_nomcand.feather')[0]: p for p in Path(input_path).iterdir()}
    #Process samples in parallel:
    print(f'Sample Processing ...')
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sample_id, sample_path in SampleID_Path.items():
            futures.append(ex.submit(clinvar_process_one, sample_id, sample_path, ClinVarFiltered_path, ClinVar))
        ok = fail = 0
        with tqdm(total=len(SampleID_Path), desc="Scanning ClinVar Submission") as pbar:
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    ok += int(ret == 1)
                    fail += int(ret == 0)
                except Exception:
                    fail += 1
                pbar.update(1)
                pbar.set_postfix(Processed=ok, Fail=fail)

    print(f'--Detected Variant With ClinVar Submission--\n')

    #2. Merge Variants:
    SampleInputs = {p.name.split(".feather")[0]:p for p in Path(ClinVarFiltered_path).iterdir()}
    print(f"Loading Variants ...")
    merged_data = []
    for _, sample_path in SampleInputs.items():
        merged_data.append(pd.read_feather(sample_path))
    merged_data = pd.concat(merged_data, ignore_index=True, copy=False)
    merged_data = merged_data.drop_duplicates(subset = ['varId'], keep = 'first').reset_index(drop = True)
    input_tuples = list(merged_data[["varId", "VariationID", "HGVSc_core"]].itertuples(index=False, name=None))

    #3. Call ClinVar API:
    print(f"Calling ClinVar API ...")
    futures = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for (varId, vid, _) in input_tuples:
            futures.append(ex.submit(database_process_one, varId, vid, ClinVarVariants_path, params, url, max_try, truncate))
        ok = fail = 0
        with tqdm(total=len(input_tuples), desc="Calling ClinVar - NCBI") as pbar:
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    ok += int(ret == 1)
                    fail += int(ret == 0)
                except Exception:
                    fail += 1
                pbar.update(1)
                pbar.set_postfix(Retreived=ok, Failed=fail)
    print(f'--ClinVar Documents Preprocessing DONE--\n')
    print(f"{sum(f.result() for f in as_completed(futures))}/{len(merged_data)} ClinVar Submissions are downloaded ...")

    #4. Using LLM:
    print(f"Database Agent (Model Name:{MODEL_NAME}) is working on ClinVar Data Records ...")
    futures = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for (varId, _, HGVSc_core) in input_tuples:
            futures.append(ex.submit(call_ollama_one, MODEL_NAME, OLLAMA_URL, TEMPERATURE, LLM_res_path, ClinVarVariants_path, varId, HGVSc_core))
        ok = fail = 0
        with tqdm(total=len(input_tuples), desc="Evaluating Evidence") as pbar:
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    ok += int(ret == 1)
                    fail += int(ret == 0)
                except Exception:
                    fail += 1
                pbar.update(1)
                pbar.set_postfix(Evaluated=ok, Failed=fail)
    print(f"--Database Agent's Work is DONE--\n")