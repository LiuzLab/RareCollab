#!/usr/bin/env python
# coding: utf-8

import os
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def agg_worst_value(col: str, s: pd.Series, fathmm_mode: str) -> float:
    vals = pd.to_numeric(s, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    if col in {"SIFT_score"}:
        return float(vals.min())
    if col in {"LRT_score"}:
        return float(vals.min())
    if col in {"LRT_Omega"}:
        return float(vals.min())
    if col in {"FATHMM_score"}:
        return float(vals.min()) if fathmm_mode == "neg_is_damaging" else float(vals.max())
    return float(vals.max())

def first_nonempty(s: pd.Series) -> str:
    for x in s.dropna():
        xs = str(x).strip()
        if xs and xs.lower() not in {"nan", "none", "null"}:
            return xs
    return ""

def fmt_score(x, nd=4):
    if pd.isna(x):
        return "NA"
    x = float(x)
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")

def any_pos(s: pd.Series) -> int:
    return int(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any())

def build_prompt_en(identifier: str, variant_key: str, gene: str, cdna_text: str, protein_text: str, score_lines: list, fathmm_note: str, include_spliceai: bool) -> str:
    # ✅ Candidate variant 结构（按你要求）
    candidate_lines = [f"- Gene: {gene}"]
    if cdna_text and protein_text != "-":
        candidate_lines.append(f"- cDNA change: {cdna_text}")
    if protein_text and protein_text != "-":
        candidate_lines.append(f"- protein change: {protein_text}")

    splice_guidance = ""
    if include_spliceai:
        splice_guidance = ("9) spliceAImax: 0-1, higher means more likely splice-altering. Heuristic bands:\n"
                            "   >=0.2 (high recall), >=0.5 (recommended), >=0.8 (high precision).\n"
                            )

    strong_predictor_list = "CADD, REVEL/M-CAP, SIFT/PolyPhen/MutationAssessor"
    if include_spliceai:
        strong_predictor_list += ", spliceAI"

    strong_or_clause = ""
    if include_spliceai:
        strong_or_clause = ("          OR spliceAImax is high (>=0.5, especially >=0.8) together with additional supporting scores.\n")

    prompt = f"""You are a clinical variant interpretation assistant.

Task:
Based ONLY on the in-silico prediction and conservation scores provided (no phenotype, no population frequency, no ClinVar, no segregation, no functional assay),
assess how strongly these scores support that the variant is pathogenic in general.

Candidate variant:
{chr(10).join(candidate_lines)}

Scores (worst-case per score across duplicate rows of the same identifier+variant_key; NA means missing):
{chr(10).join(score_lines)}

Score interpretation guidance (use as heuristics; do NOT treat any single score as definitive):
1) CADD (PHRED-like): higher means more deleterious. Rough percentile meaning: >=10 ~ top 10%, >=20 ~ top 1%, >=30 ~ top 0.1%.
   Treat CADD_phred and CADD_PHRED as the same kind of score; prefer the higher one when both exist.
2) REVEL: 0-1, higher more likely pathogenic for missense. Common thresholds: >=0.5 suggests pathogenic tendency; >=0.75 is more stringent.
3) M-CAP: higher more likely pathogenic for missense; >0.025 is a commonly used cutoff for "potentially pathogenic".
4) SIFT: 0-1, LOWER is more damaging; <0.05 is commonly considered deleterious.
5) PolyPhen-2: 0-1, higher is more damaging; roughly:
   - probably damaging >=0.909
   - possibly damaging 0.446-0.908
   - benign <=0.445
6) MutationAssessor: higher indicates larger functional impact; around:
   neutral <=0.8, low 0.8-1.9, medium 1.9-3.5, high >3.5.
7) DANN: 0-1, higher more deleterious; values >0.9 often treated as deleterious.
8) FATHMM-MKL (coding): 0-1, higher more deleterious; 0.5 is a common default threshold.
{splice_guidance}10) Conservation:
   - GERPpp_RS: larger positive means more evolutionarily constrained (often RS>=2 considered constrained).
   - phyloP100way_vertebrate: positive means conservation, negative means acceleration.
   Conservation alone is SUPPORTING context, not decisive evidence.
11) LRT:
   - LRT_score is a two-sided p-value for codon constraint (smaller supports constraint).
   - LRT_Omega is dN/dS (omega); omega < 1 suggests constraint.
12) {fathmm_note}

How to assign the 3-tier conclusion:
- Strong: multiple independent predictors (e.g., {strong_predictor_list}) are jointly in clearly damaging bands with minimal contradictions;
{strong_or_clause}- Moderate: some consistent damaging signals (e.g., CADD >=20 plus at least one strong missense predictor like REVEL>=0.5 / M-CAP>0.025 / SIFT<0.05 / PolyPhen probably damaging),
            but with missingness or mild contradictions.
- Weak: mostly benign/low scores, mixed or contradictory signals, or too much missing data to argue strongly.

Output format MUST be valid JSON on a single line. No markdown. No code fences. No extra keys.
Schema (keys must match exactly):{{"Reasoning":"<brief but concrete reasoning grounded in the numbers and directions>","Conclusion":"<Weak|Moderate|Strong>"}}

If you cannot comply, output exactly:{{"Reasoning":"parse_error","Conclusion":"Weak"}}
"""
    return prompt

# ===== Per-variant LLM worker =====

def insilico_llm_one(model_name, ollama_url, temperature,
                     llm_res_dir, identifier, varId, row,
                     display_order, insert_pos, fathmm_note,
                     overwrite=False):
    """
    Call LLM for one (identifier, varId) row. Atomic write of txt + json.
    """
    output_dir = llm_res_dir / identifier
    output_dir.mkdir(parents=True, exist_ok=True)
    output_txt = output_dir / f"{varId}.txt"
    output_json = output_dir / f"{varId}.json"

    if output_json.exists() and not overwrite:
        return 1

    include_spliceai = int(row.is_splice_cons) > 0
    include_keys = (display_order[:insert_pos] + ["spliceAImax"] + display_order[insert_pos:]
                    if include_spliceai else display_order)
    score_lines = [f"- {k}: {fmt_score(getattr(row, k))}" for k in include_keys]

    gene = str(row.geneSymbol)
    cdna_text = str(row.HGVSc)
    protein_text = str(row.HGVSp)

    prompt_text = build_prompt_en(
        identifier=identifier, variant_key=varId,
        gene=gene, cdna_text=cdna_text, protein_text=protein_text,
        score_lines=score_lines, fathmm_note=fathmm_note,
        include_spliceai=include_spliceai,
    )

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

    conclusion = obj.get('Conclusion', 'Weak')
    reasoning_line = obj.get('Reasoning', 'parse_error')
    final_text = f"{reasoning_line}\nConclusion: {conclusion}\n"

    # Atomic write
    tmp_txt = output_txt.parent / f".{output_txt.name}.tmp"
    tmp_txt.write_text(final_text, encoding="utf-8")
    os.replace(tmp_txt, output_txt)

    tmp_json = output_json.parent / f".{output_json.name}.tmp"
    obj.setdefault('Reasoning', 'parse_error')
    obj.setdefault('Conclusion', 'Weak')
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp_json, output_json)
    return 1


def RunAgent(samplesheet, work_dir, llm_config, overwrite=False):
    """
    Run InSilico Agent on candidates from each sample.

    Pipeline:
      1. Per-sample: filter candidates by rank_InSilico <= threshold,
         aggregate scores per (identifier, varId), produce one merged feather.
      2. Per-(identifier, varId): call LLM, write txt + json under
         <work_dir>/Agents/InSilico/AgentEvaluation/<identifier>/<varId>.{txt,json}

    Args:
        samplesheet: Must contain sampleID, candidates_path,
            vartogene_feather_path columns.
        work_dir: Pipeline work directory.
        llm_config: Dict with keys model_name, ollama_url, temperature, num_parallel.
        config: Optional worker config dict. Recognized keys:
            insilico_merge_workers (int, default 1)
        overwrite: Re-run even if outputs exist.
    """

    work_dir = Path(work_dir)

    for col in ("candidates_path", "vartogene_feather_path"):
        if col not in samplesheet.columns:
            raise ValueError(
                f"samplesheet missing '{col}' column. "
                f"Run upstream pipeline steps first."
            )

    DE_RANKING_THRESHOLD = 100
    SCORE_COLS = [
        "CADD_phred", "CADD_PHRED", "DANN_score", "REVEL_score", "fathmm_MKL_coding_score",
        "Polyphen2_HDIV_score", "Polyphen2_HVAR_score", "SIFT_score",
        "FATHMM_score", "M_CAP_score", "MutationAssessor_score", "spliceAImax",
        "GERPpp_RS", "LRT_Omega", "LRT_score", "phyloP100way_vertebrate",
    ]
    SPLICE_CONS_COLS = ["cons_splice_acceptor_variant", "cons_splice_donor_variant"]
    keep_cols = (
        ["varId", "identifier", "geneSymbol", "HGVSc", "HGVSp"]
        + SCORE_COLS + SPLICE_CONS_COLS
    )

    DISPLAY_ORDER = [
        "CADD_phred", "CADD_PHRED", "CADD_worst", "DANN_score", "REVEL_score", "M_CAP_score",
        "Polyphen2_HDIV_score", "Polyphen2_HVAR_score", "SIFT_score", "MutationAssessor_score",
        "FATHMM_score", "fathmm_MKL_coding_score", "GERPpp_RS", "phyloP100way_vertebrate",
        "LRT_score", "LRT_Omega",
    ]
    insert_pos = DISPLAY_ORDER.index('GERPpp_RS')

    LLM_WORKERS = llm_config.get("num_parallel", 1)

    output_root = work_dir / "Agents" / "InSilico"
    merged_dir = output_root / "MergedVariants"
    llm_res_dir = output_root / "AgentEvaluation"
    merged_dir.mkdir(parents=True, exist_ok=True)
    llm_res_dir.mkdir(parents=True, exist_ok=True)
    merged_path = merged_dir / "MergedVariants.feather"

    # ===== Step 1: Merge per-sample candidates into one variant table =====
    if merged_path.exists() and not overwrite:
        print("1. Merged file detected, loading ...")
        merged_table = pd.read_feather(merged_path)
    else:
        print("1. Aggregating candidates per sample ...")
        per_sample_dfs = []
        pbar = tqdm(samplesheet.itertuples(index=True), total=len(samplesheet),
                    desc="Merging candidates")
        for row in pbar:
            sample_id = row.sampleID
            pbar.set_postfix(sample=sample_id)
            
            feature_table = pd.read_feather(row.vartogene_feather_path, columns=keep_cols)
            cand_table = pd.read_feather(row.candidates_path)
            cand_table = cand_table[cand_table['rank_InSilico'] <= DE_RANKING_THRESHOLD]
            cand_table = feature_table[feature_table['varId'].isin(cand_table['varId'])].reset_index(drop=True)
            
            min_val = pd.to_numeric(cand_table["FATHMM_score"], errors="coerce").min()
            fathmm_mode = "neg_is_damaging" if (pd.notna(min_val) and min_val < 0) else "high_is_damaging"
            
            agg_dict = {}
            for col in ["geneSymbol", "HGVSc", "HGVSp"]:
                agg_dict[col] = first_nonempty
            for col in SCORE_COLS:
                agg_dict[col] = (lambda s, col=col: agg_worst_value(col, s, fathmm_mode))
            agg_dict["cons_splice_acceptor_variant"] = any_pos
            agg_dict["cons_splice_donor_variant"] = any_pos
            
            agg_df = cand_table.groupby("varId", sort=False, dropna=False).agg(agg_dict).reset_index()
            agg_df["is_splice_cons"] = ((agg_df["cons_splice_acceptor_variant"].astype(int) |
                                        agg_df["cons_splice_donor_variant"].astype(int)).astype(int))
            agg_df = agg_df.drop(columns=SPLICE_CONS_COLS)
            agg_df["CADD_worst"] = agg_df[["CADD_phred", "CADD_PHRED"]].max(axis=1, skipna=True)
            agg_df['identifier'] = sample_id
            per_sample_dfs.append(agg_df)
        
        merged_table = pd.concat(per_sample_dfs, ignore_index=True, copy=False)
        
        # Atomic write
        tmp = merged_dir / f".{merged_path.name}.tmp"
        merged_table.to_feather(tmp)
        os.replace(tmp, merged_path)
        print(f"  Merged {len(merged_table)} variants across {len(per_sample_dfs)} samples.")

    # ===== Step 2: Determine FATHMM mode globally =====
    min_val = pd.to_numeric(merged_table["FATHMM_score"], errors="coerce").min()
    fathmm_mode = "neg_is_damaging" if (pd.notna(min_val) and min_val < 0) else "high_is_damaging"
    fathmm_note = (
        "FATHMM_score direction: more negative = more damaging (as observed in this dataset)."
        if fathmm_mode == "neg_is_damaging"
        else "FATHMM_score direction: higher = more damaging (as observed in this dataset)."
    )

    # ===== Step 3: Run LLM in parallel per (identifier, varId) =====
    print(f"2. Evaluating variants by InSilico Agent (model={llm_config['model_name']}) ...")
    print(f"   LLM workers: {LLM_WORKERS}")

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        futures = [
            ex.submit(
                insilico_llm_one,
                llm_config["model_name"],
                llm_config["ollama_url"],
                llm_config["temperature"],
                llm_res_dir,
                row.identifier, row.varId, row,
                DISPLAY_ORDER, insert_pos, fathmm_note,
                overwrite,
            )
            for row in merged_table.itertuples(index=False)
        ]
        ok = fail = 0
        with tqdm(total=len(futures), desc="Evaluating Evidence") as pbar:
            for fut in as_completed(futures):
                try:
                    fut.result()
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] InSilico LLM: {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(Evaluated=ok, Failed=fail)

    # ===== Update samplesheet =====
    samplesheet = samplesheet.copy()
    samplesheet["insilico_agent_root"] = str(output_root)

    samplesheet_path = work_dir / "samplesheet_with_paths.csv"
    tmp = work_dir / ".samplesheet_with_paths.csv.tmp"
    samplesheet.to_csv(tmp, index=False)
    os.replace(tmp, samplesheet_path)
    print(f"Updated samplesheet: {samplesheet_path}")

    print("--InSilico Agent DONE--\n")
    return samplesheet