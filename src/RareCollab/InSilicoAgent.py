#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from tqdm.notebook import tqdm

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

def RunAgent(work_path, MODEL_NAME, OLLAMA_URL, TEMPERATURE):
    DE_RANKING_THRESHOLD = 100
    SCORE_COLS = [
        "CADD_phred", "CADD_PHRED", "DANN_score", "REVEL_score", "fathmm_MKL_coding_score",
        "Polyphen2_HDIV_score", "Polyphen2_HVAR_score", "SIFT_score",
        "FATHMM_score", "M_CAP_score", "MutationAssessor_score", "spliceAImax",
        "GERPpp_RS", "LRT_Omega", "LRT_score", "phyloP100way_vertebrate"]
    SPLICE_CONS_COLS = ["cons_splice_acceptor_variant", "cons_splice_donor_variant"]
    GENE_COL = "geneSymbol"
    HGVSC_COL = "HGVSc"
    HGVSP_COL = "HGVSp"

    keep_cols = ["varId","identifier","geneSymbol", "HGVSc", "HGVSp"] + SCORE_COLS + SPLICE_CONS_COLS
    #1. Confirm data (Candidate list):
    input_path = work_path + '/Diagnostic_results/Candidates/'
    if not Path(input_path).exists():
        raise ValueError(f"Input file NOT detected, please run 'DiagnosticEngine.Candidates' first and check the work_path ... ")

    #2. Merged File:
    print(f"Creating work dir ~{Path(work_path + '/Agents/InSilico/')}")
    Path(work_path + '/Agents/InSilico/MergedVariants/').mkdir(parents=True, exist_ok=True)
    MergedFile_Path = Path(work_path + '/Agents/InSilico/MergedVariants/MergedVariants.feather')
    #check if file has been merged already?
    if not MergedFile_Path.exists():
        Cand_Path = {p.name.split('_nomcand.feather')[0]:p for p in Path(input_path).iterdir()}
        Sample_Path = {p.name.split('.feather')[0]:p for p in Path(work_path + '/AIM_Preprocess/VarToGene/out').iterdir()}
        #Merge Cand_table:
        print('Merging Files')
        merged_table = []
        pbar = tqdm(Cand_Path.keys(), desc="Merge Table", total=len(Cand_Path))
        for sample_id in pbar:
            pbar.set_postfix(sample=sample_id)
            feature_table = pd.read_feather(Sample_Path[sample_id], columns = keep_cols)
            cand_table = pd.read_feather(Cand_Path[sample_id])
            cand_table = cand_table[cand_table['rank_InSilico'] <= DE_RANKING_THRESHOLD]
            cand_table = feature_table[feature_table['varId'].isin(cand_table['varId'])].reset_index(drop=True)

            #determine the damge direction
            min_val = pd.to_numeric(cand_table["FATHMM_score"], errors="coerce").min()
            fathmm_mode = "neg_is_damaging" if (pd.notna(min_val) and min_val < 0) else "high_is_damaging"

            #Rule of how to choose the value:
            agg_dict = {}
            # a) gene / HGVS: first non-empty
            for col in ["geneSymbol", "HGVSc", "HGVSp"]:
                agg_dict[col] = first_nonempty

            # b) score: worst-case
            for col in SCORE_COLS:
                agg_dict[col] = (lambda s, col=col: agg_worst_value(col, s, fathmm_mode))

            agg_dict["cons_splice_acceptor_variant"] = any_pos
            agg_dict["cons_splice_donor_variant"]    = any_pos

            # 2) compress multiple same var_id to one row:
            agg_df = cand_table.groupby("varId", sort=False, dropna=False).agg(agg_dict).reset_index()
            agg_df["is_splice_cons"] = ((agg_df["cons_splice_acceptor_variant"].astype(int) |
                                        agg_df["cons_splice_donor_variant"].astype(int)).astype(int))
            agg_df = agg_df.drop(columns=["cons_splice_acceptor_variant", "cons_splice_donor_variant"])
            agg_df["CADD_worst"] = agg_df[["CADD_phred", "CADD_PHRED"]].max(axis=1, skipna=True)
            agg_df['identifier'] = sample_id
            merged_table.append(agg_df)
        #Save Merged File:    
        merged_table = pd.concat(merged_table, ignore_index=True, copy=False)
        merged_table.to_feather(MergedFile_Path)
    else:
        print('Merged File Detected. Loading Merged File ...')
        merged_table = pd.read_feather(MergedFile_Path)

    #3 Run LLM:
    print(f"Generating Prompts and Evaluating Variants by InSilico Agent ...")
    output_path = work_path + '/Agents/InSilico/AgentEvaluation/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    DISPLAY_ORDER = ["CADD_phred", "CADD_PHRED", "CADD_worst","DANN_score", "REVEL_score", "M_CAP_score",
                    "Polyphen2_HDIV_score", "Polyphen2_HVAR_score", "SIFT_score","MutationAssessor_score",
                    "FATHMM_score", "fathmm_MKL_coding_score","GERPpp_RS", "phyloP100way_vertebrate","LRT_score", "LRT_Omega"]
    insert_pos = DISPLAY_ORDER.index('GERPpp_RS')
    min_val = pd.to_numeric(merged_table["FATHMM_score"], errors="coerce").min()
    fathmm_mode = "neg_is_damaging" if (pd.notna(min_val) and min_val < 0) else "high_is_damaging"
    fathmm_note = "FATHMM_score direction: more negative = more damaging (as observed in this dataset)." if fathmm_mode == "neg_is_damaging" else "FATHMM_score direction: higher = more damaging (as observed in this dataset)."
    pbar = tqdm(merged_table.itertuples(index=False), desc="Evaluating Evidence", total=len(merged_table))
    for row in pbar:
        identifier = row.identifier
        varId = row.varId
        pbar.set_postfix(sample=identifier)
        Path(output_path + "/" + identifier).mkdir(parents=True, exist_ok=True)
        save_file_txt = Path(output_path + "/" + identifier + "/" + varId + ".txt")
        save_file_json = Path(output_path + "/" + identifier + "/" + varId + ".json")
        if save_file_json.exists():
            continue
        include_spliceai = int(row.is_splice_cons) > 0
        include_keys = DISPLAY_ORDER[:insert_pos] + ["spliceAImax"] + DISPLAY_ORDER[insert_pos:] if include_spliceai else DISPLAY_ORDER
        score_lines = [f"- {k}: {fmt_score(getattr(row, k))}" for k in include_keys]

        gene = str(row.geneSymbol)
        cdna_text = str(row.HGVSc)
        protein_text = str(row.HGVSp)

        prompt_text = build_prompt_en(identifier=identifier, variant_key=varId, gene=gene, cdna_text=cdna_text, protein_text=protein_text,
                                    score_lines=score_lines, fathmm_note=fathmm_note, include_spliceai = include_spliceai)
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
        conclusion = obj['Conclusion'] if 'Conclusion' in obj else 'Weak'
        reasoning_line = obj['Reasoning'] if 'Reasoning' in obj else 'parse_error'
        final_text = f"{reasoning_line}\nConclusion: {conclusion}\n"

        Path(save_file_txt).write_text(final_text, encoding="utf-8")
        with Path(save_file_json).open("w", encoding="utf-8") as f: json.dump(obj, f)
    print(f"--InSilico Agent's Work is DONE--\n")