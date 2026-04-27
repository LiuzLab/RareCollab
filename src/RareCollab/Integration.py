#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import json
import pysam

from pathlib import Path
from collections import defaultdict
from itertools import combinations
from tqdm.notebook import tqdm

def load_LLM_res(sub_root_path, model_type, candidate_type):
    res = []
    label_reason, label_conclusion = model_type + "_Reasoning", model_type + "_Conclusion"
    loading_list = {p.name.split('.json')[0]:p for p in Path(sub_root_path).iterdir() if p.suffix == '.json'}
    for GeneName, loading_path in loading_list.items():
        with loading_path.open("r", encoding = 'utf-8') as f:Evaluation = json.load(f)
        res.append([GeneName, Evaluation['Reasoning'], Evaluation['Conclusion']])
    res = pd.DataFrame(res, columns = [candidate_type,label_reason, label_conclusion])
    return res

def load_DatabaseLLM_res(sub_root_path):
    res = []
    loading_list = {p.name.split('.json')[0]:p for p in Path(sub_root_path).iterdir() if p.suffix == '.json'}
    for GeneName, loading_path in loading_list.items():
        with loading_path.open("r", encoding = 'utf-8') as f:Evaluation = json.load(f)
        res.append([GeneName, Evaluation['reasoning'], Evaluation['conclusion'], Evaluation['zygosity']])
    res = pd.DataFrame(res, columns = ['varId', 'Database_Reasoning', 'Database_Conclusion', 'Database_Zygosity'])
    return res

def load_RNALLM_res(sub_root_path, model_type, candidate_type):
    res = []
    label_reason, label_event, label_conclusion = model_type + "_Reasoning", model_type + "_Event", model_type + "_Conclusion"
    loading_list = {p.name.split('.json')[0]:p for p in Path(sub_root_path).iterdir() if p.suffix == '.json'}
    for GeneName, loading_path in loading_list.items():
        with loading_path.open("r", encoding = 'utf-8') as f:Evaluation = json.load(f)
        res.append([GeneName, Evaluation['Reasoning'], Evaluation['Event'], Evaluation['Conclusion']])
    res = pd.DataFrame(res, columns = [candidate_type,label_reason, label_event, label_conclusion])
    return res

def best_tandem_repeat(seq: str, min_k: int = 1, max_k: int = 6, min_repeats: int = 3):
    n = len(seq)
    best_k, best_motif, best_rep = 0, "", 0

    for k in range(min_k, min(max_k, n // min_repeats) + 1):
        for i in range(n - k * min_repeats + 1):
            motif = seq[i:i+k]
            if "N" in motif:
                continue

            rep = 1
            j = i + k
            while j + k <= n and seq[j:j+k] == motif:
                rep += 1
                j += k

            if rep >= min_repeats and rep > best_rep:
                best_k, best_motif, best_rep = k, motif, rep

    return best_k, best_motif, best_rep

def TandemRepeatDetection(data, fa):
    MOTIF_MIN_REPEATS = 10     # within +/-30bp window
    W_REPEAT = 30              # +/-30bp (61bp total)
    MIN_K = 1
    MAX_K = 6
    output_df = data[['varId','Chromosome','Pos','HGVSc']]
    #output_df = output_df.loc[output_df["HGVSc"].str.contains(r"del|dup", regex=True, na=False)].reset_index(drop = True)
    sub_res = []
    for row in output_df.itertuples(index=True):
        if "del" not in row.HGVSc and "dup" not in row.HGVSc:
            sub_res.append([0, False, "", 0, 0, ""])
            continue
        chrom = row.Chromosome
        pos = row.Pos
        _, _, ref, alt = row.varId.split('_')
        length = abs(len(ref) - len(alt))
        start = max(0, (pos - 1) - W_REPEAT)
        end = (pos - 1) + W_REPEAT + 1
        try:
            seq = fa.fetch(chrom, start, end).upper()
        except KeyError:
            seq = ""
            sub_res.append([length, False, "", 0, 0, seq])
            continue
        
        k, motif, rep = best_tandem_repeat(seq, min_k=MIN_K, max_k=MAX_K, min_repeats=3)
        k, motif, rep = best_tandem_repeat(seq, min_k=MIN_K, max_k=MAX_K, min_repeats=3)
        sub_res.append([length, rep >= MOTIF_MIN_REPEATS, motif if rep > 0 else "", k if rep > 0 else 0, rep, seq])
    sub_res = pd.DataFrame(sub_res, columns = ['length','has_repeat_motif','best_repeat_motif','best_repeat_k','best_repeat_repeats','seq_pos_pm30'])
    output_df = pd.concat([output_df, sub_res], axis = 1)
    output_df = output_df.drop(columns = ['Chromosome','Pos'])
    return output_df



# -------------------------
# Copeland pairwise majority voting
# -------------------------
PK_PHENO_SCORE = {
    "not fit": 0,
    "partial fit": 1,
    "good fit": 2,
    "strong": 3,
}

PK_CLINVAR_SCORE = {
    "against": 0,
    "neutral": 1,
    "supporting": 2,
    "convincing": 3,
}

PK_RNA_SCORE = {
    "strong rna evidence": 2.0,
    "weak rna evidence": 1.0,
}

PK_INSILICO_SCORE = {
    "weak": 1.0,
    "moderate": 2.0,
    "strong": 3.0,
}


def _pk_norm(x):
    """Normalize LLM labels used by the pairwise module."""
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())

    if s in {"not fit", "no fit"}:
        return "not fit"
    if s in {"partial fit", "partially fit", "partial"}:
        return "partial fit"
    if s in {"good fit", "good"}:
        return "good fit"
    if s in {"stand alone strong evidence", "standalone strong evidence", "strong evidence", "strong"}:
        return "strong"

    if s in {"against", "neutral", "supporting", "convincing"}:
        return s

    if s in {"strong rna evidence", "weak rna evidence", "no rna evidence", "no evidence"}:
        return s

    if s in {"weak", "moderate", "strong"}:
        return s

    return s


def _pk_to_bool(x):
    if isinstance(x, bool):
        return x
    if x is None or pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def _pk_to_float(x, default=np.nan):
    val = pd.to_numeric(x, errors="coerce")
    if pd.isna(val):
        return default
    return float(val)


def _pk_clean_str(x):
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def _pk_best_rna_label(row):
    gene_label = _pk_norm(row.get("RNA_GeneLevel_Conclusion", ""))
    var_label = _pk_norm(row.get("RNA_VarLevel_Conclusion", ""))
    gene_score = PK_RNA_SCORE.get(gene_label, 0.0)
    var_score = PK_RNA_SCORE.get(var_label, 0.0)
    if var_score > gene_score:
        return var_label
    if gene_score > 0:
        return gene_label
    if var_score > 0:
        return var_label
    return ""


def _pk_metric_from_row(row):
    rank = _pk_to_float(row.get("Diagnostic_Engine_Rank", np.nan), default=np.inf)
    if not np.isfinite(rank):
        rank = np.inf

    pheno_label = _pk_norm(row.get("HPO_Conclusion", ""))
    pheno_score = float(PK_PHENO_SCORE.get(pheno_label, 0))

    clinvar_label = _pk_norm(row.get("Database_Conclusion", ""))
    clinvar_score = float(PK_CLINVAR_SCORE.get(clinvar_label, 0))

    rna_label = _pk_best_rna_label(row)
    rna_score = float(PK_RNA_SCORE.get(rna_label, 0.0))

    frameshift = int(_pk_to_bool(row.get("frame_shift", False)))

    inheritance_match = _pk_to_float(row.get("inheritance_match", 1), default=1.0)
    if pd.isna(inheritance_match):
        inheritance_match = 1.0

    insilico_label = _pk_norm(row.get("Insilico_Conclusion", ""))
    insilico_rank = _pk_to_float(row.get("rank::InSilico", np.nan), default=np.inf)
    if not np.isfinite(insilico_rank):
        insilico_rank = np.inf

    # Same gate as code2: in-silico only breaks ties when the in-silico
    # expert ranked this candidate within the top 100.
    if insilico_rank <= 100:
        insilico_score = float(PK_INSILICO_SCORE.get(insilico_label, 0.0))
    else:
        insilico_score = 0.0

    return {
        "diag_rank": float(rank),
        "pheno_label": pheno_label,
        "pheno_score": pheno_score,
        "clinvar_label": clinvar_label,
        "clinvar_score": clinvar_score,
        "rna_label": rna_label,
        "rna_score": rna_score,
        "frameshift": frameshift,
        "inheritance_match": float(inheritance_match),
        "insilico_label": insilico_label,
        "insilico_rank": float(insilico_rank),
        "insilico_score": insilico_score,
    }


def _pk_compound_mean_metrics(member_metrics):
    """Compound-vs-compound uses the mean of member-variant metrics."""
    if len(member_metrics) == 1:
        return member_metrics[0]

    diag_vals = [m["diag_rank"] for m in member_metrics if np.isfinite(m["diag_rank"])]
    diag_rank_mean = float(np.mean(diag_vals)) if diag_vals else np.inf

    return {
        "diag_rank": diag_rank_mean,
        "pheno_label": "mean(" + ",".join(sorted(set(str(m["pheno_label"]) for m in member_metrics))) + ")",
        "pheno_score": float(np.mean([m["pheno_score"] for m in member_metrics])),
        "clinvar_label": "mean(" + ",".join(sorted(set(str(m["clinvar_label"]) for m in member_metrics))) + ")",
        "clinvar_score": float(np.mean([m["clinvar_score"] for m in member_metrics])),
        "rna_label": "mean(" + ",".join(sorted(set(str(m["rna_label"]) for m in member_metrics))) + ")",
        "rna_score": float(np.mean([m["rna_score"] for m in member_metrics])),
        "frameshift": float(np.mean([m["frameshift"] for m in member_metrics])),
        "inheritance_match": float(np.mean([m["inheritance_match"] for m in member_metrics])),
        "insilico_label": "mean(" + ",".join(sorted(set(str(m["insilico_label"]) for m in member_metrics))) + ")",
        "insilico_rank": float(np.min([m["insilico_rank"] for m in member_metrics if pd.notna(m["insilico_rank"])])) if any(pd.notna(m["insilico_rank"]) for m in member_metrics) else np.inf,
        "insilico_score": float(np.mean([m["insilico_score"] for m in member_metrics])),
    }


def _pk_pick_stronger_member(member_ids, variant_metrics):
    """Compound-vs-single uses the stronger member of the compound entity."""
    best_vid = None
    best_key = None
    for vid in member_ids:
        m = variant_metrics[vid]
        key = (
            float(m["diag_rank"]),
            -float(m["pheno_score"]),
            -float(m["clinvar_score"]),
            -float(m["rna_score"]),
            -float(m["inheritance_match"]),
            -float(m["frameshift"]),
            str(vid),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_vid = vid
    return best_vid, variant_metrics[best_vid]


def _pk_criterion_vote(a_val, b_val, lower_is_better=False):
    if pd.isna(a_val) and pd.isna(b_val):
        return "tie"
    if pd.isna(a_val):
        return "B"
    if pd.isna(b_val):
        return "A"

    if lower_is_better:
        if a_val < b_val:
            return "A"
        if b_val < a_val:
            return "B"
        return "tie"

    if a_val > b_val:
        return "A"
    if b_val > a_val:
        return "B"
    return "tie"


def _pk_compare(metrics_a, metrics_b):
    """
    Code2-consistent Copeland majority voting:
    primary votes are diagnostic rank, phenotype, database, RNA,
    inheritance match and frameshift. In-silico is used only as a tie-breaker;
    if still tied, diagnostic rank breaks the tie.
    """
    rank_vote = _pk_criterion_vote(float(metrics_a["diag_rank"]), float(metrics_b["diag_rank"]), lower_is_better=True)
    pheno_vote = _pk_criterion_vote(float(metrics_a.get("pheno_score", 0.0)), float(metrics_b.get("pheno_score", 0.0)))
    clinvar_vote = _pk_criterion_vote(float(metrics_a.get("clinvar_score", 0.0)), float(metrics_b.get("clinvar_score", 0.0)))
    rna_vote = _pk_criterion_vote(float(metrics_a.get("rna_score", 0.0)), float(metrics_b.get("rna_score", 0.0)))
    inheritance_vote = _pk_criterion_vote(float(metrics_a.get("inheritance_match", 1.0)), float(metrics_b.get("inheritance_match", 1.0)))
    frameshift_vote = _pk_criterion_vote(float(metrics_a.get("frameshift", 0)), float(metrics_b.get("frameshift", 0)))

    primary_votes = [rank_vote, pheno_vote, clinvar_vote, rna_vote, inheritance_vote, frameshift_vote]
    votes_a = sum(v == "A" for v in primary_votes)
    votes_b = sum(v == "B" for v in primary_votes)
    votes_tie = sum(v == "tie" for v in primary_votes)

    insilico_vote = "not_used"
    tie_reason = ""

    if votes_a > votes_b:
        winner = "A"
    elif votes_b > votes_a:
        winner = "B"
    else:
        insilico_vote = _pk_criterion_vote(
            float(metrics_a.get("insilico_score", 0.0)),
            float(metrics_b.get("insilico_score", 0.0)),
        )
        if insilico_vote == "A":
            winner = "A"
            tie_reason = "tie_break_by_insilico"
        elif insilico_vote == "B":
            winner = "B"
            tie_reason = "tie_break_by_insilico"
        else:
            rank_a = float(metrics_a["diag_rank"])
            rank_b = float(metrics_b["diag_rank"])
            if rank_a < rank_b:
                winner = "A"
                tie_reason = "tie_break_by_rank"
            elif rank_b < rank_a:
                winner = "B"
                tie_reason = "tie_break_by_rank"
            else:
                winner = "tie"
                tie_reason = "full_tie"

    return {
        "votes_A": votes_a,
        "votes_B": votes_b,
        "votes_tie": votes_tie,
        "winner": winner,
        "tie_breaker": tie_reason,
        "rank_point": rank_vote,
        "pheno_level_point": pheno_vote,
        "clinvar_point": clinvar_vote,
        "rna_point": rna_vote,
        "inheritance_point": inheritance_vote,
        "frameshift_point": frameshift_vote,
        "insilico_point": insilico_vote,
    }


def _pk_build_compound_group_ids(vdf, sampleid=""):
    """Build pair-specific compound-het entity IDs from the partner column."""
    vdf = vdf.copy()
    vdf["compound_het_group_id"] = ""
    if "partner" not in vdf.columns or vdf.empty:
        return vdf

    valid_keys = set(zip(vdf["geneSymbol"].astype(str), vdf["varId"].astype(str)))
    for idx, row in vdf.iterrows():
        gene = str(row.get("geneSymbol", ""))
        vid = str(row.get("varId", ""))
        partner = _pk_clean_str(row.get("partner", ""))
        if partner == "":
            continue
        if (gene, partner) not in valid_keys:
            continue
        pair = sorted([vid, partner])
        gid = f"{sampleid}|{gene.upper()}|CH|{pair[0]}|{pair[1]}"
        vdf.at[idx, "compound_het_group_id"] = gid
    return vdf


def PairWiseComp(data, tier=(1, 2), sampleid=""):
    """
    Code2-consistent entity-level Copeland pairwise majority voting.

    Compound handling follows code2:
    - single vs single: compare the single variants directly
    - compound vs single: represent the compound by its stronger member
    - compound vs compound: compare mean metrics across constituent variants

    Returns
    -------
    res_df : pd.DataFrame
        Per-variant pairwise scores and entity metadata for variants in `tier`.
    pk_log_df : pd.DataFrame
        Pairwise-comparison log for auditing/debugging.
    """
    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = data.copy()
    if "Tier" not in df.columns:
        df["Tier"] = np.nan
    df["TierNum"] = pd.to_numeric(df["Tier"], errors="coerce")

    tier_set = {int(x) for x in tier}
    vdf = df[df["TierNum"].isin([float(x) for x in tier_set])].copy()
    if vdf.empty:
        return pd.DataFrame(), pd.DataFrame()

    defaults = {
        "varId": "",
        "geneSymbol": "",
        "Diagnostic_Engine_Rank": np.inf,
        "HPO_Conclusion": "",
        "Database_Conclusion": "",
        "RNA_GeneLevel_Conclusion": "",
        "RNA_VarLevel_Conclusion": "",
        "frame_shift": False,
        "Insilico_Conclusion": "",
        "rank::InSilico": np.inf,
        "inheritance_match": 1,
        "partner": "",
        "HGVSc": "",
        "HGVSp": "",
    }
    for col, default in defaults.items():
        if col not in vdf.columns:
            vdf[col] = default

    vdf["varId"] = vdf["varId"].astype(str)
    vdf["geneSymbol"] = vdf["geneSymbol"].astype(str)
    vdf = vdf.drop_duplicates(subset=["geneSymbol", "varId"], keep="first").reset_index(drop=True)
    vdf = _pk_build_compound_group_ids(vdf, sampleid=sampleid)

    variant_metrics = {}
    variant_rows = {}
    variant_member_sort_keys = {}
    for _, row in vdf.iterrows():
        vid = str(row["varId"])
        metric = _pk_metric_from_row(row)
        variant_metrics[vid] = metric
        variant_rows[vid] = row
        variant_member_sort_keys[vid] = (
            float(metric["diag_rank"]),
            -float(metric["pheno_score"]),
            -float(metric["clinvar_score"]),
            -float(metric["rna_score"]),
            -float(metric["inheritance_match"]),
            -float(metric["frameshift"]),
            vid,
        )

    entities = []
    used_variants = set()

    grouped = vdf[vdf["compound_het_group_id"].astype(str).str.strip().ne("")]
    for gid, gmem in grouped.groupby("compound_het_group_id", sort=False):
        mem_ids = gmem["varId"].astype(str).tolist()
        mem_ids = [vid for vid in mem_ids if vid in variant_metrics]
        if len(set(mem_ids)) < 2:
            continue
        mem_ids = list(dict.fromkeys(mem_ids))
        used_variants.update(mem_ids)
        entities.append({
            "entity_id": f"COMPOUND::{gid}",
            "entity_type": "compound",
            "compound_het_group_id": gid,
            "member_variant_ids": mem_ids,
            "members_df": gmem.copy(),
        })

    singles = vdf[~vdf["varId"].astype(str).isin(used_variants)].copy()
    for _, row in singles.iterrows():
        vid = str(row["varId"])
        entities.append({
            "entity_id": f"SINGLE::{sampleid}::{vid}",
            "entity_type": "single",
            "compound_het_group_id": "",
            "member_variant_ids": [vid],
            "members_df": pd.DataFrame([row]),
        })

    if len(entities) == 0:
        return pd.DataFrame(), pd.DataFrame()

    entity_by_id = {e["entity_id"]: e for e in entities}
    entity_metrics = {}
    entity_stronger_metrics = {}
    entity_stronger_member_id = {}
    entity_rank_for_sort = {}
    entity_member_order = {}
    entity_meta = {}

    for entity in entities:
        eid = entity["entity_id"]
        member_ids = [str(x) for x in entity["member_variant_ids"]]
        member_metrics = [variant_metrics[vid] for vid in member_ids]

        if entity["entity_type"] == "single":
            mean_metric = member_metrics[0]
            stronger_vid = member_ids[0]
            stronger_metric = member_metrics[0]
        else:
            mean_metric = _pk_compound_mean_metrics(member_metrics)
            stronger_vid, stronger_metric = _pk_pick_stronger_member(member_ids, variant_metrics)

        entity_metrics[eid] = mean_metric
        entity_stronger_metrics[eid] = stronger_metric
        entity_stronger_member_id[eid] = stronger_vid
        entity_rank_for_sort[eid] = float(mean_metric["diag_rank"])
        entity_member_order[eid] = sorted(member_ids, key=lambda vid: variant_member_sort_keys[vid])

        members_df = entity["members_df"]
        entity_meta[eid] = {
            "entity_id": eid,
            "entity_type": entity["entity_type"],
            "compound_het_group_id": entity["compound_het_group_id"],
            "member_variant_ids": "|".join(member_ids),
            "geneSymbol_set": "|".join(sorted(set(members_df["geneSymbol"].astype(str).tolist()))),
            "HGVSc_set": "|".join(sorted(set(members_df["HGVSc"].astype(str).tolist()))) if "HGVSc" in members_df.columns else "",
            "HGVSp_set": "|".join(sorted(set(members_df["HGVSp"].astype(str).tolist()))) if "HGVSp" in members_df.columns else "",
        }

    ent_ids = [e["entity_id"] for e in entities]
    win = {eid: 0 for eid in ent_ids}
    loss = {eid: 0 for eid in ent_ids}
    tie_count = {eid: 0 for eid in ent_ids}
    pk_logs = []

    for i in range(len(ent_ids)):
        eid_a = ent_ids[i]
        ent_a = entity_by_id[eid_a]
        meta_a = entity_meta[eid_a]

        for j in range(i + 1, len(ent_ids)):
            eid_b = ent_ids[j]
            ent_b = entity_by_id[eid_b]
            meta_b = entity_meta[eid_b]

            if ent_a["entity_type"] == "single":
                met_a = entity_metrics[eid_a]
                used_mode_a = "single"
                used_member_a = entity_stronger_member_id[eid_a]
            else:
                if ent_b["entity_type"] == "single":
                    met_a = entity_stronger_metrics[eid_a]
                    used_mode_a = "compound_stronger_member"
                    used_member_a = entity_stronger_member_id[eid_a]
                else:
                    met_a = entity_metrics[eid_a]
                    used_mode_a = "compound_mean"
                    used_member_a = "|".join(ent_a["member_variant_ids"])

            if ent_b["entity_type"] == "single":
                met_b = entity_metrics[eid_b]
                used_mode_b = "single"
                used_member_b = entity_stronger_member_id[eid_b]
            else:
                if ent_a["entity_type"] == "single":
                    met_b = entity_stronger_metrics[eid_b]
                    used_mode_b = "compound_stronger_member"
                    used_member_b = entity_stronger_member_id[eid_b]
                else:
                    met_b = entity_metrics[eid_b]
                    used_mode_b = "compound_mean"
                    used_member_b = "|".join(ent_b["member_variant_ids"])

            res = _pk_compare(met_a, met_b)
            if res["winner"] == "A":
                win[eid_a] += 1
                loss[eid_b] += 1
            elif res["winner"] == "B":
                win[eid_b] += 1
                loss[eid_a] += 1
            else:
                tie_count[eid_a] += 1
                tie_count[eid_b] += 1

            pk_logs.append({
                "sample_id": sampleid,
                "PK_TierGroup": "+".join(str(x) for x in sorted(tier_set)),
                "A_entity_id": meta_a["entity_id"],
                "A_entity_type": meta_a["entity_type"],
                "A_compound_het_group_id": meta_a["compound_het_group_id"],
                "A_member_variant_ids": meta_a["member_variant_ids"],
                "A_geneSymbol_set": meta_a["geneSymbol_set"],
                "A_HGVSc_set": meta_a["HGVSc_set"],
                "A_HGVSp_set": meta_a["HGVSp_set"],
                "A_used_mode": used_mode_a,
                "A_used_member_variant_id": used_member_a,
                "A_diag_rank": met_a["diag_rank"],
                "A_pheno_label": met_a["pheno_label"],
                "A_pheno_score": met_a["pheno_score"],
                "A_clinvar_label": met_a.get("clinvar_label", ""),
                "A_clinvar_score": met_a.get("clinvar_score", 0.0),
                "A_rna_label": met_a.get("rna_label", ""),
                "A_rna_score": met_a.get("rna_score", 0.0),
                "A_inheritance_match": met_a.get("inheritance_match", 1.0),
                "A_frameshift": met_a.get("frameshift", 0),
                "A_insilico_label": met_a.get("insilico_label", ""),
                "A_insilico_score": met_a.get("insilico_score", 0.0),
                "B_entity_id": meta_b["entity_id"],
                "B_entity_type": meta_b["entity_type"],
                "B_compound_het_group_id": meta_b["compound_het_group_id"],
                "B_member_variant_ids": meta_b["member_variant_ids"],
                "B_geneSymbol_set": meta_b["geneSymbol_set"],
                "B_HGVSc_set": meta_b["HGVSc_set"],
                "B_HGVSp_set": meta_b["HGVSp_set"],
                "B_used_mode": used_mode_b,
                "B_used_member_variant_id": used_member_b,
                "B_diag_rank": met_b["diag_rank"],
                "B_pheno_label": met_b["pheno_label"],
                "B_pheno_score": met_b["pheno_score"],
                "B_clinvar_label": met_b.get("clinvar_label", ""),
                "B_clinvar_score": met_b.get("clinvar_score", 0.0),
                "B_rna_label": met_b.get("rna_label", ""),
                "B_rna_score": met_b.get("rna_score", 0.0),
                "B_inheritance_match": met_b.get("inheritance_match", 1.0),
                "B_frameshift": met_b.get("frameshift", 0),
                "B_insilico_label": met_b.get("insilico_label", ""),
                "B_insilico_score": met_b.get("insilico_score", 0.0),
                "votes_A": res["votes_A"],
                "votes_B": res["votes_B"],
                "votes_tie": res["votes_tie"],
                "rank_point": res["rank_point"],
                "pheno_level_point": res["pheno_level_point"],
                "clinvar_point": res["clinvar_point"],
                "rna_point": res["rna_point"],
                "inheritance_point": res["inheritance_point"],
                "frameshift_point": res["frameshift_point"],
                "insilico_point": res["insilico_point"],
                "winner": res["winner"],
                "tie_breaker": res["tie_breaker"],
            })

    copeland_score = {eid: win[eid] - loss[eid] for eid in ent_ids}
    ordered_entities = sorted(
        ent_ids,
        key=lambda eid: (
            -copeland_score[eid],
            -win[eid],
            loss[eid],
            entity_rank_for_sort.get(eid, np.inf),
            eid,
        ),
    )

    rows = []
    for entity_rank, eid in enumerate(ordered_entities, start=1):
        entity = entity_by_id[eid]
        member_ids = entity_member_order[eid]
        for within_rank, vid in enumerate(member_ids, start=1):
            source_row = variant_rows[vid]
            rows.append({
                "geneSymbol": source_row.get("geneSymbol", ""),
                "varId": vid,
                "PairWiseScore": copeland_score[eid],
                "PairWiseWins": win[eid],
                "PairWiseLosses": loss[eid],
                "PairWiseTies": tie_count[eid],
                "PairWiseEntityRank": entity_rank,
                "PairWiseWithinEntityRank": within_rank,
                "pairwise_entity_id": eid,
                "pairwise_entity_type": entity["entity_type"],
                "compound_het_group_id": entity["compound_het_group_id"],
            })

    res_df = pd.DataFrame(rows)
    pk_log_df = pd.DataFrame(pk_logs)
    return res_df, pk_log_df

def Review(work_path, reference_genome, output_path):
    input_root_nomcand = work_path + "/Diagnostic_results/Candidates/"
    input_root_phenotype = work_path + "/Agents/Phenotype/AgentEvaluation/"
    input_root_insilico = work_path + "/Agents/InSilico/AgentEvaluation/"
    input_root_rna = work_path + "/Agents/RNA/"

    #Tandem Repeats Param:
    print(f"Loading reference genome - Tandem Repeats")
    if not Path(reference_genome).exists():
        raise ValueError(f"{reference_genome} NOT Found. Skip")
    assert Path(reference_genome + ".fai").exists(), f"Missing fasta index (.fai). Run: samtools faidx {reference_genome}"
    fa = pysam.FastaFile(reference_genome)

    input_root_database = work_path + '/Agents/Database/AgentEvaluation/json/'
    database_res = load_DatabaseLLM_res(input_root_database)
    sample_ids = [p.name.split("_nomcand.feather")[0] for p in Path(input_root_nomcand).iterdir()]
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    pairwise_pk_logs = []

    print(f"Reviewing Cases - Pairwise Comparison")
    pbar = tqdm(sample_ids, desc="Reviewing Cases", total=len(sample_ids))
    for sampleid in pbar:
        pbar.set_postfix(sample=sampleid)
        sub_nomcand_path = input_root_nomcand + sampleid + '_nomcand.feather'
        #phenotypes:
        sub_hpo_path = input_root_phenotype + 'HPO_Agent/' + sampleid
        sub_omim_path = input_root_phenotype + 'OMIM_Agent/' + sampleid
        sub_lit_path = input_root_phenotype + 'Literature_Agent/' + sampleid
        sub_insilico_path = input_root_insilico + sampleid
        sub_rna_genelevel_path = input_root_rna + 'GeneLevelEval/' + sampleid
        sub_rna_varlevel_path = input_root_rna + 'VariantLevelEval/' + sampleid

        #Load data:
        data = pd.read_feather(sub_nomcand_path).reset_index(drop = True)
        HPO_res = load_LLM_res(sub_hpo_path, model_type = 'HPO', candidate_type = 'geneSymbol')
        OMIM_res = load_LLM_res(sub_omim_path, model_type = 'OMIM', candidate_type = 'geneSymbol')
        Lit_res = load_LLM_res(sub_lit_path, model_type = 'Literature', candidate_type = 'geneSymbol')
        Insilico_res = load_LLM_res(sub_insilico_path, model_type = 'Insilico', candidate_type = 'varId')
        RNA_gene_res = load_RNALLM_res(sub_rna_genelevel_path, model_type = 'RNA_GeneLevel', candidate_type = 'geneSymbol')
        RNA_var_res = load_RNALLM_res(sub_rna_varlevel_path, model_type = 'RNA_VarLevel', candidate_type = 'geneSymbol_VarId')
        RNA_var_res[['geneSymbol', 'varId']] = RNA_var_res['geneSymbol_VarId'].str.split('_VarId_', n=1, expand=True)
        AQ_path = work_path + '/Agents/RNA/AlleleQuantification/' + sampleid + '.feather'
        if Path(AQ_path).exists():
            AlleleQuant = pd.read_feather(AQ_path).reset_index(drop = True)
        else:
            AlleleQuant = pd.DataFrame({'varId':['None'], 'geneSymbol': None, 'transcript_id':None, 'ref_count_max':0,
                                        'ref_count_mean':0, 'ref_count_min':0, 'alt_count_max':0, 'alt_count_mean': 0, 'alt_count_min':0})
        AlleleQuant = AlleleQuant.drop(columns = ['varId','geneSymbol','transcript_id'])
        data = pd.concat([data, AlleleQuant], axis = 1)

        #Merge res:
        data = data.merge(database_res, on = 'varId', how = 'left')
        data = data.merge(HPO_res, on  = 'geneSymbol', how = 'left')
        data = data.merge(OMIM_res, on  = 'geneSymbol', how = 'left')
        data = data.merge(Lit_res, on  = 'geneSymbol', how = 'left')
        data = data.merge(Insilico_res, on = 'varId', how = 'left')
        data = data.merge(RNA_gene_res, on = 'geneSymbol', how = 'left')
        data = data.merge(RNA_var_res, on = ['varId', 'geneSymbol'], how = 'left')

        #TandemRepeats:
        TandemRepeats = TandemRepeatDetection(data = data, fa = fa)
        data['has_repeat_motif'] = TandemRepeats['has_repeat_motif']

        #Classify variants into 2 tiers:
        #Tier 1 must be True for all 5 following criteria:
        data['misscall_rna_flag'] = True
        data.loc[(data['ref_count_mean'] > 20) & (data['alt_count_min'] == 0), 'misscall_rna_flag'] = False

        data['new_diseasegene_flag'] = True
        data.loc[(data['Literature_Conclusion'] == 'Not Fit') &
                (data['Database_Conclusion'].isin({'Neutral', 'Against'}) |
                data['Database_Conclusion'].isna()),'new_diseasegene_flag'] = False

        data['omim_flag'] = True
        data.loc[data['OMIM_Conclusion'] == 'Impossible', 'omim_flag'] = False

        data['strong_nom_flag'] = True
        data.loc[data['evidence_rules'] == 'Compound Heterozygous (Compound Het)', 'strong_nom_flag'] = False
        data.loc[data['evidence_rules'].isin({'Strong RNA Evidence','Strong RNA Evidence, Compound Heterozygous (Compound Het)'}) & 
                (data['RNA_GeneLevel_Conclusion'] != 'Strong RNA Evidence') & (data['RNA_VarLevel_Conclusion'] != 'Strong RNA Evidence'),'strong_nom_flag'] = False
        
        data['Tier'] = 2
        data.loc[data['misscall_rna_flag'] & (~data['has_repeat_motif']) & data['new_diseasegene_flag'] & data['omim_flag'] & data['strong_nom_flag'], 'Tier'] = 1

        data = data.sort_values(by=["Tier", "Diagnostic_Engine_Rank"], ascending=[True, True]).reset_index(drop = True)
        data['new_rank'] = range(1, len(data) + 1)

        #Find out if Tier1 has compound het gene:
        data['frame_shift'] = False
        data.loc[(data['cons_frameshift_variant'] == 1) & (data['HGVSp'].str.contains("Ter", na=False)), 'frame_shift'] = True

        data['find_compound_het'] = False
        data['partner'] = ''

        genehash = set()
        for i in range(len(data)):
            if data.loc[i,'geneSymbol'] in genehash:
                continue
            genehash.add(data.loc[i,'geneSymbol'])
            if ((data.loc[i,'recessive'] == 1) | (data.loc[i,'Database_Zygosity'] in {'compound_heterozygous', 'compound heterozygous'})) & (data.loc[i,'zyg'] == 1) & (data.loc[i,'Database_Zygosity'] != 'heterozygous'):
                Enough_reads = False
                ratio_direction = 0
                ref_mean = data.loc[i, 'ref_count_mean'] 
                alt_mean = data.loc[i, 'alt_count_mean']
                if ref_mean >= 50 and alt_mean >= 50:
                    Enough_reads = True
                    ratio = ref_mean / alt_mean
                    if ratio > 1.2:
                        ratio_direction = 1
                    elif ratio < 0.8:
                        ratio_direction = -1
                        
                het_candidates = data.loc[(data['geneSymbol'] == data.loc[i, 'geneSymbol']) & (data['varId'] != data.loc[i, 'varId']), 
                                        ['varId','geneSymbol','Database_Conclusion','RNA_GeneLevel_Conclusion','RNA_VarLevel_Conclusion',
                                        'frame_shift','new_rank','ref_count_mean','alt_count_mean']]
                het_candidates = het_candidates.loc[het_candidates['Database_Conclusion'].isin(set(['Supporting','Convincing'])) |
                                                    (het_candidates['RNA_GeneLevel_Conclusion'] == 'Strong RNA Evidence') |
                                                    (het_candidates['RNA_VarLevel_Conclusion'] == 'Strong RNA Evidence') |
                                                    het_candidates['frame_shift'] | (het_candidates['new_rank'] <= 50),]
                if len(het_candidates) == 0:
                    partner_varid = None
                else:
                    partner_varid = None
                    for candgene in het_candidates.itertuples(index=True):
                        if Enough_reads & (ratio_direction != 0) & (candgene.ref_count_mean >= 50) & (candgene.alt_count_mean >= 50):
                            candgene_ratio = candgene.ref_count_mean / candgene.alt_count_mean
                            if ratio_direction == 1 and candgene_ratio > 1.2:
                                continue
                            if ratio_direction == -1 and candgene_ratio < 0.8:
                                continue
                        partner_varid = candgene.varId
                        partner_j = candgene.Index
                        break
                if partner_varid:
                    data.loc[i, 'find_compound_het'] = True
                    data.loc[i, 'partner'] = partner_varid
                    data.loc[partner_j, 'find_compound_het'] = True
                    data.loc[partner_j, 'partner'] = data.loc[i, 'varId']
                    data.loc[partner_j, 'Tier'] = data.loc[i, 'Tier']
                
        # Reorder variants that are incompatible with the expected inheritance pattern.
        # This value is also used as one of the primary votes in the code2-style
        # pairwise majority voting.
        data['inheritance_match'] = 1

        for i in range(len(data)):
            zyg_ok = data.loc[i, "Database_Zygosity"] in {"homozygous", "compound heterozygous", "compound_heterozygous"}
            if ((data.loc[i, 'zyg'] == 1) & 
                    (data.loc[i,'Chromosome'] != 'chrX') &
                    (not data.loc[i,'find_compound_het']) &
                    (zyg_ok | (data.loc[i,'recessive'] == 1) & (data.loc[i,'dominant'] == 0) & (data.loc[i, "Database_Zygosity"] != 'heterozygous'))):
                data.loc[i, 'Tier'] = 3

        data = data.sort_values(by=["Tier", "Diagnostic_Engine_Rank"], ascending=[True, True]).reset_index(drop = True)
        PW_Res = PairWiseComp(data, tier = [1])
        data = data.merge(PW_Res, on=['geneSymbol', 'varId'], how='left')
        data = data.sort_values(by=["Tier", "PairWiseScore"], ascending=[True, False]).reset_index(drop = True)
        data.to_csv(output_path + '/' + sampleid + '.csv', index = False)
    print(f"-- Review Done --")
