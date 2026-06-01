"""
ANNOTATE_TIER worker module for RareCollab.

Provides Process-safe (i.e. picklable, module-level) versions of:
  - _is_num_ge, _assign_gene_tiers, _recount_per_gene (helpers)
  - ANNOTATE_TIER                  (per-(sample, chr) tier classification)
  - run_annotate_tier_parallel     (orchestrator using ProcessPoolExecutor)
"""

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import multiprocessing as mp

_TIER_WORKER_RESOURCES = None


def _init_tier_worker(references, ref_ver):
    """ProcessPoolExecutor initializer: load omim_inh_df once per worker."""
    global _TIER_WORKER_RESOURCES
    import pandas as pd
    inh_path = references.ref_var_tier_dir / ref_ver / "genemap2.Inh.F.txt"
    omim_inh_df = pd.read_csv(inh_path, sep="\t")
    _TIER_WORKER_RESOURCES = {"omim_inh_df": omim_inh_df}


def _assign_gene_tiers(g_df, gene):
    """Compute TierAD/TierAR/No.Var.* for one gene's variants."""
    impact_max_arr = g_df["IMPACT.max"].to_numpy()
    tier_ad = np.where(impact_max_arr < 3, 4, 5 - impact_max_arr).astype(float)

    g_df2 = g_df
    if "HOM" in g_df["GT"].values:
        extra = g_df[g_df["GT"] == "1/1"]
        g_df2 = pd.concat([g_df, extra], ignore_index=True)

    impact_max_for_ar = g_df2["IMPACT.max"].to_numpy()
    n_h = int((impact_max_for_ar == 4).sum())
    n_m = int((impact_max_for_ar == 3).sum())

    tier_ar = np.empty(len(g_df), dtype=float)
    original_impact = g_df["IMPACT.max"].to_numpy()

    if n_h >= 2:
        tier_ar = (5 - original_impact).astype(float)
        tier_ar[original_impact == 3] = 1.5
    elif n_h == 1 and n_m >= 1:
        tier_ar[original_impact >= 3] = 1.5
        mask_lt = original_impact < 3
        tier_ar[mask_lt] = 5 - original_impact[mask_lt]
    elif n_h == 1:
        tier_ar[original_impact == 4] = 3
        mask_le = original_impact <= 3
        tier_ar[mask_le] = 5 - original_impact[mask_le]
    elif n_m >= 2:
        tier_ar[original_impact == 3] = 2
        mask_lt = original_impact < 3
        tier_ar[mask_lt] = 5 - original_impact[mask_lt]
    elif n_m == 1:
        tier_ar[original_impact == 3] = 3
        mask_lt = original_impact < 3
        tier_ar[mask_lt] = 5 - original_impact[mask_lt]
    else:
        tier_ar = (5 - original_impact).astype(float)

    no_hm = int((impact_max_for_ar >= 3).sum())
    no_h = int((impact_max_for_ar == 4).sum())
    no_m = int((impact_max_for_ar == 3).sum())
    no_l = int((impact_max_for_ar == 2).sum())

    return g_df.assign(Gene=gene, TierAD=tier_ad, TierAR=tier_ar, No_Var_HM=no_hm, No_Var_H=no_h, No_Var_M=no_m, No_Var_L=no_l)


def _recount_per_gene(g_df, gene):
    g_df2 = g_df
    if "HOM" in g_df["GT"].values:
        extra = g_df[g_df["GT"] == "1/1"]
        g_df2 = pd.concat([g_df, extra], ignore_index=True)
    impact_max = g_df2["IMPACT.max"].to_numpy()
    return g_df.assign(Gene=gene, No_Var_HM=int((impact_max >= 3).sum()), No_Var_H=int((impact_max == 4).sum()),
                       No_Var_M=int((impact_max == 3).sum()), No_Var_L=int((impact_max == 2).sum()))

def ANNOTATE_TIER(scores_csv_path, sample_id, work_dir, references, ref_ver,
                  omim_inh_df=None, overwrite=True):
    """
    Direct port of AIM's VarTierDiseaseDBFalse.R (no R/container needed).

    Classifies each (variant, gene) pair into TierAD/TierAR based on variant
    IMPACT, computes per-gene impact-variant counts, and joins with OMIM
    inheritance to flag AD.matched/AR.matched.

    Bit-for-bit reproduction of the R script's current behavior, including
    the STEP 6 typo bug that silently drops anno_noGeneID rows (Gene=="-").

    Args:
        scores_csv_path: per-chr scores CSV from ANNOTATE_BY_MODULES.
        sample_id: Sample identifier.
        work_dir: Pipeline work directory.
        references: AimReferences dataclass.
        ref_ver: 'hg19' or 'hg38'.
        omim_inh_df: pre-loaded genemap2.Inh.F.txt as a DataFrame (lazy if None).
        overwrite: Recompute even if output exists.

    Returns:
        Path to per-chr Tier output: features/<sample>/chr<N>.vep_Tier.v2.tsv
        (mirrors Nextflow's "${scores.simpleName}_Tier.v2.tsv")
    """
    scores_csv_path = Path(scores_csv_path)
    work_dir = Path(work_dir)

    if not scores_csv_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv_path}")

    scores_basename = scores_csv_path.name[:-len(".csv")]  # "chr1.vep_scores"
    chr_label = scores_basename.replace(".vep_scores", "")  # "chr1"
    tier_basename = scores_basename.replace("_scores", "")  # "chr1.vep"

    sample_dir = work_dir / "features" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f"{tier_basename}_Tier.v2.tsv"

    if overwrite:
        if output_path.exists() or output_path.is_symlink():
            output_path.unlink()
    else:
        if output_path.exists():
            return output_path

    if omim_inh_df is None:
        inh_path = references.ref_var_tier_dir / ref_ver / "genemap2.Inh.F.txt"
        omim_inh_df = pd.read_csv(inh_path, sep="\t")

    # =========================================================================
    # STEP 0: Load scores.csv, keep 13 columns, clean varId
    # =========================================================================
    anno_columns = [
        "varId", "zyg", "geneSymbol", "geneEnsId",
        "gnomadAF", "gnomadAFg",
        "omimSymptomSimScore", "hgmdSymptomSimScore",
        "IMPACT", "Consequence", "hgmdVarFound", "clinvarSignDesc",
        "spliceAImax",
    ]

    anno_orig = pd.read_csv(scores_csv_path, low_memory = False)
    anno = anno_orig[anno_columns].copy()

    # Strip "_-..." (intergenic) and "_E..." (transcript) suffixes from varId.
    # R: str_split(varId, "_-", simplify=T)[,1] -> str_split(_, "_E", simplify=T)[,1]
    anno["varId"] = anno["varId"].astype(str).str.split("_-", n=1).str[0]
    anno["varId"] = anno["varId"].str.split("_E", n=1).str[0]

    # Rename to match R's VEP-style column names
    anno = anno.rename(columns={
        "varId": "Uploaded_variation",
        "zyg": "GT",
        "geneSymbol": "SYMBOL",
        "geneEnsId": "Gene",
    })

    # =========================================================================
    # STEP 1: filter rows where Gene != "-" and dedup
    # R: anno %>% filter(Gene != "-") %>% distinct()
    # =========================================================================
    anno = anno[anno["Gene"] != "-"].drop_duplicates().reset_index(drop=True)

    # Apply spliceAI override THEN compute IMPACT.no
    anno["IMPACT"] = np.where(
        anno["spliceAImax"].apply(lambda x: _is_num_ge(x, 0.8)),
        "HIGH",
        anno["IMPACT"],
    )
    impact_to_no = {"MODIFIER": 1, "LOW": 2, "MODERATE": 3, "HIGH": 4}
    anno["IMPACT.no"] = anno["IMPACT"].map(impact_to_no).fillna(4).astype(int)

    # =========================================================================
    # STEP 1 (continued): mark Tier-4 genes (only L variants in entire gene)
    # =========================================================================
    gene_all = anno["Gene"].unique()
    gene_impact = anno[["Gene", "IMPACT"]].drop_duplicates()
    gene_impact_hm = gene_impact[gene_impact["IMPACT"].isin(["HIGH", "MODERATE"])]
    gene_tier4 = set(gene_all) - set(gene_impact_hm["Gene"])

    # R: anno %>% mutate(TierAD = ifelse(Gene %in% GeneTier4, 4, NA))
    anno["TierAD"] = np.where(anno["Gene"].isin(gene_tier4), 4.0, np.nan)
    anno["TierAR"] = np.where(anno["Gene"].isin(gene_tier4), 4.0, np.nan)

    # =========================================================================
    # STEP 2: get the most severe IMPACT for each (variant, gene)
    # Only the rows with NaN TierAD (i.e. NOT GeneTier4) are processed further.
    # =========================================================================
    idx_run_tier = anno["TierAD"].isna()
    var_run_tier = set(anno.loc[idx_run_tier, "Uploaded_variation"])
    parse_var_no_tier = (~idx_run_tier).any()
    if parse_var_no_tier:
        var_no_tier = set(anno.loc[~idx_run_tier, "Uploaded_variation"])

    anno_f1 = anno[idx_run_tier].copy()

    # Per (variant, gene): IMPACT.max = max IMPACT.no among transcripts
    anno_f1["IMPACT.max"] = anno_f1.groupby(
        ["Uploaded_variation", "Gene"]
    )["IMPACT.no"].transform("max")

    # Note: rareVar/intronicVar/rareIntronicVar block in R computes a set but
    # never uses it later. We omit it (same effect as STEP 6 bug — code is dead).

    vep_f2 = anno_f1[[
        "Uploaded_variation", "SYMBOL", "Gene", "GT",
        "IMPACT.max",
        "omimSymptomSimScore", "hgmdSymptomSimScore",
    ]].drop_duplicates().reset_index(drop=True)

    # =========================================================================
    # STEP 3 + 4: assign TierAD / TierAR per gene
    # R uses group_modify; we replicate with apply on groups.
    # =========================================================================
    if len(vep_f2) > 0:
        pieces = []
        for gene, g_df in vep_f2.groupby("Gene", sort=False):
            pieces.append(_assign_gene_tiers(g_df, gene))
        vep_tier_part1 = pd.concat(pieces, ignore_index=True)
    else:
        vep_tier_part1 = vep_f2.assign(
            TierAD=pd.Series(dtype=float),
            TierAR=pd.Series(dtype=float),
            No_Var_HM=pd.Series(dtype=int),
            No_Var_H=pd.Series(dtype=int),
            No_Var_M=pd.Series(dtype=int),
            No_Var_L=pd.Series(dtype=int),
        )

    # TierAR.adj = TierAR (R does this unconditionally)
    vep_tier_part1["TierAR.adj"] = vep_tier_part1["TierAR"]

    # Reorder columns to match R's VEP.Tier.part1
    vep_tier_part1 = vep_tier_part1[[
        "Uploaded_variation", "Gene", "GT", "IMPACT.max",
        "TierAD", "TierAR", "TierAR.adj",
        "No_Var_HM", "No_Var_H", "No_Var_M", "No_Var_L",
    ]]

    # =========================================================================
    # STEP 5: add Var.NoTier (GeneTier4 rows) back
    # =========================================================================
    if parse_var_no_tier:
        vep_tier_part2 = anno.loc[
            anno["Uploaded_variation"].isin(var_no_tier),
            ["Uploaded_variation", "Gene", "GT"],
        ].copy()
        vep_tier_part2["IMPACT.max"] = 1
        vep_tier_part2["TierAD"] = 4.0
        vep_tier_part2["TierAR"] = 4.0
        vep_tier_part2["TierAR.adj"] = 4.0
        vep_tier_part2["No_Var_HM"] = np.nan
        vep_tier_part2["No_Var_H"] = np.nan
        vep_tier_part2["No_Var_M"] = np.nan
        vep_tier_part2["No_Var_L"] = np.nan
        vep_tier = pd.concat([vep_tier_part1, vep_tier_part2], ignore_index=True)
    else:
        vep_tier = vep_tier_part1

    # =========================================================================
    # STEP 5.5: per-gene No.Var recount on the merged frame, with HOM duplication
    # R: VEP.Tier.wGene <- VEP.Tier %>% group_by(Gene) %>% group_modify(...)
    # =========================================================================
    if len(vep_tier) > 0:
        pieces = []
        for gene, g_df in vep_tier.groupby("Gene", sort=False):
            pieces.append(_recount_per_gene(g_df, gene))
        vep_tier_wgene = pd.concat(pieces, ignore_index=True)
    else:
        vep_tier_wgene = vep_tier

    # Drop GT column (R: select(-GT))
    vep_tier_wgene = vep_tier_wgene.drop(columns=["GT"])

    # =========================================================================
    # STEP 6: In AIM's R script there's a typo where VEP.Tier.final <-
    # VEP.Tier.wGene runs unconditionally, silently discarding the
    # anno_noGeneID computation. We skip computing anno_noGeneID entirely
    # (it's dead code in AIM, confirmed with the AIM team).
    # =========================================================================
    vep_tier_final = vep_tier_wgene

    # Rename IMPACT.max -> IMPACT.from.Tier (R: colnames assignment)
    vep_tier_final = vep_tier_final.rename(columns={"IMPACT.max": "IMPACT.from.Tier"})

    # Fill TierAR.adj NaN with TierAR value (R has this fallback)
    mask_adj_na = vep_tier_final["TierAR.adj"].isna()
    vep_tier_final.loc[mask_adj_na, "TierAR.adj"] = vep_tier_final.loc[mask_adj_na, "TierAR"]

    # =========================================================================
    # STEP 7: merge with OMIM inheritance, compute AD.matched / AR.matched
    # R: merge(VEP.Tier.final, genemap2.Inh.F, by.x="Gene", by.y="Gene", all.x=T)
    # Note: genemap2.Inh.F first column gets renamed to "Gene" in R.
    # =========================================================================
    inh = omim_inh_df.copy()
    inh.columns = ["Gene"] + list(inh.columns[1:])  # rename col 1 to Gene

    vep_tier_winh = vep_tier_final.merge(
        inh, on="Gene", how="left",
    )
    vep_tier_winh["dominant"] = vep_tier_winh["dominant"].fillna(0).astype(int)
    vep_tier_winh["recessive"] = vep_tier_winh["recessive"].fillna(0).astype(int)

    vep_tier_winh["AD.matched"] = (
        (vep_tier_winh["TierAD"] <= 2) & (vep_tier_winh["dominant"] == 1)
    ).astype(int)
    vep_tier_winh["AR.matched"] = (
        (vep_tier_winh["TierAR"] <= 2) & (vep_tier_winh["recessive"] == 1)
    ).astype(int)

    # Rename our snake_case No_Var_* back to R's dot-style No.Var.*
    vep_tier_winh = vep_tier_winh.rename(columns={
        "No_Var_HM": "No.Var.HM",
        "No_Var_H": "No.Var.H",
        "No_Var_M": "No.Var.M",
        "No_Var_L": "No.Var.L",
    })

    # Final column order: matches the R baseline you showed
    final_cols = [
        "Gene", "Uploaded_variation", "IMPACT.from.Tier",
        "TierAD", "TierAR", "TierAR.adj",
        "No.Var.HM", "No.Var.H", "No.Var.M", "No.Var.L",
        "dominant", "recessive", "AD.matched", "AR.matched",
    ]
    vep_tier_winh = vep_tier_winh[final_cols]

    # Cast numeric columns to int where appropriate to match R int output.
    # TierAD/TierAR/TierAR.adj can be 1.5 (float), keep as float.
    for col in ["IMPACT.from.Tier", "dominant", "recessive", "AD.matched", "AR.matched"]:
        vep_tier_winh[col] = vep_tier_winh[col].astype(int)

    vep_tier_winh.to_csv(output_path, sep="\t", index=False)
    return output_path


def _is_num_ge(value, threshold):
    """Try to interpret value as a float and compare. Used for spliceAImax,
    which is often '-' for missing."""
    try:
        return float(value) >= threshold
    except (ValueError, TypeError):
        return False

def _annotate_tier_worker_run(scores_csv, sample_id, work_dir, references,
                              ref_ver, overwrite):
    """Worker entry point: dispatch ANNOTATE_TIER with cached omim_inh_df."""
    res = _TIER_WORKER_RESOURCES
    return ANNOTATE_TIER(
        scores_csv, sample_id, work_dir, references, ref_ver,
        omim_inh_df=res["omim_inh_df"],
        overwrite=overwrite,
    )


def run_annotate_tier_parallel(samplesheet, work_dir, references, ref_ver,
                               max_workers, overwrite=True):
    """
    Run ANNOTATE_TIER in parallel across (sample, chr) tasks using
    ProcessPoolExecutor. Each worker process loads omim_inh_df ONCE via
    _init_tier_worker, then processes its assigned tasks.

    Returns: dict[row_index -> sorted list of output paths]
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    tasks = []
    tier_results = {}
    ok = fail = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_tier_worker,
        initargs=(references, ref_ver),
        mp_context=mp.get_context("spawn"),
    ) as ex:
        for row in samplesheet.itertuples(index=True):
            for scores_csv in row.scores_csv_paths:
                future = ex.submit(
                    _annotate_tier_worker_run,
                    scores_csv, row.sampleID, work_dir, references, ref_ver,
                    overwrite,
                )
                tasks.append({
                    "index": row.Index,
                    "sampleID": row.sampleID,
                    "scores_csv": scores_csv,
                    "future": future,
                })

        with tqdm(total=len(tasks), desc="ANNOTATE TIER") as pbar:
            future_to_task = {t["future"]: t for t in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    tier_path = future.result()
                    tier_results.setdefault(task["index"], []).append(str(tier_path))
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] ANNOTATE_TIER failed for sample {task['sampleID']}")
                    print(f"[ERROR] Input: {task['scores_csv']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    if fail > 0:
        raise RuntimeError(f"ANNOTATE_TIER failed for {fail} task(s).")

    # Sort paths within each sample for consistent ordering
    return {idx: sorted(paths) for idx, paths in tier_results.items()}