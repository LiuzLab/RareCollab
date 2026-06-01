"""
MERGE_SCORES_BY_CHROMOSOME worker module for RareCollab.

Per-sample pipeline that:
  1. Concatenates 24 chr-level Tier.v2.tsv files (one per chr)
  2. Concatenates 24 chr-level scores.txt.gz files (one per chr)
  3. Runs mod5 PPI network diffusion using phrank scores
  4. Runs Chaozhong's feature_engineering (fillna_tier) — ~1000-line port
  5. Runs simple repeat annotation (replaces bedtools with native Python
     interval lookup, similar to JOIN_PHRANK's optimization)

Outputs <sample>.matrix.txt: the prediction-ready feature matrix.

Uses ProcessPoolExecutor across samples (5 samples → 5 workers).
"""

import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from scipy import stats
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------------------------------------------------
# Worker-local cache
# ----------------------------------------------------------------------
_MERGE_WORKER_RESOURCES = None


def _init_merge_worker(references, ref_ver):
    """ProcessPoolExecutor initializer."""
    global _MERGE_WORKER_RESOURCES
    _MERGE_WORKER_RESOURCES = prepare_merge_resources(references, ref_ver)


def _merge_worker_run(sample_id, tier_paths, score_paths, phrank_path,
                      work_dir, references, ref_ver, overwrite):
    res = _MERGE_WORKER_RESOURCES
    return MERGE_SCORES_BY_CHROMOSOME(
        sample_id, tier_paths, score_paths, phrank_path,
        work_dir, references, ref_ver,
        feature_stats=res["feature_stats"],
        net_norm=res["net_norm"],
        cor_gene_id=res["cor_gene_id"],
        simple_repeats_by_chrom=res["simple_repeats_by_chrom"],
        overwrite=overwrite,
    )

# ----------------------------------------------------------------------
# Resource prep
# ----------------------------------------------------------------------
def prepare_merge_resources(references, ref_ver):
    """Load mod5 network + feature_stats + simple_repeats once per worker."""
    # mod5 PPI network
    mod5_path = references.ref_mod5_diffusion_dir / "net_norm_cor_GeneID.npz"
    npz = np.load(mod5_path, allow_pickle=True)
    net_norm = npz["net_norm"]
    cor_gene_id = pd.DataFrame(npz["cor_GeneID_arr"], columns=["ID"])

    # Feature stats (fallback values for fillna in single-variant case)
    feature_stats_path = references.ref_annot_dir / "feature_stats.csv"
    feature_stats = pd.read_csv(feature_stats_path, index_col=0)

    # Simple repeats: load BED, group by chrom, sort by start
    sp_bed = references.ref_merge_expand_dir / ref_ver / f"simpleRepeats.{ref_ver}.bed"
    simple_repeats_by_chrom = _load_simple_repeats(sp_bed)

    return {
        "net_norm": net_norm,
        "cor_gene_id": cor_gene_id,
        "feature_stats": feature_stats,
        "simple_repeats_by_chrom": simple_repeats_by_chrom,
    }


def _load_simple_repeats(bed_path):
    """
    Load simple-repeats BED into per-chrom sorted (start, end) arrays.
    Enables O(log n) interval lookup per query, replacing bedtools.
    """
    bed = pd.read_csv(bed_path, sep="\t", header=None,
                      names=["chrom", "start", "end"],
                      usecols=[0, 1, 2], dtype={"chrom": str, "start": int, "end": int})
    out = {}
    for chrom, group in bed.groupby("chrom", sort=False):
        sorted_g = group.sort_values("start", kind="stable")
        out[chrom] = (sorted_g["start"].to_numpy(), sorted_g["end"].to_numpy())
    return out


# ----------------------------------------------------------------------
# mod5 diffusion (direct port of mod5_diffusion.py)
# ----------------------------------------------------------------------
def _diffusion(nn, y, alpha=0.5, max_iter=100):
    """Helper: run network diffusion. Direct port of AIM's diffusion()."""
    y = y.to_numpy().astype("float32") if hasattr(y, "to_numpy") else y.astype("float32")
    F = y
    Fs = np.empty((nn.shape[0], 0), "float32")
    for i in range(0, max_iter):
        Fs = np.append(Fs, F, axis=1)
        F = nn @ (alpha * Fs[:, [i]]) + (1 - alpha) * y
    return F


def _diffuse_sample(sample_id, anno_df, phrank_path, net_norm, cor_gene_id):
    """
    Port of mod5_diffusion.diffuseSample.

    Args:
        sample_id: Sample ID (unused except for log messages in original).
        anno_df: merged scores DataFrame (with geneEnsId column).
        phrank_path: Path to <sample>.phrank.txt.
        net_norm, cor_gene_id: shared PPI resources.

    Returns:
        DataFrame indexed by varId with single column "diffuse_Phrank_STRING".
        None if phrank file is empty.
    """
    phrank_path = Path(phrank_path)
    if not phrank_path.exists() or phrank_path.stat().st_size == 0:
        return None

    phrank = pd.read_csv(phrank_path, sep="\t", header=None)
    phrank = phrank.rename({0: "Ensembl_Gene_ID", 1: "Score"}, axis="columns")
    phrank["Ensembl_Gene_ID"] = phrank["Ensembl_Gene_ID"].replace("", np.nan)
    phrank = phrank.dropna(subset=["Ensembl_Gene_ID"])
    # Keep highest score per gene, preserve original order
    phrank = (
        phrank.sort_values("Score", ascending=False)
        .drop_duplicates("Ensembl_Gene_ID")
        .sort_index()
    )
    simi = phrank.rename({0: "Ensembl_Gene_ID", "Score": "Similarity_Score"},
                         axis="columns")

    m12_df = anno_df
    m12_genes = [g for g in m12_df["geneEnsId"].tolist() if "ENSG" in g]

    # Filter: genes in module 1-2 ∩ phrank, minus those in cor_gene_id
    m12_wsimi = list(set(phrank["Ensembl_Gene_ID"]) & set(m12_genes))
    m12_wsimi_wocor = list(set(m12_wsimi) - set(cor_gene_id["ID"]))
    m12_wsimi_wocor_df = simi[simi["Ensembl_Gene_ID"].isin(m12_wsimi_wocor)]

    # Set initial heat Y: cor_gene_id joined with similarity score, NaN -> 0
    Y = cor_gene_id.merge(simi, left_on="ID", right_on="Ensembl_Gene_ID",
                          how="left")
    Y = Y[["ID", "Similarity_Score"]].fillna(0)
    Y.set_index("ID", inplace=True)

    # Diffuse
    diff_res = _diffusion(net_norm, Y, 0.5, 100)

    # Combine
    diff_w_gene_id = pd.DataFrame(diff_res, columns=["Final_Heat"])
    diff_w_gene_id["GeneID"] = cor_gene_id["ID"].values
    diff_w_gene_id = diff_w_gene_id[diff_w_gene_id["Final_Heat"] != 0]
    diff_w_gene_id = diff_w_gene_id[["GeneID", "Final_Heat"]]

    # Combine diffusion + similarity for genes in m12
    diff_in_m12 = diff_w_gene_id[diff_w_gene_id["GeneID"].isin(m12_genes)]
    wsimi_renamed = m12_wsimi_wocor_df.rename(
        columns={"Similarity_Score": "Final_Heat"}
    )
    diff_renamed = diff_in_m12.rename(columns={"GeneID": "Ensembl_Gene_ID"})
    final_heat = pd.concat([wsimi_renamed, diff_renamed], ignore_index=True,
                           sort=True)
    final_heat = final_heat.sort_values(by="Final_Heat", ascending=False)
    final_heat_indexed = final_heat.set_index("Ensembl_Gene_ID")

    # Score per variant in m12 order
    score_ordered = []
    gene_ordered = m12_df["geneEnsId"].tolist()
    for gene in gene_ordered:
        if gene in final_heat_indexed.index:
            val = final_heat_indexed.loc[gene, "Final_Heat"]
            # If multiple matches, take the first (preserves AIM behavior)
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            score_ordered.append(val)
        else:
            score_ordered.append(0)

    # Normalize to z-score percentile
    scores = stats.rankdata(score_ordered, "max") / len(score_ordered)

    m12_df = m12_df.copy()
    m12_df["diffuse_Phrank_STRING"] = scores
    m12_df = m12_df.drop_duplicates(subset=["varId"])
    m12_df = m12_df[["varId", "diffuse_Phrank_STRING"]].set_index("varId",
                                                                  drop=True)
    return m12_df


# ----------------------------------------------------------------------
# simple_repeat_anno (replaces bedtools with native Python)
# ----------------------------------------------------------------------
def _simple_repeat_anno(sample_df, simple_repeats_by_chrom):
    """
    Mark each variant 1 if its position falls within any simple-repeat
    interval on the same chrom, else 0.

    Port of simple_repeat_anno.py; replaces bedtools intersect with native
    Python per-chrom interval lookup (NumPy + binary search).
    """
    # Parse varId_dash: "<chrom>-<pos>-<ref>-<alt>"
    parts = sample_df["varId_dash"].str.split("-", n=4, expand=True)
    chroms = parts[0].to_numpy()
    positions = parts[1].astype(int).to_numpy()

    result = np.zeros(len(sample_df), dtype=int)

    for chrom in np.unique(chroms):
        if chrom not in simple_repeats_by_chrom:
            continue
        starts, ends = simple_repeats_by_chrom[chrom]

        # Variants on this chrom
        idx = np.where(chroms == chrom)[0]
        if len(idx) == 0:
            continue
        pos_on_chrom = positions[idx]

        # For each pos, find all intervals with start <= pos and end >= pos.
        # Binary-search start <= pos: position counts (np.searchsorted side='right')
        upper = np.searchsorted(starts, pos_on_chrom, side="right")

        # Per pos, check end >= pos within candidates [0:upper]
        for k, pos in enumerate(pos_on_chrom):
            ub = upper[k]
            if ub == 0:
                continue
            if (ends[:ub] >= pos).any():
                result[idx[k]] = 1

    sample_df = sample_df.copy()
    sample_df["simple_repeat"] = result
    return sample_df


# ----------------------------------------------------------------------
# Main: per-sample merge pipeline
# ----------------------------------------------------------------------
def MERGE_SCORES_BY_CHROMOSOME(sample_id, tier_paths, score_paths, phrank_path,
                                work_dir, references, ref_ver,
                                feature_stats=None, net_norm=None,
                                cor_gene_id=None,
                                simple_repeats_by_chrom=None,
                                overwrite=True):
    """
    Per-sample merge: concat 24 chr Tier + scores, run mod5 diffusion,
    feature engineering, simple repeat annotation, write <sample>.matrix.txt.

    Direct port of AIM's MERGE_SCORES_BY_CHROMOSOME + post_processing.py.
    """
    work_dir = Path(work_dir)
    sample_dir = work_dir / "merged" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f"{sample_id}.matrix.txt"

    if overwrite:
        if output_path.exists() or output_path.is_symlink():
            output_path.unlink()
    else:
        scores_long_path = sample_dir / f"{sample_id}.scores.txt.gz"
        if output_path.exists() and scores_long_path.exists():
            return {
                "matrix_txt_path": str(output_path),
                "scores_long_path": str(scores_long_path),
            }

    # Lazy-load shared resources if not provided
    if any(x is None for x in [feature_stats, net_norm, cor_gene_id,
                                simple_repeats_by_chrom]):
        res = prepare_merge_resources(references, ref_ver)
        feature_stats = res["feature_stats"]
        net_norm = res["net_norm"]
        cor_gene_id = res["cor_gene_id"]
        simple_repeats_by_chrom = res["simple_repeats_by_chrom"]

    # ---- STEP 1: concat 24 Tier.v2.tsv files ----
    tier_dfs = []
    for p in sorted(tier_paths):
        tier_dfs.append(pd.read_csv(p, sep="\t"))
    tier_combined = pd.concat(tier_dfs, ignore_index=True)

    # ---- STEP 2: concat 24 scores.txt.gz files ----
    score_dfs = []
    for p in sorted(score_paths):
        # Keep first file's first column (the rownumber index) as data,
        # subsequent files concat without re-reading header.
        score_dfs.append(pd.read_csv(p, sep="\t", low_memory=False))
    merged = pd.concat(score_dfs, ignore_index=True)
    # AIM's gz outputs have a leading index column from to_csv default;
    # drop it if present (it's just row numbers and is meaningless after concat).
    if merged.columns[0].startswith("Unnamed"):
        merged = merged.drop(columns=merged.columns[0])
    
    scores_long_path = sample_dir / f"{sample_id}.scores.txt.gz"
    merged.to_csv(scores_long_path, sep="\t", index=False, compression="gzip")

    # ---- STEP 3: mod5 diffusion (phrank → diffuse_Phrank_STRING) ----
    phrank_empty = (Path(phrank_path).stat().st_size == 0)
    if not phrank_empty:
        mod5 = _diffuse_sample(sample_id, merged, phrank_path, net_norm,
                                cor_gene_id)
        if mod5 is None:
            phrank_empty = True

    # ---- STEP 4: fillna_tier feature engineering ----
    iff = _feature_engineering(merged, tier_combined, feature_stats)

    # Insert diffuse_Phrank_STRING column at position 0
    if phrank_empty:
        iff.insert(loc=0, column="diffuse_Phrank_STRING", value=0)
    else:
        iff.insert(loc=0, column="diffuse_Phrank_STRING",
                   value=mod5["diffuse_Phrank_STRING"])

    # ---- STEP 5: remove chr 26 (AIM: locations on unmapped contigs) ----
    iff = iff.loc[~iff.index.str.startswith("26")]

    # ---- STEP 6: simple repeat annotation ----
    iff = _simple_repeat_anno(iff, simple_repeats_by_chrom)
    iff = iff.drop(columns=["varId_dash"])

    # ---- Write ----
    iff.to_csv(output_path, sep="\t")
    return {
    "matrix_txt_path": str(output_path),
    "scores_long_path": str(scores_long_path),
}


# ----------------------------------------------------------------------
# fillna_tier feature_engineering — port of fillna_tier.py
# ----------------------------------------------------------------------
def _parse_numeric(val_str, select="min"):
    """Port of fillna_tier._parse_numeric."""
    select_method = {"min": min, "max": max}
    vals = val_str.split(",")
    if "-" in vals or "." in vals:
        return np.nan
    vals = [float(i) for i in vals]
    return select_method[select](vals)


def _feature_engineering(score_file, tier_file, feature_stats):
    """
    Direct port of fillna_tier.feature_engineering.

    Heavy if/else conversion of '-'/string features to numeric, fillna,
    min/max-select multi-value strings, etc. Logic preserved bit-for-bit
    from AIM source.
    """
    variable_name = [
        "varId", "varId_dash", "hgmdSymptomScore", "omimSymMatchFlag",
        "hgmdSymMatchFlag", "clinVarSymMatchFlag", "omimGeneFound",
        "omimVarFound", "hgmdGeneFound", "hgmdVarFound", "clinVarVarFound",
        "clinVarGeneFound", "clinvarTotalNumVars", "clinvarNumP",
        "clinvarNumLP", "clinvarNumLB", "clinvarNumB", "dgvVarFound",
        "decipherVarFound", "curationScoreHGMD", "curationScoreOMIM",
        "curationScoreClinVar", "conservationScoreDGV", "omimSymptomSimScore",
        "hgmdSymptomSimScore", "clin_code", "GERPpp_RS", "gnomadAF",
        "gnomadAFg", "LRT_score", "LRT_Omega", "phyloP100way_vertebrate",
        "gnomadGeneZscore", "gnomadGenePLI", "gnomadGeneOELof",
        "gnomadGeneOELofUpper", "IMPACT", "CADD_phred", "CADD_PHRED",
        "DANN_score", "REVEL_score", "fathmm_MKL_coding_score",
        "conservationScoreGnomad", "conservationScoreOELof",
        "Polyphen2_HDIV_score", "Polyphen2_HVAR_score", "SIFT_score", "zyg",
        "FATHMM_score", "M_CAP_score", "MutationAssessor_score",
        "ESP6500_AA_AF", "ESP6500_EA_AF", "hom", "hgmd_rs", "spliceAImax",
        "clin_PLP_perc", "Consequence", "nc_ClinVar_Exp", "c_ClinVar_Exp",
        "c_HGMD_Exp", "nc_HGMD_Exp", "nc_isPLP", "nc_isBLB", "c_isPLP",
        "c_isBLB", "nc_CLNREVSTAT", "c_CLNREVSTAT", "nc_RANKSCORE",
        "c_RANKSCORE", "CLASS", "phrank",
    ]

    patient = score_file.copy()
    patient = patient[variable_name]
    patient["varId"] = patient["varId"].str.split("_-").str[0]
    patient = patient.fillna("-")
    indel_index = [
        len(i.split("_")[-1]) != 1 or len(i.split("_")[-2]) != 1
        for i in patient["varId"].to_list()
    ]

    # --- numeric/categorical conversions (mirroring fillna_tier.py exactly) ---
    patient.loc[patient["phrank"] == "-", "phrank"] = 0
    patient["phrank"] = patient["phrank"].astype("float64")

    patient.loc[patient["hgmdSymptomScore"] == "-", "hgmdSymptomScore"] = 0
    patient["hgmdSymptomScore"] = patient["hgmdSymptomScore"].astype("float64")

    patient.loc[patient["omimSymMatchFlag"] == "-", "omimSymMatchFlag"] = 0
    patient["omimSymMatchFlag"] = patient["omimSymMatchFlag"].astype("float64")

    patient.loc[patient["hgmdSymMatchFlag"] == "-", "hgmdSymMatchFlag"] = 0
    patient["hgmdSymMatchFlag"] = patient["hgmdSymMatchFlag"].astype("float64")

    patient.loc[patient["clinVarSymMatchFlag"] == "-", "clinVarSymMatchFlag"] = 0
    patient["clinVarSymMatchFlag"] = patient["clinVarSymMatchFlag"].astype("float64")

    patient["clinvarNumLP"] = (patient["clinvarNumLP"] + patient["clinvarNumP"]) / patient["clinvarTotalNumVars"]
    patient["clinvarNumLP"] = patient["clinvarNumLP"].fillna(0.0)
    patient["clinvarNumP"] = patient["clinvarNumP"] / patient["clinvarTotalNumVars"]
    patient["clinvarNumP"] = patient["clinvarNumP"].fillna(0.0)
    patient["clinvarNumLB"] = (patient["clinvarNumLB"] + patient["clinvarNumB"]) / patient["clinvarTotalNumVars"]
    patient["clinvarNumLB"] = patient["clinvarNumLB"].fillna(1.0)
    patient["clinvarNumB"] = patient["clinvarNumB"] / patient["clinvarTotalNumVars"]
    patient["clinvarNumB"] = patient["clinvarNumB"].fillna(1.0)
    patient = patient.drop(columns=["clinvarTotalNumVars"])
    variable_name.remove("clinvarTotalNumVars")

    for col in ["curationScoreHGMD", "curationScoreOMIM", "curationScoreClinVar",
                "conservationScoreDGV"]:
        patient.loc[patient[col] == "Low", col] = 1
        patient.loc[patient[col] == "Medium", col] = 2
        patient.loc[patient[col] == "High", col] = 3
        patient[col] = patient[col].astype("float64")

    patient.loc[patient["omimSymptomSimScore"] == "-", "omimSymptomSimScore"] = 0.0
    patient["omimSymptomSimScore"] = patient["omimSymptomSimScore"].astype("float64")
    patient.loc[patient["hgmdSymptomSimScore"] == "-", "hgmdSymptomSimScore"] = 0.0
    patient["hgmdSymptomSimScore"] = patient["hgmdSymptomSimScore"].astype("float64")

    # isB/LB, isP/LP from clin_code + clin_PLP_perc
    patient["isB/LB"] = 0.0
    patient["isP/LP"] = 0.0
    patient.loc[patient["clin_code"].str.contains("Benign"), "isB/LB"] = 1
    patient.loc[patient["clin_code"].str.contains("Likely_benign"), "isB/LB"] = 1
    patient.loc[patient["clin_code"].str.contains("Conflicting_interpretations_of_pathogenicity"), "isB/LB"] = 0
    patient.loc[patient["clin_code"].str.contains("Likely_pathogenic"), "isP/LP"] = 1
    patient.loc[patient["clin_code"].str.contains("Pathogenic"), "isP/LP"] = 1
    patient.loc[patient["clin_code"].str.contains("Conflicting_interpretations_of_pathogenicity"), "isP/LP"] = 0
    patient.loc[patient["clin_PLP_perc"] != "-", "isP/LP"] = patient.loc[patient["clin_PLP_perc"] != "-", "clin_PLP_perc"].astype("float64")
    variable_name.append("isB/LB")
    variable_name.append("isP/LP")
    variable_name.remove("clin_PLP_perc")
    variable_name.remove("clin_code")

    # Helper for the "fill from feature_stats if all NaN else from describe" pattern
    def _fill_numeric(col, fallback_kind, indel_fallback_kind=None):
        """fallback_kind: 'min', 'max', 'mean', '50%'. indel_fallback_kind defaults to same."""
        if indel_fallback_kind is None:
            indel_fallback_kind = fallback_kind
        patient.loc[patient[col] == "-", col] = np.nan
        if np.all(pd.isna(patient[col])):
            patient.loc[(pd.isna(patient[col])) & np.array(indel_index), col] = feature_stats.loc[col, indel_fallback_kind]
            patient.loc[pd.isna(patient[col]), col] = feature_stats.loc[col, fallback_kind]
            patient[col] = patient[col].astype("float64")
        else:
            patient[col] = patient[col].astype("float64")
            patient.loc[(pd.isna(patient[col])) & np.array(indel_index), col] = patient[col].describe()[indel_fallback_kind]
            patient.loc[pd.isna(patient[col]), col] = patient[col].describe()[fallback_kind]

    _fill_numeric("GERPpp_RS", "min", "mean")

    # gnomadAF / gnomadAFg with cross-fill
    gnomad_af = np.array([_parse_numeric(str(i), "min") for i in patient["gnomadAF"]]).astype(float)
    patient["gnomadAF"] = gnomad_af
    gnomad_afg = np.array([_parse_numeric(str(i), "min") for i in patient["gnomadAFg"]]).astype(float)
    patient["gnomadAFg"] = gnomad_afg
    patient.loc[patient["gnomadAFg"].isna(), "gnomadAFg"] = patient.loc[patient["gnomadAFg"].isna(), "gnomadAF"]
    patient.loc[patient["gnomadAF"].isna(), "gnomadAF"] = patient.loc[patient["gnomadAF"].isna(), "gnomadAFg"]
    patient["gnomadAF"] = patient["gnomadAF"].fillna(0.0)
    patient["gnomadAFg"] = patient["gnomadAFg"].fillna(0.0)

    _fill_numeric("LRT_score", "min", "mean")
    _fill_numeric("LRT_Omega", "min", "mean")
    _fill_numeric("phyloP100way_vertebrate", "min", "mean")
    _fill_numeric("gnomadGeneZscore", "min")
    _fill_numeric("gnomadGenePLI", "min")
    _fill_numeric("gnomadGeneOELof", "max")
    _fill_numeric("gnomadGeneOELofUpper", "max")

    impact_map = {"-": 0, "MODIFIER": 1, "LOW": 2, "MODERATE": 3, "HIGH": 4}
    for k, v in impact_map.items():
        patient.loc[patient["IMPACT"] == k, "IMPACT"] = v
    patient["IMPACT"] = patient["IMPACT"].astype("float64")

    _fill_numeric("CADD_phred", "min", "mean")
    _fill_numeric("CADD_PHRED", "min", "mean")
    _fill_numeric("DANN_score", "min", "mean")

    # REVEL_score: split commas, max
    patient.loc[patient["REVEL_score"] == "-", "REVEL_score"] = np.nan
    for i in patient[~patient["REVEL_score"].isna()].index:
        score_list = str(patient.loc[i, "REVEL_score"]).split(",")
        score_list = [float(x) for x in score_list if x != "-" and x != "."]
        patient.loc[i, "REVEL_score"] = np.nan if not score_list else max(score_list)
    _fill_numeric("REVEL_score", "min", "mean")

    _fill_numeric("fathmm_MKL_coding_score", "min", "mean")

    for col in ["conservationScoreGnomad", "conservationScoreOELof"]:
        patient.loc[patient[col] == "-", col] = 1
        patient.loc[patient[col] == "Low", col] = 1
        patient.loc[patient[col] == "High", col] = 2
        patient[col] = patient[col].astype("float64")

    # Polyphen2_HDIV_score: split, max
    for col in ["Polyphen2_HDIV_score", "Polyphen2_HVAR_score"]:
        patient.loc[patient[col] == "-", col] = np.nan
        for i in patient[~patient[col].isna()].index:
            score_list = str(patient.loc[i, col]).split(",")
            score_list = [float(x) for x in score_list if x != "-" and x != "."]
            patient.loc[i, col] = np.nan if not score_list else max(score_list)
        _fill_numeric(col, "min", "50%")

    # SIFT_score: split, min (lower is more deleterious)
    patient.loc[patient["SIFT_score"] == "-", "SIFT_score"] = np.nan
    for i in patient[~patient["SIFT_score"].isna()].index:
        score_list = str(patient.loc[i, "SIFT_score"]).split(",")
        score_list = [float(x) for x in score_list if x != "-" and x != "."]
        patient.loc[i, "SIFT_score"] = np.nan if not score_list else min(score_list)
    _fill_numeric("SIFT_score", "max", "50%")

    zyg_map = {"HET": 1, "HOM": 2, "-": 0}
    for k, v in zyg_map.items():
        patient.loc[patient["zyg"] == k, "zyg"] = v
    patient["zyg"] = patient["zyg"].astype("float64")

    # FATHMM_score: split, min
    patient.loc[patient["FATHMM_score"] == "-", "FATHMM_score"] = np.nan
    for i in patient[~patient["FATHMM_score"].isna()].index:
        score_list = str(patient.loc[i, "FATHMM_score"]).split(",")
        score_list = [float(x) for x in score_list if x != "-" and x != "."]
        patient.loc[i, "FATHMM_score"] = np.nan if not score_list else min(score_list)
    _fill_numeric("FATHMM_score", "max", "50%")

    _fill_numeric("M_CAP_score", "min", "mean")

    # MutationAssessor_score: split, max
    patient.loc[patient["MutationAssessor_score"] == "-", "MutationAssessor_score"] = np.nan
    for i in patient[~patient["MutationAssessor_score"].isna()].index:
        score_list = str(patient.loc[i, "MutationAssessor_score"]).split(",")
        score_list = [float(x) for x in score_list if x != "-" and x != "."]
        patient.loc[i, "MutationAssessor_score"] = np.nan if not score_list else max(score_list)
    _fill_numeric("MutationAssessor_score", "min", "50%")

    for col in ["ESP6500_AA_AF", "ESP6500_EA_AF"]:
        patient.loc[patient[col] == "-", col] = 0.0
        patient[col] = patient[col].astype("float64")

    # hom: split, max, fillna 0
    patient.loc[patient["hom"] == "-", "hom"] = np.nan
    for i in patient[~patient["hom"].isna()].index:
        score_list = str(patient.loc[i, "hom"]).split(",")
        score_list = [float(x) for x in score_list if x != "-" and x != "."]
        patient.loc[i, "hom"] = np.nan if not score_list else max(score_list)
    patient["hom"] = pd.to_numeric(patient["hom"], errors="coerce").fillna(0).astype("float64")

    patient["hgmd_rs"] = patient["hgmd_rs"].apply(
        lambda x: x.split(",")[0] if type(x) == str else str(x)
    )
    patient.loc[patient["hgmd_rs"] == "-", "hgmd_rs"] = 0
    patient["hgmd_rs"] = patient["hgmd_rs"].astype("float64")

    # spliceAImax
    patient.loc[patient["spliceAImax"] == "-", "spliceAImax"] = np.nan
    if np.all(pd.isna(patient["spliceAImax"])):
        patient.loc[pd.isna(patient["spliceAImax"]), "spliceAImax"] = feature_stats.loc["spliceAImax", "min"]
        patient["spliceAImax"] = patient["spliceAImax"].astype("float64")
    else:
        patient["spliceAImax"] = patient["spliceAImax"].astype("float64")
        patient.loc[pd.isna(patient["spliceAImax"]), "spliceAImax"] = patient.loc[~pd.isna(patient["spliceAImax"]), "spliceAImax"].describe()["min"]

    # Consequence → one-hot cons_<consequence>
    consequence = [
        "transcript_ablation", "splice_acceptor_variant", "splice_donor_variant",
        "stop_gained", "frameshift_variant", "stop_lost", "start_lost",
        "transcript_amplification", "inframe_insertion", "inframe_deletion",
        "missense_variant", "protein_altering_variant", "splice_region_variant",
        "splice_donor_5th_base_variant", "splice_donor_region_variant",
    ]
    for cons in consequence:
        patient[f"cons_{cons}"] = patient["Consequence"].str.contains(cons).astype("int")
        variable_name.append(f"cons_{cons}")
    variable_name.remove("Consequence")

    # nc_isPLP, nc_isBLB, c_isPLP, c_isBLB: bool → 0/1
    for col in ["nc_isPLP", "nc_isBLB", "c_isPLP", "c_isBLB"]:
        patient.loc[patient[col] != True, col] = 0
        patient.loc[patient[col] == True, col] = 1
        patient[col] = patient[col].astype("float64")

    # nc_ClinVar_Exp, nc_HGMD_Exp: "nonCoding" → 1
    for col in ["nc_ClinVar_Exp", "nc_HGMD_Exp"]:
        patient.loc[patient[col] != "nonCoding", col] = 0
        patient.loc[patient[col] == "nonCoding", col] = 1
        patient[col] = patient[col].astype("float64")

    # c_ClinVar_Exp one-hot
    for exp in ["Del_to_Missense", "Different_pChange", "Same_pChange"]:
        patient[f"c_ClinVar_Exp_{exp}"] = patient["c_ClinVar_Exp"].str.contains(exp).astype("int")
        variable_name.append(f"c_ClinVar_Exp_{exp}")
    variable_name.remove("c_ClinVar_Exp")

    # c_HGMD_Exp one-hot
    for exp in ["Del_to_Missense", "Different_pChange", "Same_pChange", "Stop_Loss", "Start_Loss"]:
        patient[f"c_HGMD_Exp_{exp}"] = patient["c_HGMD_Exp"].str.contains(exp).astype("int")
        variable_name.append(f"c_HGMD_Exp_{exp}")
    variable_name.remove("c_HGMD_Exp")

    # CLNREVSTAT scoring
    cln_stat = [
        "-", "no_assertion_provided", "no_assertion_criteria_provided",
        "no_assertion_for_the_individual_variant",
        "criteria_provided,_single_submitter",
        "criteria_provided,_conflicting_interpretations",
        "criteria_provided,_multiple_submitters,_no_conflicts",
        "reviewed_by_expert_panel", "practice_guideline",
    ]
    cln_score = [0, 0, 0, 0, 1, 1, 2, 3, 4]
    for k, v in zip(cln_stat, cln_score):
        patient.loc[patient["nc_CLNREVSTAT"] == k, "nc_CLNREVSTAT"] = v
        patient.loc[patient["c_CLNREVSTAT"] == k, "c_CLNREVSTAT"] = v
    patient["nc_CLNREVSTAT"] = patient["nc_CLNREVSTAT"].astype("float64")
    patient["c_CLNREVSTAT"] = patient["c_CLNREVSTAT"].astype("float64")

    patient.loc[patient["nc_RANKSCORE"] == "-", "nc_RANKSCORE"] = 0
    patient["nc_RANKSCORE"] = patient["nc_RANKSCORE"].astype("float64")
    patient.loc[patient["c_RANKSCORE"] == "-", "c_RANKSCORE"] = 0
    patient["c_RANKSCORE"] = patient["c_RANKSCORE"].astype("float64")

    class_map = {"-": 0, "DM?": 1, "DM": 2}
    for k, v in class_map.items():
        patient.loc[patient["CLASS"] == k, "CLASS"] = v
    patient["CLASS"] = patient["CLASS"].astype("float64")

    # Negate cols (for groupby max-aggregation, then negate back)
    neg_cols = ["gnomadAF", "gnomadAFg", "gnomadGeneOELof", "gnomadGeneOELofUpper",
                "SIFT_score", "FATHMM_score", "ESP6500_AA_AF", "ESP6500_EA_AF"]
    patient.loc[:, neg_cols] = -patient.loc[:, neg_cols]
    patient = patient.groupby(["varId"], sort=False)[variable_name[1:]].max()
    patient.loc[:, neg_cols] = -patient.loc[:, neg_cols]

    # ---- Add tier features ----
    tier = tier_file.copy()
    tier_vars = [
        "IMPACT.from.Tier", "TierAD", "TierAR", "TierAR.adj",
        "No.Var.HM", "No.Var.H", "No.Var.M", "No.Var.L",
        "AD.matched", "AR.matched", "recessive", "dominant",
    ]
    tier.loc[:, ["TierAD", "TierAR", "TierAR.adj"]] = -tier.loc[:, ["TierAD", "TierAR", "TierAR.adj"]]
    tier = tier.groupby(["Uploaded_variation"], sort=False)[tier_vars].max()
    tier.loc[:, ["TierAD", "TierAR", "TierAR.adj"]] = -tier.loc[:, ["TierAD", "TierAR", "TierAR.adj"]]

    patient = pd.concat([patient, tier], axis=1)
    patient.loc[patient["IMPACT.from.Tier"].isna(), "IMPACT.from.Tier"] = 1
    patient.loc[patient["TierAD"].isna(), "TierAD"] = 4
    patient.loc[patient["TierAR"].isna(), "TierAR"] = 4
    patient.loc[patient["TierAR.adj"].isna(), "TierAR.adj"] = 4
    for var in tier_vars[4:8]:
        patient.loc[patient[var].isna(), var] = 0
    patient.loc[patient["AD.matched"].isna(), "AD.matched"] = 0
    patient.loc[patient["AR.matched"].isna(), "AR.matched"] = 0
    patient.loc[patient["recessive"].isna(), "recessive"] = 0
    patient.loc[patient["dominant"].isna(), "dominant"] = 0

    return patient


# ----------------------------------------------------------------------
# Parallel orchestrator
# ----------------------------------------------------------------------
def run_merge_parallel(samplesheet, work_dir, references, ref_ver,
                       max_workers, overwrite=True):
    """
    Run MERGE_SCORES_BY_CHROMOSOME in parallel across samples.

    samplesheet must have columns:
      sampleID, tier_tsv_paths, joined_scores_paths, phrank_txt_path
    """
    tasks = []
    results = {}
    ok = fail = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_merge_worker,
        initargs=(references, ref_ver),
        mp_context=mp.get_context("spawn"),
    ) as ex:
        for row in samplesheet.itertuples(index=True):
            future = ex.submit(
                _merge_worker_run,
                row.sampleID,
                row.tier_tsv_paths,
                row.joined_scores_paths,
                row.phrank_txt_path,
                work_dir, references, ref_ver, overwrite,
            )
            tasks.append({"index": row.Index, "sampleID": row.sampleID,
                           "future": future})

        with tqdm(total=len(tasks), desc="MERGE SCORES") as pbar:
            future_to_task = {t["future"]: t for t in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results[task["index"]] = future.result()
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] MERGE failed for {task['sampleID']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    if fail > 0:
        raise RuntimeError(f"MERGE_SCORES_BY_CHROMOSOME failed for {fail} sample(s).")
    return results