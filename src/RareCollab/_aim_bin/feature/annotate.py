from typing import Optional

import numpy as np
import pandas as pd

from annotation.annotation_pipeline import getAnnotateInfoRows_3


def _add_dgv_annotations(annotate_info_df: pd.DataFrame, dgv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized DGV annotation. Logic is IDENTICAL to the original per-row version:
    for each variant, find the first DGV record (in dgv_df row order) where
    Chr matches AND Start <= variant.start AND Stop >= variant.stop.
    """
    #print("=== USING VECTORIZED DGV ANNOTATION (NEW VERSION) ===")
    if dgv_df.empty:
        return annotate_info_df

    n = len(annotate_info_df)

    # Defaults match the original "no match" branch.
    dgv_type_list = [["-"] for _ in range(n)]
    dgv_subtype_list = [["-"] for _ in range(n)]
    dgv_dict_list = [[] for _ in range(n)]
    dgv_var_found = [0] * n  # Python int, matching original behavior

    # Convert variant coords once. astype(int) mirrors original int(var_obj.X)
    # behavior (works on both numeric and string columns).
    chrom_arr = annotate_info_df["chrom"].astype(int).to_numpy()
    start_arr = annotate_info_df["start"].astype(int).to_numpy()
    stop_arr = annotate_info_df["stop"].astype(int).to_numpy()

    # Group DGV by chromosome. groupby preserves within-group original row
    # order, so the "first match" semantics match the original mask + iloc[0].
    dgv_by_chr = {}
    for chrom_val, sub_df in dgv_df.groupby("Chr", sort=False):
        dgv_by_chr[int(chrom_val)] = (
            sub_df["Start"].to_numpy(dtype=np.int64),
            sub_df["Stop"].to_numpy(dtype=np.int64),
            sub_df["type"].to_numpy(),
            sub_df["subType"].to_numpy(),
        )

    unique_chroms = np.unique(chrom_arr)
    for c in unique_chroms:
        c_int = int(c)
        if c_int not in dgv_by_chr:
            continue
        dgv_start, dgv_stop, dgv_type_col, dgv_subtype_col = dgv_by_chr[c_int]

        var_idx = np.where(chrom_arr == c)[0]
        var_start_sub = start_arr[var_idx]
        var_stop_sub = stop_arr[var_idx]

        # Vectorized contain check: (n_var, n_dgv) bool matrix.
        contain_mask = (
            (dgv_start[None, :] <= var_start_sub[:, None])
            & (dgv_stop[None, :] >= var_stop_sub[:, None])
        )
        has_match = contain_mask.any(axis=1)
        # argmax returns the index of the first True (or 0 if all False).
        # We guard with has_match to skip the "all False" case.
        first_match_col = contain_mask.argmax(axis=1)

        for local_i, global_i in enumerate(var_idx):
            if has_match[local_i]:
                j = first_match_col[local_i]
                dgv_var_found[global_i] = 1
                dgv_type_list[global_i] = [dgv_type_col[j]]
                dgv_subtype_list[global_i] = [dgv_subtype_col[j]]
                # dgv_dict_list stays [] (legacy behavior)

    annotate_info_df["dgvDictList"] = dgv_dict_list
    annotate_info_df["dgvTypeList"] = dgv_type_list
    annotate_info_df["dgvSubtypeList"] = dgv_subtype_list
    annotate_info_df["dgvVarFound"] = dgv_var_found
    return annotate_info_df


def annotate_variants(
    transcript_df: pd.DataFrame,
    genome_ref: str,
    clinvar_gene_df: pd.DataFrame,
    clinvar_allele_df,
    omim_gene_sorted_df: pd.DataFrame,
    omim_allele_list,
    hgmd_hpo_gene_sorted_df: pd.DataFrame,
    decipher_sorted_df: pd.DataFrame,
    gnomad_metrics_sorted_df: pd.DataFrame,
    dgv_df: pd.DataFrame,
) -> pd.DataFrame:
    annotate_info_df = getAnnotateInfoRows_3(
        transcript_df,
        genome_ref,
        clinvar_gene_df,
        clinvar_allele_df,
        omim_gene_sorted_df,
        omim_allele_list,
        hgmd_hpo_gene_sorted_df,
        decipher_sorted_df,
        gnomad_metrics_sorted_df,
    )

    annotate_info_df = _add_dgv_annotations(annotate_info_df, dgv_df)

    return annotate_info_df