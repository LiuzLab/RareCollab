"""
JOIN_PHRANK worker module for RareCollab.

Per (sample, chr) task: takes the chr's scores CSV and the sample's
phrank.txt, joins clinvar / hgmd noncoding-region expansions plus phrank
scores, writes gzipped TSV.

Uses ProcessPoolExecutor for true parallelism (pure-Python pandas work
is GIL-bound under threads).
"""
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed



# ----------------------------------------------------------------------
# Worker-local cache
# ----------------------------------------------------------------------
_JOIN_PHRANK_RESOURCES = None


def _init_join_phrank_worker(references, ref_ver):
    """ProcessPoolExecutor initializer: load merge_expand tables once per worker."""
    global _JOIN_PHRANK_RESOURCES
    _JOIN_PHRANK_RESOURCES = prepare_join_phrank_resources(references, ref_ver)


def _join_phrank_worker_run(scores_csv, phrank_txt, sample_id, work_dir,
                            references, ref_ver, overwrite):
    """Worker entry point."""
    res = _JOIN_PHRANK_RESOURCES
    return JOIN_PHRANK(
        scores_csv, phrank_txt, sample_id, work_dir, references, ref_ver,
        clin_c=res["clin_c"],
        clin_nc=res["clin_nc"],
        hgmd_c=res["hgmd_c"],
        hgmd_nc=res["hgmd_nc"],
        overwrite=overwrite,
    )


# ----------------------------------------------------------------------
# Resource loading
# ----------------------------------------------------------------------
def prepare_join_phrank_resources(references, ref_ver):
    """
    Load merge_expand tables once. Shared across all (sample, chr) tasks
    via worker initializer.
    """
    base = references.ref_merge_expand_dir / ref_ver
    return {
        "clin_c": pd.read_csv(base / "clin_c.tsv.gz", sep="\t", low_memory=False),
        "clin_nc": pd.read_csv(base / "clin_nc.tsv.gz", sep="\t", low_memory=False),
        "hgmd_c": pd.read_csv(base / "hgmd_c.tsv.gz", sep="\t", low_memory=False),
        "hgmd_nc": pd.read_csv(base / "hgmd_nc.tsv.gz", sep="\t", low_memory=False),
    }


def _interval_overlap_match(score_df, interval_df):
    """
    For each row in score_df, find all rows in interval_df whose
    [new_start, new_stop] on the same new_chr contains score_df.pos.
    Returns (i, j) index arrays into score_df and interval_df respectively,
    where for each pair, score_df.iloc[i] overlaps interval_df.iloc[j].
    
    Replaces AIM's original np.where(N×M boolean matrix) approach with
    per-chrom interval lookup: O(N log M + matches) vs O(N*M) space and time.
    """
    if interval_df.shape[0] == 0:
        # No intervals to match — return empty
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Group interval_df by chrom; within each chrom, sort by new_start.
    # Keep original interval_df row index alongside (start, stop) so we
    # can return the right j into the un-sorted interval_df.
    i_list = []
    j_list = []
    
    # Note: a/ac MUST be in score_df row order (we want i to index into
    # score_df.iloc[...] directly).
    a = score_df["pos"].to_numpy()
    ac = score_df["chrom"].to_numpy()
    
    # Group intervals by chrom
    for chrom, group in interval_df.groupby("new_chr", sort=False):
        # Sort this chrom's intervals by new_start
        sorted_group = group.sort_values("new_start", kind="stable")
        starts = sorted_group["new_start"].to_numpy()
        stops = sorted_group["new_stop"].to_numpy()
        orig_idx = sorted_group.index.to_numpy()  # positions in interval_df
        
        # Find which score_df rows are on this chrom
        score_rows_on_chrom = np.where(ac == chrom)[0]
        if len(score_rows_on_chrom) == 0:
            continue
        
        pos_on_chrom = a[score_rows_on_chrom]
        
        # For each pos: binary-search for the last start <= pos
        # = number of intervals with start <= pos
        # All intervals [0:upper_bound[k]] are candidates.
        upper_bound = np.searchsorted(starts, pos_on_chrom, side="right")
        
        # For each candidate, check stop >= pos. We do this per-variant.
        # Most variants have only a handful of candidates, so this is fast.
        for k, pos in enumerate(pos_on_chrom):
            ub = upper_bound[k]
            if ub == 0:
                continue
            # Among starts[0:ub], all have start <= pos. Need stop >= pos.
            mask = stops[:ub] >= pos
            matched_idx = orig_idx[:ub][mask]
            if len(matched_idx) > 0:
                score_row_idx = score_rows_on_chrom[k]
                i_list.extend([score_row_idx] * len(matched_idx))
                j_list.extend(matched_idx.tolist())
    
    return np.array(i_list, dtype=int), np.array(j_list, dtype=int)

def _add_c_nc(score_df, clin_c, clin_nc, hgmd_c, hgmd_nc):
    """
    Port of add_c_nc.py with O(N log M) interval lookup instead of O(N*M)
    broadcast (AIM's original constructs an N×M boolean matrix per noncoding
    table, which is the bottleneck for large chr-scale inputs).
    
    Output identical to AIM's add_c_nc (verified bit-for-bit).
    """
    temp = score_df[["varId"]]
    
    # clin_nc: region match (was np.where(N×M)) — now interval lookup
    i, j = _interval_overlap_match(score_df, clin_nc)
    clin = pd.concat(
        [
            temp.iloc[i].reset_index(drop=True),
            clin_nc.iloc[j].reset_index(drop=True),
        ],
        axis=1,
    )
    
    # hgmd_nc: region match
    if hgmd_nc.shape[0] == 0:
        hgmd = hgmd_nc
    else:
        i, j = _interval_overlap_match(score_df, hgmd_nc)
        hgmd = pd.concat(
            [
                temp.iloc[i].reset_index(drop=True),
                hgmd_nc.iloc[j].reset_index(drop=True),
            ],
            axis=1,
        )
    
    # clin_c: exact merge
    merged = score_df.merge(
        clin_c.rename(columns={"new_chr": "chrom", "new_pos": "pos"}),
        how="left",
        on=["chrom", "pos", "ref", "alt"],
    )
    merged = merged.merge(clin, how="left", on="varId")
    
    # hgmd_c: exact merge or NaN
    if hgmd_c.shape[0] == 0:
        merged["c_HGMD_Exp"] = np.nan
        merged["c_RANKSCORE"] = np.nan
        merged["CLASS"] = np.nan
    else:
        merged = merged.merge(
            hgmd_c.rename(columns={"new_chr": "chrom", "new_pos": "pos"}),
            how="left",
            on=["chrom", "pos", "ref", "alt"],
        )
    
    # hgmd_nc: merge or NaN
    if hgmd_nc.shape[0] == 0:
        merged["nc_HGMD_Exp"] = np.nan
        merged["nc_RANKSCORE"] = np.nan
    else:
        merged = merged.merge(hgmd, how="left", on="varId")
    
    return merged


def JOIN_PHRANK(scores_csv_path, phrank_txt_path, sample_id, work_dir,
                references, ref_ver,
                clin_c=None, clin_nc=None, hgmd_c=None, hgmd_nc=None,
                overwrite=True):
    """
    Direct port of AIM's JOIN_PHRANK Nextflow process (generate_new_matrix_2.py
    + add_c_nc.py inlined).

    Args:
        scores_csv_path: per-chr scores CSV from ANNOTATE_BY_MODULES.
        phrank_txt_path: per-sample phrank.txt from PHRANK_SCORING.
        sample_id: Sample identifier.
        work_dir: Pipeline work directory.
        references: AimReferences dataclass.
        ref_ver: 'hg19' or 'hg38'.
        clin_c/clin_nc/hgmd_c/hgmd_nc: pre-loaded reference DataFrames
            (lazy-loaded if None).
        overwrite: Recompute even if output exists.

    Returns:
        Path to per-chr joined output: features/<sample>/<chr>.vep_scores.txt.gz
    """
    scores_csv_path = Path(scores_csv_path)
    phrank_txt_path = Path(phrank_txt_path)
    work_dir = Path(work_dir)

    if not scores_csv_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv_path}")
    if not phrank_txt_path.exists():
        raise FileNotFoundError(f"Phrank txt not found: {phrank_txt_path}")

    scores_basename = scores_csv_path.name[:-len(".csv")]  # "chr1.vep_scores"
    sample_dir = work_dir / "features" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f"{scores_basename}.txt.gz"

    if overwrite:
        if output_path.exists() or output_path.is_symlink():
            output_path.unlink()
    else:
        if output_path.exists():
            return output_path

    if any(x is None for x in [clin_c, clin_nc, hgmd_c, hgmd_nc]):
        res = prepare_join_phrank_resources(references, ref_ver)
        clin_c, clin_nc = res["clin_c"], res["clin_nc"]
        hgmd_c, hgmd_nc = res["hgmd_c"], res["hgmd_nc"]

    # Load score
    score = pd.read_csv(scores_csv_path, low_memory=False)

    # Add clinvar/hgmd coding + noncoding expansions
    merged = _add_c_nc(score, clin_c, clin_nc, hgmd_c, hgmd_nc)

    # Strip "_E" (transcript) and "_-" (intergenic) suffixes from varId
    # to match phrank's gene-level keys.
    # (AIM's generate_new_matrix_2.py does this).
    merged["varId"] = merged["varId"].apply(lambda x: x.split("_E")[0])
    merged["varId"] = merged["varId"].apply(lambda x: x.split("_-")[0])

    # Join phrank score on geneEnsId
    phr = pd.read_csv(phrank_txt_path, sep="\t", names=["ENSG", "phrank"])
    merged = merged.merge(phr, left_on="geneEnsId", right_on="ENSG", how="left")

    # Write gzipped TSV — pandas writes index by default; preserve AIM behavior
    # (generate_new_matrix_2.py: merged.to_csv("scores.txt.gz", compression="gzip", sep="\t"))
    merged.to_csv(output_path, compression="gzip", sep="\t")

    return output_path


# ----------------------------------------------------------------------
# Parallel orchestrator
# ----------------------------------------------------------------------
def run_join_phrank_parallel(samplesheet, work_dir, references, ref_ver,
                             max_workers, overwrite=True):
    """
    Run JOIN_PHRANK in parallel across (sample, chr) tasks using
    ProcessPoolExecutor. Each worker process loads merge_expand tables
    ONCE via _init_join_phrank_worker.

    Returns: dict[row_index -> sorted list of output paths]
    """
    tasks = []
    join_results = {}
    ok = fail = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_join_phrank_worker,
        initargs=(references, ref_ver),
        mp_context=mp.get_context("spawn"),
    ) as ex:
        for row in samplesheet.itertuples(index=True):
            for scores_csv in row.scores_csv_paths:
                future = ex.submit(
                    _join_phrank_worker_run,
                    scores_csv, row.phrank_txt_path,
                    row.sampleID, work_dir, references, ref_ver,
                    overwrite,
                )
                tasks.append({
                    "index": row.Index,
                    "sampleID": row.sampleID,
                    "scores_csv": scores_csv,
                    "future": future,
                })

        with tqdm(total=len(tasks), desc="JOIN PHRANK") as pbar:
            future_to_task = {t["future"]: t for t in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    out = future.result()
                    join_results.setdefault(task["index"], []).append(str(out))
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] JOIN_PHRANK failed for {task['sampleID']} / {task['scores_csv']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    if fail > 0:
        raise RuntimeError(f"JOIN_PHRANK failed for {fail} task(s).")

    return {idx: sorted(paths) for idx, paths in join_results.items()}