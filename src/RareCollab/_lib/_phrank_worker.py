"""
Phrank scoring worker module for RareCollab.

Provides Process-safe (i.e. picklable, module-level) versions of:
  - _load_loc_maps, _build_interval_lookup, _find_overlapping_genes
    (helpers; correct strict-containment replacement for AIM's
    location_to_gene.py binary_search)
  - prepare_phrank_resources       (one-shot reference loading)
  - PHRANK_SCORING                 (per-sample phrank pipeline)
  - run_phrank_parallel            (orchestrator using ProcessPoolExecutor)

ProcessPoolExecutor requires worker functions and their arguments to live at
module level so they can be pickled across the process boundary.

NOTE: Diverges from AIM's location_to_gene.py on purpose. AIM's binary_search
silently returns the nearest gene by sort position even when the variant
falls outside every gene interval. This module replaces that with strict
containment (start <= pos <= end), so phrank only sees genes that actually
contain the variant.
"""

import gzip
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from phrank import Phrank
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------------------------------------------------
# Worker-local cache: each ProcessPoolExecutor worker process loads these
# resources ONCE in its initializer, then reuses them across all samples it
# processes. Reduces per-sample overhead from ~5s (initial load) to ~0s.
# ----------------------------------------------------------------------
_WORKER_RESOURCES = None


def _init_worker(references):
    """ProcessPoolExecutor initializer: build per-worker resources once."""
    global _WORKER_RESOURCES
    _WORKER_RESOURCES = prepare_phrank_resources(references)


def _phrank_worker_run(final_vcf_path, hpo_path, sample_id, work_dir,
                       references, overwrite):
    """Worker entry point: dispatches to PHRANK_SCORING with cached resources."""
    res = _WORKER_RESOURCES
    return PHRANK_SCORING(
        final_vcf_path, hpo_path, sample_id, work_dir, references,
        res["interval_lookup"],
        res["ensembl_to_symbol"],
        res["symbol_to_ensembl"],
        res["phrank_instance"],
        overwrite,
    )


# ----------------------------------------------------------------------
# Core: strict-containment replacement for AIM's location_to_gene.py
# ----------------------------------------------------------------------
def _build_interval_lookup(df):
    """
    Build per-chrom (sorted_starts, sorted_ends, genes) tuples for
    O(log N + matches) gene containment lookup.

    Returns: dict[chrom -> (starts: np.ndarray, ends: np.ndarray, genes: list[str])]

    Note: starts is sorted ascending; ends and genes follow the same row order
    as starts (i.e. ends is NOT independently sorted).
    """
    out = {}
    for chrom, g in df.groupby("chrom", sort=False):
        g_sorted = g.sort_values("start", kind="mergesort")  # stable sort
        out[chrom] = (
            g_sorted["start"].to_numpy(),
            g_sorted["end"].to_numpy(),
            g_sorted["gene"].to_list(),
        )
    return out


def _find_overlapping_genes(interval_lookup, chrom, pos):
    """
    Return the set of gene IDs whose [start, end] interval contains pos
    on the given chrom. Strict containment: start <= pos <= end.

    Multiple overlapping genes are all returned -- downstream phrank scoring
    decides per-gene weights.
    """
    if chrom not in interval_lookup:
        return set()

    starts, ends, genes = interval_lookup[chrom]

    # Step 1: binary search for the upper bound of intervals with start <= pos.
    # np.searchsorted(..., side='right') returns the insertion index such that
    # starts[0:upper] are all <= pos.
    upper = np.searchsorted(starts, pos, side="right")

    if upper == 0:
        # pos is to the left of all gene starts
        return set()

    # Step 2: among the first `upper` candidates, keep those with end >= pos.
    candidate_ends = ends[:upper]
    matched_mask = candidate_ends >= pos

    if not matched_mask.any():
        return set()

    # Step 3: collect all matching genes
    matched_indices = np.where(matched_mask)[0]
    return {genes[i] for i in matched_indices}


def _load_loc_maps(loc_file):
    """
    Load Ensembl gene location reference and build per-chrom interval lookup
    for O(log N + matches) strict-containment query.

    Symbols containing "." (e.g. ENSG...XX.YY versioned forms) are skipped to
    match AIM's location_to_gene.py behavior.

    Strips "chr" prefix from chrom values so they match VCF-style chrom IDs.
    """
    df = pd.read_csv(
        loc_file, sep="\t", header=None,
        names=["gene", "chrom", "start", "end"], low_memory=False,
    )
    df = df[~df["gene"].astype(str).str.contains(r"\.", regex=True)]
    df["chrom"] = df["chrom"].astype(str).str.replace(r"^chr", "", regex=True)
    return _build_interval_lookup(df)


# ----------------------------------------------------------------------
# Resource prep + per-sample function
# ----------------------------------------------------------------------
def prepare_phrank_resources(references):
    """
    Build per-sample-independent resources for PHRANK_SCORING:
      - interval_lookup: per-chrom (starts, ends, genes) for containment queries
      - ensembl_to_symbol / symbol_to_ensembl: bidirectional Ensembl <-> symbol
        maps used for the symbol round-trip that mirrors AIM's Nextflow bash.
      - phrank_instance: Phrank ranking engine.
    """
    interval_lookup = _load_loc_maps(references.ensembl_to_location_file)

    ensembl_to_symbol = {}
    symbol_to_ensembl = {}
    with open(references.ensembl_to_symbol_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            eid, symbol = parts[0], parts[1]
            ensembl_to_symbol[eid] = symbol
            symbol_to_ensembl.setdefault(symbol, set()).add(eid)

    phrank_instance = Phrank(
        str(references.phrank.dagfile),
        str(references.phrank.disease_annotation),
        str(references.phrank.disease_gene),
        str(references.phrank.gene_annotation),
    )
    return {
        "interval_lookup": interval_lookup,
        "ensembl_to_symbol": ensembl_to_symbol,
        "symbol_to_ensembl": symbol_to_ensembl,
        "phrank_instance": phrank_instance,
    }


def PHRANK_SCORING(final_vcf_path, hpo_path, sample_id, work_dir, references,
                   interval_lookup=None, ensembl_to_symbol=None,
                   symbol_to_ensembl=None, phrank_instance=None, overwrite=True):
    """
    Compute phrank scores: per-gene ranking based on phenotype similarity
    between patient's HPO terms and disease-gene associations.

    Direct port of AIM's PHRANK_SCORING Nextflow subworkflow:
      VCF_TO_VARIANTS -> VARIANTS_TO_ENSEMBL -> ENSEMBL_TO_GENESYM -> GENESYM_TO_PHRANK

    The VARIANTS_TO_ENSEMBL step uses strict gene containment (start <= pos <=
    end) -- diverges from AIM's buggy "nearest gene by sort position" behavior.
    """
    final_vcf_path = Path(final_vcf_path)
    hpo_path = Path(hpo_path)
    work_dir = Path(work_dir)

    if not final_vcf_path.exists():
        raise FileNotFoundError(f"VCF not found for sample {sample_id}: {final_vcf_path}")
    if not hpo_path.exists():
        raise FileNotFoundError(f"HPO file not found for sample {sample_id}: {hpo_path}")

    sample_dir = work_dir / "process_vcf" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f"{sample_id}.phrank.txt"

    if overwrite:
        if output_path.exists() or output_path.is_symlink():
            output_path.unlink()
    else:
        if output_path.exists():
            return output_path

    if interval_lookup is None:
        interval_lookup = _load_loc_maps(references.ensembl_to_location_file)

    if ensembl_to_symbol is None or symbol_to_ensembl is None:
        ensembl_to_symbol = {}
        symbol_to_ensembl = {}
        with open(references.ensembl_to_symbol_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                eid, symbol = parts[0], parts[1]
                ensembl_to_symbol[eid] = symbol
                symbol_to_ensembl.setdefault(symbol, set()).add(eid)

    if phrank_instance is None:
        phrank_instance = Phrank(
            str(references.phrank.dagfile),
            str(references.phrank.disease_annotation),
            str(references.phrank.disease_gene),
            str(references.phrank.gene_annotation),
        )

    # STEP 1: VCF -> unique variants
    variants = set()
    opener = gzip.open if final_vcf_path.suffix == ".gz" else open
    with opener(final_vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            chrom = cols[0]
            if chrom.startswith("chr"):
                chrom = chrom[3:]
            variants.add(f"{chrom}:{cols[1]}:{cols[3]}:{cols[4]}")

    # STEP 2: variants -> Ensembl gene IDs (strict containment)
    variant_ensembl_ids = set()
    for var in variants:
        parts = var.split(":")
        chrom = parts[0]
        pos = int(parts[1])
        variant_ensembl_ids.update(_find_overlapping_genes(interval_lookup, chrom, pos))

    # STEP 3: symbol round-trip (matches Nextflow bash double join)
    symbols_hit = {ensembl_to_symbol[eid] for eid in variant_ensembl_ids
                   if eid in ensembl_to_symbol}
    unique_ensembl_ids = set()
    for s in symbols_hit:
        unique_ensembl_ids.update(symbol_to_ensembl[s])

    # STEP 4: patient HPO terms
    patient_hpos = set()
    with open(hpo_path) as f:
        for line in f:
            term = line.strip().split("\t")[0]
            if term:
                patient_hpos.add(term)

    # STEP 5: rank
    ranking = phrank_instance.rank_genes(unique_ensembl_ids, patient_hpos)

    with open(output_path, "w") as out:
        for item in ranking:
            out.write(f"{item[1]}\t{item[0]}\n")

    return output_path


# ----------------------------------------------------------------------
# Parallel orchestrator
# ----------------------------------------------------------------------
def run_phrank_parallel(samplesheet, work_dir, references, max_workers,
                        overwrite=True):
    """
    Run PHRANK_SCORING in parallel across samples using ProcessPoolExecutor.

    Each worker process loads phrank resources ONCE via _init_worker, then
    processes its assigned samples. References are passed via initargs (must
    be picklable -- verified separately).

    Returns: dict[row_index -> output_path_str]
    """
    tasks = []
    phrank_results = {}
    ok = fail = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(references,),
        mp_context=mp.get_context("spawn"),
    ) as ex:
        for row in samplesheet.itertuples(index=True):
            future = ex.submit(
                _phrank_worker_run,
                row.final_vcf_path,
                row.hpo_path,
                row.sampleID,
                work_dir,
                references,
                overwrite,
            )
            tasks.append({"index": row.Index, "sampleID": row.sampleID, "future": future})

        with tqdm(total=len(tasks), desc="PHRANK SCORING") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    phrank_txt = future.result()
                    phrank_results[task["index"]] = str(phrank_txt)
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] PHRANK_SCORING failed for sample {task['sampleID']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    if fail > 0:
        raise RuntimeError(f"PHRANK_SCORING failed for {fail} sample(s).")

    return phrank_results