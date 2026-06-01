"""
HPO_SIM worker module for RareCollab.

Provides Process-safe (i.e. picklable, module-level) versions of:
  - prepare_hpo_resources          (loads ontology + builds IC + sparse ancestors)
  - HPO_SIM                        (per-sample HGMD + OMIM similarity scoring)
  - run_hpo_sim_parallel           (orchestrator using ProcessPoolExecutor)

ProcessPoolExecutor requires worker functions and their arguments to live at
module level so they can be pickled across the process boundary. Heavy
resources (ontology, IC array, ancestor sparse matrix) are NOT pickled across
processes — each worker process loads them ONCE via _init_worker, then
caches them in a module-level _WORKER_RESOURCES dict for reuse across
all samples it processes.
"""

import re
import pronto
import pyreadr
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------------------------------------------------
# Worker-local cache: each ProcessPoolExecutor worker process loads these
# resources ONCE in its initializer, then reuses them across all samples it
# processes. Reduces per-sample overhead from ~15s (initial load) to ~0s.
# ----------------------------------------------------------------------
_WORKER_RESOURCES = None


def _init_worker(references):
    """ProcessPoolExecutor initializer: build per-worker resources once."""
    global _WORKER_RESOURCES
    _WORKER_RESOURCES = {
        "references": references,
        "hpo_resources": prepare_hpo_resources(references),
    }


def _hpo_sim_worker_run(hpo_path, sample_id, work_dir, overwrite):
    """Worker entry point: dispatches to HPO_SIM with cached resources."""
    res = _WORKER_RESOURCES
    return HPO_SIM(
        hpo_path, sample_id, work_dir,
        res["references"],
        res["hpo_resources"],
        overwrite,
    )


# ----------------------------------------------------------------------
# Core: HPO ontology loading + Lin similarity
# ----------------------------------------------------------------------
def _ancestors_with_self(hpo_ontology, term_id):
    """Return ancestor HPO IDs of `term_id` INCLUDING `term_id` itself.
    Matches R's ontologyIndex `ontology$ancestors[[t]]` behavior."""
    if term_id not in hpo_ontology:
        return set()
    term = hpo_ontology[term_id]
    ancs = {term.id}
    ancs.update(a.id for a in term.superclasses(with_self=False))
    return ancs


def _lin_similarity_matrix(query_terms, target_term_lists, hpo_resources):
    """
    Vectorized Lin similarity + best-match-average, using sparse ancestor matrix.
    Per-query work releases the GIL during numpy ops, so this scales with threads.
    """
    term_id_to_idx = hpo_resources["term_id_to_idx"]
    ic_array = hpo_resources["ic_array"]
    ancestors_sparse = hpo_resources["ancestors_sparse"]
    n = len(ic_array)

    valid_q_idx = [term_id_to_idx[t] for t in query_terms if t in term_id_to_idx]
    if not valid_q_idx:
        return np.zeros(len(target_term_lists))

    n_q = len(valid_q_idx)
    lin_matrix = np.zeros((n_q, n), dtype=np.float64)

    for row, q_idx in enumerate(valid_q_idx):
        q_anc_cols = ancestors_sparse[q_idx].indices
        if len(q_anc_cols) == 0:
            continue

        sub = ancestors_sparse[:, q_anc_cols]
        ic_sub = ic_array[q_anc_cols]

        weighted = sub.multiply(ic_sub)
        mica_ic = np.asarray(weighted.max(axis=1).todense()).flatten()

        q_ic = ic_array[q_idx]
        denom = q_ic + ic_array
        with np.errstate(divide='ignore', invalid='ignore'):
            lin_row = np.where(denom > 0, 2.0 * mica_ic / denom, 0.0)
        lin_matrix[row] = lin_row

    scores = np.zeros(len(target_term_lists))
    for i, target_terms in enumerate(target_term_lists):
        valid_t_idx = [term_id_to_idx[t] for t in target_terms if t in term_id_to_idx]
        if not valid_t_idx:
            continue
        sub = lin_matrix[:, valid_t_idx]
        scores[i] = sub.max(axis=1).mean()

    return scores


def prepare_hpo_resources(references):
    """
    Load HPO ontology, compute information content, and build vectorization
    structures so that HPO_SIM can run in numpy.

    Returns a dict with:
        hpo_ontology: pronto.Ontology
        term_ic: dict[str, float]
        term_id_to_idx: dict[str, int]
        idx_to_term_id: list[str]
        ic_array: np.ndarray[N]
        ancestors_sparse: scipy.sparse.csr_matrix[N, N] bool

    Excludes obsolete terms from IC denominator N -- stricter than R's
    ontologyIndex default. See validation_report for design notes.
    """
    omim_refs = references.omim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnicodeWarning)
        hpo_ontology = pronto.Ontology(str(omim_refs.omim_obo))

    idx_to_term_id = [t.id for t in hpo_ontology.terms() if not t.obsolete]
    term_id_to_idx = {tid: i for i, tid in enumerate(idx_to_term_id)}
    n = len(idx_to_term_id)

    rows, cols = [], []
    for i, tid in enumerate(idx_to_term_id):
        ancs = _ancestors_with_self(hpo_ontology, tid)
        for a in ancs:
            if a in term_id_to_idx:
                rows.append(i)
                cols.append(term_id_to_idx[a])

    data = np.ones(len(rows), dtype=np.bool_)
    ancestors_sparse = csr_matrix(
        (data, (rows, cols)), shape=(n, n), dtype=np.bool_
    )

    descendant_counts = np.asarray(ancestors_sparse.sum(axis=0)).flatten()
    ic_array = np.zeros(n, dtype=np.float64)
    nonzero = descendant_counts > 0
    ic_array[nonzero] = -np.log(descendant_counts[nonzero] / n)

    term_ic = {idx_to_term_id[i]: ic_array[i] for i in range(n)}

    return {
        "hpo_ontology": hpo_ontology,
        "term_ic": term_ic,
        "term_id_to_idx": term_id_to_idx,
        "idx_to_term_id": idx_to_term_id,
        "ic_array": ic_array,
        "ancestors_sparse": ancestors_sparse,
    }


def HPO_SIM(hpo_path, sample_id, work_dir, references, hpo_resources, overwrite=True):
    """
    Compute HPO similarity scores between the patient's HPO terms and:
        - HGMD gene phenotypes (output: <sample_id>-cz)
        - OMIM disease phenotypes (output: <sample_id>-dx)
    """
    hpo_path = Path(hpo_path)
    work_dir = Path(work_dir)

    if not hpo_path.exists():
        raise FileNotFoundError(
            f"HPO file not found for sample {sample_id}: {hpo_path}"
        )

    sample_dir = work_dir / "features" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_cz = sample_dir / f"{sample_id}-cz"
    output_dx = sample_dir / f"{sample_id}-dx"

    if overwrite:
        for p in [output_cz, output_dx]:
            if p.exists() or p.is_symlink():
                p.unlink()
    else:
        if output_cz.exists() and output_dx.exists():
            return output_cz, output_dx

    omim_refs = references.omim
    hpo_ontology = hpo_resources["hpo_ontology"]

    # ---- Load patient HPO terms ----
    with open(hpo_path) as f:
        patient_hpo = [line.strip().split("\t")[0] for line in f if line.strip()]
    patient_hpo = [t for t in patient_hpo if re.match(r"^HP:\d{7}$", t)]
    if not patient_hpo:
        patient_hpo = ["HP:0000001"]

    # ============================================================
    # PART 1: HGMD gene-phenotype similarity -> <sample_id>-cz
    # ============================================================
    hgmd_df = pd.read_csv(omim_refs.omim_hgmd_phen, sep="\t")

    if len(hgmd_df) == 0:
        empty_cols = ["acc_num", "phen_id", "gene_sym", "HPO", "HPO_list", "Similarity_Score"]
        pd.DataFrame(columns=empty_cols).to_csv(output_cz, sep="\t", index=False)
    else:
        hgmd_df = hgmd_df.dropna(subset=["hpo_id"])

        grouped_hgmd = (
            hgmd_df.groupby(["acc_num", "phen_id", "gene_sym"], dropna=False)["hpo_id"]
            .apply(lambda s: " ".join(s.astype(str)))
            .reset_index()
            .rename(columns={"hpo_id": "HPO"})
        )
        grouped_hgmd["HPO_list"] = grouped_hgmd["HPO"].str.split(" ")

        sim_scores = _lin_similarity_matrix(
            patient_hpo,
            grouped_hgmd["HPO_list"].tolist(),
            hpo_resources,
        )
        grouped_hgmd["Similarity_Score"] = sim_scores
        grouped_hgmd["HPO_list"] = grouped_hgmd["HPO_list"].apply(lambda lst: "|".join(lst))
        grouped_hgmd = grouped_hgmd.sort_values("Similarity_Score", ascending=False)
        grouped_hgmd.to_csv(output_cz, sep="\t", index=False)

    # ============================================================
    # PART 2: OMIM disease similarity -> <sample_id>-dx
    # ============================================================
    omim_hpo = pd.read_csv(
        omim_refs.omim_pheno,
        sep="\t",
        comment="#",
        quotechar='"',
        on_bad_lines="skip",
    )
    omim_hpo = omim_hpo[["OMIM_ID", "DiseaseName", "HPO_ID"]].rename(
        columns={"DiseaseName": "Disease_Name", "HPO_ID": "HPO_ID"}
    ).drop_duplicates()

    grouped_omim = (
        omim_hpo.groupby(["OMIM_ID", "Disease_Name"], dropna=False)["HPO_ID"]
        .apply(lambda s: " ".join(s.astype(str)))
        .reset_index()
        .rename(columns={"HPO_ID": "HPO"})
    )
    grouped_omim["HPO_list"] = grouped_omim["HPO"].str.split(" ")

    sim_scores = _lin_similarity_matrix(
        patient_hpo,
        grouped_omim["HPO_list"].tolist(),
        hpo_resources,
    )
    grouped_omim["Similarity_Score"] = sim_scores

    def _terms_to_names(term_list):
        names = []
        for t in term_list:
            if t in hpo_ontology:
                names.append(hpo_ontology[t].name)
        return "|".join(names)

    grouped_omim["HPO_term"] = grouped_omim["HPO_list"].apply(_terms_to_names)

    # ---- Merge with OMIM genemap2 to attach gene info ----
    rds_data = pyreadr.read_r(str(omim_refs.omim_genemap2))
    genemap2 = next(iter(rds_data.values()))

    genemap2_uniq = genemap2[
        ["Pheno_ID", "Approved_Gene_Symbol", "Ensembl_Gene_ID", "Entrez_Gene_ID"]
    ].drop_duplicates()

    omim_with_gene = pd.merge(
        genemap2_uniq,
        grouped_omim[["OMIM_ID", "Disease_Name", "Similarity_Score", "HPO_term"]],
        left_on="Pheno_ID",
        right_on="OMIM_ID",
    )
    omim_with_gene = omim_with_gene.rename(columns={"Approved_Gene_Symbol": "Gene_Symbol"})
    omim_with_gene = omim_with_gene.drop(columns=["OMIM_ID"])

    omim_with_gene = omim_with_gene.sort_values("Similarity_Score", ascending=False)
    omim_with_gene = omim_with_gene[omim_with_gene["Similarity_Score"] >= 0]

    omim_with_gene.to_csv(output_dx, sep="\t", index=False)

    if not output_cz.exists():
        raise RuntimeError(f"HPO_SIM output -cz not created for {sample_id}: {output_cz}")
    if not output_dx.exists():
        raise RuntimeError(f"HPO_SIM output -dx not created for {sample_id}: {output_dx}")

    return output_cz, output_dx


# ----------------------------------------------------------------------
# Parallel orchestrator
# ----------------------------------------------------------------------
def run_hpo_sim_parallel(samplesheet, work_dir, references, max_workers, overwrite=True):
    """
    Run HPO_SIM in parallel across samples using ProcessPoolExecutor.

    Each worker process loads HPO ontology + builds IC + sparse ancestor
    matrix ONCE via _init_worker, then processes its assigned samples.

    Returns: dict[row_index -> {"hgmd_sim_path": str, "omim_sim_path": str}]
    """
    tasks = []
    hpo_results = {}
    ok = fail = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(references,),
        mp_context=mp.get_context("spawn"),
    ) as ex:
        for row in samplesheet.itertuples(index=True):
            future = ex.submit(
                _hpo_sim_worker_run,
                row.hpo_path,
                row.sampleID,
                work_dir,
                overwrite,
            )
            tasks.append({
                "index": row.Index,
                "sampleID": row.sampleID,
                "hpo_path": row.hpo_path,
                "future": future,
            })

        with tqdm(total=len(tasks), desc="HPO SIM") as pbar:
            future_to_task = {task["future"]: task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                sample_id = task["sampleID"]
                row_index = task["index"]

                try:
                    cz, dx = future.result()
                    hpo_results[row_index] = {
                        "hgmd_sim_path": str(cz),
                        "omim_sim_path": str(dx),
                    }
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] HPO_SIM failed for sample {sample_id}")
                    print(f"[ERROR] Input HPO: {task['hpo_path']}")
                    print(f"[ERROR] {type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(done=ok, fail=fail)

    if fail > 0:
        raise RuntimeError(f"HPO_SIM failed for {fail} sample(s).")

    return hpo_results