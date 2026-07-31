#!/usr/bin/env python
# coding: utf-8
"""
RNA Agent.

Reads each sample's candidate table, quantifies raw allele support straight
from the BAM, and asks the model for a gene-level and a variant-level reading
of the RNA evidence.

    Agents/RNA/AlleleQuantification/<sampleID>.feather
    Agents/RNA/GeneLevelEval/<sampleID>/<gene>.json
    Agents/RNA/VariantLevelEval/<sampleID>/<gene>_VarId_<varId>.json
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_prompt_RNA_agent_gene(gene: str, fraser_pvalue_gene: float, outrider_items: dict, gene_constraints: dict, inheritance_flags: dict, clingen_dosage: list) -> str:
    od_lines = []
    od_lines.append(f"""- OUTRIDER # RawCounts: {outrider_items['RawCounts']} | RawCount_ZScore: {outrider_items['RawCount_ZScore']} | OUTRIDER_ZScore: {outrider_items['OUTRIDER_ZScore']} | OUTRIDER_pValue: {outrider_items['OUTRIDER_pValue']}""")

    pli = float(f"{gene_constraints['gnomadGenePLI']:.4g}")
    oe_lof = float(f"{gene_constraints['gnomadGeneOELof']:.4g}")
    oe_lof_u = float(f"{gene_constraints['gnomadGeneOELofUpper']:.4g}")

    inh_parts = []
    if inheritance_flags["recessive"] == 1:
        inh_parts.append("recessive")
    if inheritance_flags["dominant"] == 1:
        inh_parts.append("dominant")
    inh_text = ", ".join(inh_parts) if inh_parts else "unknown"

    fr_block = (
        f"- geneMinPValue: {fraser_pvalue_gene} (smaller values indicate stronger evidence of any aberrant splicing across this gene; "
        "≈0.01–0.1 = moderate, ≤0.01 = strong, >0.1 = usually weak or no signal)."
        if fraser_pvalue_gene is not None
        else "(no gene-level splicing value available for this gene)."
    )
    od_block = "\n".join(od_lines) if od_lines else "(no OUTRIDER signals retained)"

    prompt = f"""You are an RNA-seq interpretation assistant.

Gene: {gene}

Context:
- FRASER2 tests splicing junction by junction. The value shown below, `geneMinPValue`, is the smallest junction-level p-value observed anywhere in this gene. It is a screening signal rather than a multiple-testing-corrected gene statistic, so a gene with many junctions reaches a small value somewhat more easily by chance. Judge it on this scale: geneMinPValue ≤ 0.01 is strong evidence of aberrant splicing somewhere in the gene, 0.01–0.1 is moderate, and > 0.1 is usually weak or no evidence. It does not indicate which exon or intron is affected.
- OUTRIDER identifies expression outliers at the gene level using z-scores; large negative z-scores (e.g., ≤ -1.5) suggest decreased expression, while large positive z-scores (e.g., ≥ +1.5) suggest increased expression. Larger |z| indicates stronger expression outlier (e.g., |z| ≈ 1.5–3 moderate, |z| ≥ 3 large).
- ASE (allele-specific expression) tests imbalance between the two alleles, but ASE results are not provided for this gene-level summary.

This gene-level summary is generated when at least one of the following is present for this gene:
- Splicing evidence: geneMinPValue ≤ 0.1
- OUTRIDER expression outlier: |OUTRIDER_zScore| ≥ 1.5

Additional considerations:
- Gene constraint: pLI (higher => loss-of-function less tolerated); gnomAD LOF observed/expected (o/e) and upper bound contextualize constraint.
- Inheritance hint: {inh_text}.

Gene constraint (gnomAD):
- pLI: {pli}
- o/e LOF: {oe_lof} (upper: {oe_lof_u}){clingen_dosage}

Evidence modules for this gene:

FRASER2 (gene-level splicing):
{fr_block}

OUTRIDER (expression outlier):
{od_block}

Task:
Classify the RNA evidence into exactly ONE of the following Events:
- No Signal
- Abnormal Splicing
- Moderately Decreased Expression
- Largely Decreased Expression
- Increased Expression

When deciding the Event:
- For splicing-based events (Abnormal Splicing), rely on whether geneMinPValue is small enough to indicate genuine splicing outliers in this gene. Note that it does not specify the exact exon/intron pattern, only that splicing is abnormal somewhere in the gene.
- For expression-based events (Moderately/Largely Decreased Expression, Increased Expression), use the magnitude and sign of OUTRIDER_zScore (e.g., |z| ≈ 1.5–3 = moderate, |z| ≥ 3 = large).

Then, considering the gene-level splicing signal, the OUTRIDER gene-level outlier signal, the gene constraint (pLI/o-e), and the inheritance tendency noted above, provide an overall confidence level:
- Conclusion ∈ {{No RNA Evidence, Weak RNA Evidence, Strong RNA Evidence}}.

Return exactly one line JSON object with keys "Reasoning", "Event" and "Conclusion". No markdown. No code fences. No extra keys.
Reasoning: <succinct rationale grounded in the gene-level splicing value, OUTRIDER values, gene constraint, and inheritance>
Event: must be exactly one of 'No Signal', 'Abnormal Splicing', 'Moderately Decreased Expression', 'Largely Decreased Expression', 'Increased Expression'
Conclusion: must be exactly one of 'No RNA Evidence', 'Weak RNA Evidence', 'Strong RNA Evidence'.

Output schema (must match exactly):{{"Reasoning":"<string>","Event":"<No Signal|Abnormal Splicing|Moderately Decreased Expression|Largely Decreased Expression|Increased Expression>","Conclusion":"<No RNA Evidence|Weak RNA Evidence|Strong RNA Evidence>"}}
If you cannot comply, output exactly:{{"Reasoning":"comply_error","Event":"No Signal","Conclusion":"No RNA Evidence"}}
"""
    return prompt


def build_prompt_RNA_agent_variant(gene: str, fraser_items: dict, outrider_items: dict, ase_items: dict, gene_constraints: dict, inheritance_flags: dict, clingen_dosage: str) -> str:
    fr_lines = [f"""- FRASER2 #
    chrom: {fraser_items['Chrom']} | junction_start: {fraser_items['FRASER_junction_start']} | junction_end: {fraser_items['FRASER_junction_end']} | pvaluesBetaBinomial_jaccard: {fraser_items['pvaluesBetaBinomial_jaccard']}
    jaccard: {fraser_items['jaccard']} | predictedMeans_jaccard: {fraser_items['predictedMeans_jaccard']} | delta_jaccard: {fraser_items['delta_jaccard']} | rawOtherCounts_jaccard: {fraser_items['rawOtherCounts_jaccard']}
    psi5: {fraser_items['psi5']} | rawOtherCounts_psi5: {fraser_items['rawOtherCounts_psi5']} | delta_psi5: {fraser_items['delta_psi5']}
    psi3: {fraser_items['psi3']} | rawOtherCounts_psi3: {fraser_items['rawOtherCounts_psi3']} | delta_psi3: {fraser_items['delta_psi3']}
    rawCountsJnonsplit: {fraser_items['rawCountsJnonsplit']}
    ClinVar in junction region -> P/LP: {fraser_items['P_LP_count']} | B/LB: {fraser_items['B_LB_count']} | ratio (P/LP ÷ B/LB): {fraser_items['P_LP_to_B_LB_ratio']}
    """]

    if outrider_items != None:
        od_lines = [f"""- OUTRIDER
        RawCounts: {outrider_items['RawCounts']} | RawCount_ZScore: {outrider_items['RawCount_ZScore']} | OUTRIDER_ZScore: {outrider_items['OUTRIDER_ZScore']} | OUTRIDER_pValue: {outrider_items['OUTRIDER_pValue']}
        """]
    else:
        od_lines = []

    if ase_items != None:
        ase_lines = [f"""- ASE
        ASE_PVAL: {ase_items['ASE_PVAL']} | REF_COUNT: {ase_items['REF_COUNT']} | ALT_COUNT: {ase_items['ALT_COUNT']} | ALT_RATIO: {ase_items['ALT_RATIO']} | zyg = heterozygous
        """]
    else:
        ase_lines = []

    pli = float(f"{gene_constraints['gnomadGenePLI']:.4g}")
    oe_lof = float(f"{gene_constraints['gnomadGeneOELof']:.4g}")
    oe_lof_u = float(f"{gene_constraints['gnomadGeneOELofUpper']:.4g}")

    inh_parts = []
    if inheritance_flags["recessive"] == 1:
        inh_parts.append("recessive")
    if inheritance_flags["dominant"] == 1:
        inh_parts.append("dominant")
    inh_text = ", ".join(inh_parts) if inh_parts else "unknown"

    fr_block = "\n".join(fr_lines) if fr_lines else "(no FRASER2 signals retained)"
    od_block = "\n".join(od_lines) if od_lines else "(no OUTRIDER signals retained)"
    ase_block = "\n".join(ase_lines) if ase_lines else "(no ASE signals retained)"

    clingen_block = clingen_dosage

    prompt = f"""You are an RNA-seq interpretation assistant.

Gene: {gene}

Context:
- FRASER2 provides a junction-level outlier p-value based on the jaccard metric: `pvaluesBetaBinomial_jaccard` (beta–binomial p-value). Smaller values mean stronger evidence for aberrant splicing at/around this junction; values ≤ 0.01 are strong, 0.01–0.1 are moderate, and > 0.1 are usually weak or no signal.
- `jaccard` measures splice-junction inclusion/exclusion using split and non-split read information; `predictedMeans_jaccard` is the expected value from the model; `delta_jaccard = observed - expected`. Large |delta_jaccard| supports stronger deviation from expectation, often consistent with exon skipping, intron retention, or cryptic splice usage (interpret together with psi5/psi3 patterns).
- `psi5`/`psi3` summarize splice-site usage; `delta_psi5`/`delta_psi3` are observed minus expected usage (negative indicates decreased usage; positive indicates increased usage). These help determine the splicing pattern type, but the statistical significance is captured here by `pvaluesBetaBinomial_jaccard`.
- OUTRIDER identifies expression outliers at the gene level using z-scores; large negative z-scores (e.g., ≤ -1.5) suggest decreased expression, while large positive z-scores (e.g., ≥ +1.5) suggest increased expression. Larger |z| indicates stronger expression outlier (e.g., |z| ≈ 1.5–3 moderate, |z| ≥ 3 large).
- ASE (allele-specific expression) tests imbalance between the two alleles; here we only consider heterozygous sites (zyg=1). `ALT_RATIO` is the fraction of reads supporting ALT. Importantly, ALT_RATIO = 0.5 means perfectly balanced (no allelic bias). ALT_RATIO > 0.5 indicates ALT higher than REF; ALT_RATIO < 0.5 indicates ALT lower than REF.

All entries shown below satisfy at least one of:
- pvaluesBetaBinomial_jaccard ≤ 0.1
- ASE_PVAL ≤ 0.1
(OUTRIDER entries, if present, are provided as additional context.)

Additional considerations:
- ClinVar burden within the junction region (counts of Pathogenic/Likely pathogenic vs Benign/Likely benign) may indicate the functional importance of that locus.
- Gene constraint: pLI (higher => loss-of-function less tolerated); gnomAD LOF observed/expected (o/e) and upper bound contextualize constraint.
- Inheritance hint: {inh_text}.

Gene constraint (gnomAD):
- pLI: {pli}
- o/e LOF: {oe_lof} (upper: {oe_lof_u}){clingen_block}

Evidence modules for this variant:

FRASER2 (splicing):
{fr_block}

OUTRIDER (expression outlier):
{od_block}

ASE (allele-specific expression):
{ase_block}

Task:
Classify the RNA evidence into exactly ONE of the following Events:
- No Signal
- Exon Skipping
- Intron Retention
- Cryptic Splicing
- Moderately Decreased Expression
- Largely Decreased Expression
- Increased Expression
- Allele Imbalance (Alt High)
- Allele Imbalance (Alt Low)

When deciding the Event:
- For splicing-based events (Exon Skipping, Intron Retention, Cryptic Splicing), base your decision primarily on FRASER2 `pvaluesBetaBinomial_jaccard` together with the direction/magnitude of `delta_jaccard` and supporting patterns in `delta_psi5`/`delta_psi3`, and the raw count context (`rawCountsJnonsplit`, `rawOtherCounts_*`).
- For expression-based events (Moderately/Largely Decreased Expression, Increased Expression), use the magnitude and sign of OUTRIDER_zScore (e.g., |z| ≈ 1.5–3 = moderate, |z| ≥ 3 = large).
- For allele-imbalance events (Alt High / Alt Low), use ASE_PVAL together with ALT_RATIO while explicitly using ALT_RATIO=0.5 as the balance point: ALT_RATIO > 0.5 = Alt High, ALT_RATIO < 0.5 = Alt Low. Remember we only include heterozygous sites.

Then, considering the ClinVar burden in the implicated junctions, the gene constraint (pLI/o-e), and the inheritance tendency noted above, provide an overall confidence level:
- Conclusion ∈ {{No RNA Evidence, Weak RNA Evidence, Strong RNA Evidence}}.

Output exactly three lines in English (no extra commentary):
Reasoning: <succinct rationale grounded in FRASER2/OUTRIDER/ASE values, ClinVar burden, pLI/o-e, and inheritance>
Event: <one label from the list above>
Conclusion: <one of No RNA Evidence | Weak RNA Evidence | Strong RNA Evidence>


Return exactly one line JSON object with keys "Reasoning", "Event" and "Conclusion". No markdown. No code fences. No extra keys.
Reasoning: <succinct rationale grounded in FRASER2/OUTRIDER/ASE values, ClinVar burden, pLI/o-e, and inheritance>
Event: <one label from the Event list above>
Conclusion: must be exactly one of 'No RNA Evidence', 'Weak RNA Evidence', 'Strong RNA Evidence'.

Output schema (must match exactly):{{"Reasoning":"<string>","Event":"<No Signal|Exon Skipping|Intron Retention|Cryptic Splicing|Moderately Decreased Expression|Largely Decreased Expression|Increased Expression|Allele Imbalance (Alt High)|Allele Imbalance (Alt Low)>","Conclusion":"<No RNA Evidence|Weak RNA Evidence|Strong RNA Evidence>"}}
If you cannot comply, output exactly:{{"Reasoning":"comply_error","Event":"No Signal","Conclusion":"No RNA Evidence"}}

"""
    return prompt


# ---------------------------------------------------------------------------
# ClinVar burden
# ---------------------------------------------------------------------------

def clinvar_region_metrics(chrom: str, start: int, end: int, assembly: str, chrom_region_cache, NCBI_EMAIL, NCBI_KEY):
    params = {"db": "clinvar",
              "retmode": "json",
              "tool": "rna_region_ratio"}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_KEY:
        params["api_key"] = NCBI_KEY
    
    EUTILS_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    NCBI_SLEEP_SEC = 0.11 if NCBI_KEY else 0.34
    
    field = {"GRCh38": "chrpos38", "GRCh37": "chrpos37"}.get(assembly, "chrpos38")
    key = (chrom, int(start), int(end), field)
    if key in chrom_region_cache:
        return chrom_region_cache[key]
    base = f'{chrom}[chr] AND {start}:{end}[{field}]'
    tl_p = f'({base}) AND ("clinsig pathogenic"[Properties] OR "clinsig likely pathogenic"[Properties])'
    tl_b = f'({base}) AND ("clinsig benign"[Properties] OR "clinsig likely benign"[Properties])'
    params["term"] = tl_p
    r = requests.get(EUTILS_ESEARCH, params=params, timeout=60)
    p_lp_count = int(r.json().get("esearchresult", {}).get("count", 0))

    time.sleep(NCBI_SLEEP_SEC)
    params["term"] = tl_b
    r = requests.get(EUTILS_ESEARCH, params=params, timeout=60)
    b_lb_count = int(r.json().get("esearchresult", {}).get("count", 0))

    ratio = (p_lp_count / b_lb_count) if b_lb_count > 0 else None
    chrom_region_cache[key] = (p_lp_count, b_lb_count, ratio)
    time.sleep(NCBI_SLEEP_SEC)
    return chrom_region_cache[key]


# ---------------------------------------------------------------------------
# Allele quantification straight from the BAM
# ---------------------------------------------------------------------------

def read_bam(bam_path, bai_path=None):
    """Open a BAM, or return None if it cannot be opened."""
    if bam_path is None or (isinstance(bam_path, float) and pd.isna(bam_path)):
        return None
    try:
        if bai_path is None or (isinstance(bai_path, float) and pd.isna(bai_path)):
            return pysam.AlignmentFile(str(bam_path), "rb")
        return pysam.AlignmentFile(str(bam_path), "rb", index_filename=str(bai_path))
    except (OSError, ValueError):
        return None


def compute_ref_alt_stats_from_bam(bam, chrom, pos_1based, ref, alt):
    ref, alt = ref.upper(), alt.upper()
    pos0 = pos_1based - 1
    nan3 = (np.nan, np.nan, np.nan)

    if len(ref) == 1 and len(alt) == 1:
        counts = {"A": 0, "C": 0, "G": 0, "T": 0}
        try:
            for col in bam.pileup(chrom, pos0, pos0 + 1, truncate=True,
                                  stepper="nofilter", min_mapping_quality=0,
                                  min_base_quality=0):
                if col.reference_pos != pos0:
                    continue
                for pr in col.pileups:
                    if pr.is_del or pr.is_refskip or pr.query_position is None:
                        continue
                    base = pr.alignment.query_sequence[pr.query_position].upper()
                    if base in counts:
                        counts[base] += 1
        except (ValueError, KeyError):
            return nan3, nan3

        ref_cnt = float(counts.get(ref, 0))
        alt_cnt = float(counts.get(alt, 0))
        return (ref_cnt, ref_cnt, ref_cnt), (alt_cnt, alt_cnt, alt_cnt)

    out = []
    for seq in (ref, alt):
        if not seq:
            out.append(nan3)
            continue

        L = len(seq)
        if L == 1:
            cnt = 0
            try:
                for col in bam.pileup(chrom, pos0, pos0 + 1, truncate=True,
                                      stepper="nofilter", min_mapping_quality=0,
                                      min_base_quality=0):
                    if col.reference_pos != pos0:
                        continue
                    for pr in col.pileups:
                        if pr.is_del or pr.is_refskip or pr.query_position is None:
                            continue
                        if pr.alignment.query_sequence[pr.query_position].upper() == seq:
                            cnt += 1
            except (ValueError, KeyError):
                out.append(nan3)
                continue
            cnt = float(cnt)
            out.append((cnt, cnt, cnt))
            continue

        try:
            a, c, g, t = bam.count_coverage(chrom, pos0, pos0 + L,
                                            quality_threshold=0)
        except (ValueError, KeyError):
            out.append(nan3)
            continue

        arr = np.empty(L, dtype=float)
        for i, base in enumerate(seq):
            if base == "A":
                arr[i] = a[i]
            elif base == "C":
                arr[i] = c[i]
            elif base == "G":
                arr[i] = g[i]
            elif base == "T":
                arr[i] = t[i]
            else:
                arr[i] = 0
        out.append((float(arr.max()), float(arr.mean()), float(arr.min())))

    return out[0], out[1]



def _contig_translator(bam):
    """
    Map the candidate table's contig names onto whatever this BAM uses.

    filter_one builds Chromosome as "chr" + the varId's numeric code, so
    candidates are always UCSC-style. A BAM aligned against an Ensembl
    reference names the same chromosomes without the prefix, and pileup on an
    unknown contig raises - an exception the per-site handler turns into NaN,
    so the whole table would come back empty with nothing to explain it.
    Translating costs a set lookup; refusing would throw away good data.

    Contigs the BAM does not actually carry are dropped, so a site on one is
    skipped deliberately rather than queried and silently counted as zero.

    Returns a dict, or None if the BAM matches neither convention.
    """
    refs = set(bam.references)
    names = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]

    if "chr1" in refs:
        mapping = {n: n for n in names}
    elif "1" in refs:
        mapping = {n: n[3:] for n in names}
        mapping["chrM"] = "MT" if "MT" in refs else "M"
    else:
        return None

    return {k: v for k, v in mapping.items() if v in refs}


def _quantify_one(sample_id, cand_path, bam_path, output_path, overwrite):
    """
    Count reads supporting each allele at every candidate variant.

    Returns (sample_id, status, detail) where status is one of 'done',
    'skipped', 'no_bam'.

    Row order is load-bearing and must not change. Integration.py attaches this
    table with pd.concat(axis=1), aligning by position rather than by varId -
    which it has to, because a variant appears once per transcript and a merge
    on varId would multiply rows. Every candidate row therefore produces
    exactly one output row, even when the site cannot be counted.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return sample_id, "skipped", ""

    if not bam_path:
        return sample_id, "no_bam", "no rna_path"
    if not Path(bam_path).exists():
        return sample_id, "no_bam", f"BAM not found: {bam_path}"

    bam = read_bam(bam_path)
    if bam is None:
        return sample_id, "no_bam", f"could not open BAM: {bam_path}"

    # Both of the next two fail the same silent way if left unchecked: pileup
    # raises, the per-site handler swallows it, and every row comes back NaN.
    if not bam.has_index():
        bam.close()
        return sample_id, "no_bam", f"BAM has no index: {bam_path}"

    contigs = _contig_translator(bam)
    if contigs is None:
        bam.close()
        return sample_id, "no_bam", (
            f"BAM contigs match neither UCSC nor Ensembl naming "
            f"(first few: {sorted(bam.references)[:4]})"
        )

    try:
        candidates = pd.read_feather(cand_path)
        head = candidates[["varId", "geneSymbol", "transcript_id"]].reset_index(drop=True)

        nan_row = [np.nan] * 6
        stats = []
        for row in candidates.itertuples(index=False):
            chrom = contigs.get(row.Chromosome)
            if chrom is None:
                # One row out per row in, always - see the note on row order.
                stats.append(nan_row)
                continue
            _, _, ref, alt = row.varId.split("_")
            (rmax, rmean, rmin), (amax, amean, amin) = compute_ref_alt_stats_from_bam(bam, chrom, row.Pos, ref, alt)
            stats.append([rmax, rmean, rmin, amax, amean, amin])
    finally:
        bam.close()

    stats = pd.DataFrame(stats, columns=[
        "ref_count_max", "ref_count_mean", "ref_count_min",
        "alt_count_max", "alt_count_mean", "alt_count_min",
    ])
    out = pd.concat([head, stats], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f".{output_path.name}.tmp"
    out.to_feather(tmp)
    os.replace(tmp, output_path)
    return sample_id, "done", ""


def AlleleQuantification(samplesheet, work_dir, overwrite=False, max_workers=4):
    """
    Count raw read support for the reference and alternate allele of every
    candidate variant, taken directly from the RNA BAM.

    Independent of the language model; Integration.py reads the result to sanity
    check calls against the actual pileup. A sample with no usable BAM is
    skipped rather than failed - Integration already substitutes zeros where
    the file is absent.

    BAM paths come from the samplesheet's rna_path column. Earlier versions
    scanned a directory and matched filenames by substring, which mismatched
    whenever one sample ID was a prefix of another.

    Args:
        samplesheet: must have sampleID, candidates_path and rna_path
        work_dir: pipeline work directory.
        overwrite: recompute samples that already have a result.
        max_workers: samples quantified at once.
    """
    work_dir = Path(work_dir)
    output_root = work_dir / "Agents" / "RNA" / "AlleleQuantification"
    output_root.mkdir(parents=True, exist_ok=True)

    if "candidates_path" not in samplesheet.columns:
        raise ValueError(
            "samplesheet missing 'candidates_path' column. "
            "Run DiagnosticEngine.Candidates first."
        )
    if "rna_path" not in samplesheet.columns:
        print("  no rna_path column; skipping allele quantification")
        return

    jobs = []
    for row in samplesheet.itertuples(index=False):
        cand = getattr(row, "candidates_path", None)
        if not cand or (isinstance(cand, float) and pd.isna(cand)):
            continue
        if not Path(cand).exists():
            continue
        bam = getattr(row, "rna_path", None)
        if isinstance(bam, float) and pd.isna(bam):
            bam = None
        jobs.append((str(row.sampleID), Path(cand), str(bam).strip() if bam else "",
                     output_root / f"{row.sampleID}.feather"))

    if not jobs:
        print("  nothing to quantify")
        return

    tally = {"done": 0, "skipped": 0, "no_bam": 0}
    no_bam = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_quantify_one, *job, overwrite) for job in jobs]
        with tqdm(total=len(futures), desc="Quantifying allele support",
                  unit="sample") as pbar:
            for fut in as_completed(futures):
                try:
                    sample_id, status, detail = fut.result()
                    tally[status] += 1
                    if status == "no_bam":
                        no_bam.append(sample_id)
                except Exception as exc:
                    print(f"\n[ERROR] AlleleQuantification: "
                          f"{type(exc).__name__}: {exc}")
                pbar.update(1)

    print(f"  quantified {tally['done']}, reused {tally['skipped']}, "
          f"no usable BAM {tally['no_bam']}")
    if no_bam:
        print(f"  without a BAM: {sorted(no_bam)}")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _write_json_atomic(path, obj):
    """
    Write via a temporary file and rename.

    Existence of the output is what marks a prompt answered, so a half-written
    file from an interrupted run would otherwise be accepted as complete.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _rna_llm_one(task, llm_config, overwrite):
    """Send one prompt and store the answer. Returns 'done' or 'skipped'."""
    out_path = Path(task["output_path"])
    if out_path.exists() and not overwrite:
        return "skipped"

    payload = {
        "model": llm_config["model_name"],
        "prompt": task["prompt"],
        "stream": False,
        "options": {"temperature": float(llm_config["temperature"])},
    }
    url = f"{llm_config['ollama_url'].rstrip('/')}/api/generate"

    last = ""
    for _ in range(3):
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        last = response.json().get("response", "").strip()
        try:
            obj = json.loads(last)
        except json.JSONDecodeError:
            continue
        if {"Reasoning", "Event", "Conclusion"}.issubset(obj):
            _write_json_atomic(out_path, obj)
            return "done"

    raise RuntimeError(
        f"{task['label']}: no valid JSON after 3 attempts. Last output: {last[:200]}"
    )


def _clingen_block(gene, clingen):
    if str(gene).upper() not in clingen.index:
        return ""
    row = clingen.loc[str(gene).upper()]
    return (f"\n- Haploinsufficiency: {row['HAPLOINSUFFICIENCY']}"
            f"\n- Triplosensitivity: {row['TRIPLOSENSITIVITY']}\n")


def _fmt(x):
    return None if pd.isna(x) else float(f"{float(x):.4g}")


def _outrider_items(rows):
    """Gene-level OUTRIDER values, or None when the gene has no expression call."""
    if pd.isna(rows["Outrider_pValue"].iloc[0]):
        return None
    return {
        "RawCounts": rows["Outrider_rawcounts"].iloc[0],
        "RawCount_ZScore": _fmt(rows["Outrider_RawZscore"].iloc[0]),
        "OUTRIDER_ZScore": _fmt(rows["Outrider_zScore"].iloc[0]),
        "OUTRIDER_pValue": _fmt(rows["Outrider_pValue"].iloc[0]),
    }


def _build_tasks(samplesheet, output_root, clingen, use_ncbi,
                 clinvar_assembly, ncbi_email, ncbi_key, overwrite):
    """
    Turn every sample's candidate table into a flat list of prompts.

    Serial and up front, for two reasons. It is cheap next to the model calls,
    and it is where the ClinVar lookups happen - those share a cache with no
    lock and are bound by NCBI's rate limit, so they cannot move into the
    worker pool with everything else.
    """
    tasks = []
    region_cache = {}
    n_no_rna = 0
    n_reused = 0

    rows = list(samplesheet.itertuples(index=False))
    for row in tqdm(rows, desc="Building prompts", unit="sample"):
        sample_id = str(row.sampleID)
        cand_path = getattr(row, "candidates_path", None)
        if not cand_path or (isinstance(cand_path, float) and pd.isna(cand_path)):
            continue
        cand_path = Path(cand_path)
        if not cand_path.exists():
            continue

        candidates = pd.read_feather(cand_path)
        candidates["Is_Splicing_variant"] = candidates["Fraser_pvaluesBetaBinomial_jaccard"] <= 0.1
        candidates["Is_Splicing_gene"] = candidates["Fraser_GenePvalue"] <= 0.1
        candidates["Is_outlier"] = candidates["Outrider_zScore"].abs() >= 1.5
        candidates["Is_ASE"] = candidates["ASE_PVAL"] <= 0.1
        candidates = candidates[
            candidates["Is_Splicing_variant"] | candidates["Is_Splicing_gene"]
            | candidates["Is_outlier"] | candidates["Is_ASE"]
        ]
        if candidates.empty:
            n_no_rna += 1
            continue

        gene_dir = output_root / "GeneLevelEval" / sample_id
        variant_dir = output_root / "VariantLevelEval" / sample_id

        for gene, gene_rows in candidates.groupby("geneSymbol", dropna=False):
            clingen_text = _clingen_block(gene, clingen)
            constraints = {
                key: gene_rows[key].dropna().iloc[0].item()
                for key in ("gnomadGenePLI", "gnomadGeneOELof", "gnomadGeneOELofUpper")
            }
            inheritance = {"recessive": int(gene_rows["recessive"].min()),
                           "dominant": int(gene_rows["dominant"].min())}

            sig_gene = gene_rows[gene_rows["Is_outlier"] | gene_rows["Is_Splicing_gene"]]
            if not sig_gene.empty:
                out_path = gene_dir / f"{gene}.json"
                if out_path.exists() and not overwrite:
                    n_reused += 1
                else:
                    outrider = _outrider_items(sig_gene)
                    if outrider is not None:
                        tasks.append({
                            "label": f"{sample_id}/{gene} (gene)",
                            "output_path": gene_dir / f"{gene}.json",
                            "prompt": build_prompt_RNA_agent_gene(
                                gene=str(gene),
                                fraser_pvalue_gene=_fmt(sig_gene["Fraser_GenePvalue"].iloc[0]),
                                outrider_items=outrider,
                                gene_constraints=constraints,
                                inheritance_flags=inheritance,
                                clingen_dosage=clingen_text)})

            sig_var = gene_rows[gene_rows["Is_Splicing_variant"] | gene_rows["Is_ASE"]]
            if sig_var.empty:
                continue
            outrider = _outrider_items(sig_var)

            for var in sig_var.itertuples(index=False):
                out_path = variant_dir / f"{gene}_VarId_{var.varId}.json"
                if out_path.exists() and not overwrite:
                    n_reused += 1
                    continue
                p_lp, b_lb, ratio = 0, 0, None
                if use_ncbi:
                    try:
                        p_lp, b_lb, ratio = clinvar_region_metrics(
                            var.Chromosome[3:],
                            int(var.Fraser_junction_start),
                            int(var.Fraser_junction_end),
                            clinvar_assembly, region_cache, ncbi_email, ncbi_key,
                        )
                    except Exception:
                        pass

                fraser_items = {
                    "Chrom": var.Chromosome,
                    "FRASER_junction_start": var.Fraser_junction_start,
                    "FRASER_junction_end": var.Fraser_junction_end,
                    "pvaluesBetaBinomial_jaccard": _fmt(var.Fraser_pvaluesBetaBinomial_jaccard),
                    "psi5": _fmt(var.Fraser_psi5),
                    "psi3": _fmt(var.Fraser_psi3),
                    "rawOtherCounts_psi5": _fmt(var.Fraser_rawOtherCounts_psi5),
                    "rawOtherCounts_psi3": _fmt(var.Fraser_rawOtherCounts_psi3),
                    "rawCountsJnonsplit": _fmt(var.Fraser_rawCountsJnonsplit),
                    "jaccard": _fmt(var.Fraser_jaccard),
                    "rawOtherCounts_jaccard": _fmt(var.Fraser_rawOtherCounts_jaccard),
                    "delta_jaccard": _fmt(var.Fraser_delta_jaccard),
                    "delta_psi5": _fmt(var.Fraser_delta_psi5),
                    "delta_psi3": _fmt(var.Fraser_delta_psi3),
                    "predictedMeans_jaccard": _fmt(var.Fraser_predictedMeans_jaccard),
                    "P_LP_count": p_lp,
                    "B_LB_count": b_lb,
                    "P_LP_to_B_LB_ratio": _fmt(ratio) if ratio is not None else None,
                }

                ase_items = None
                if not pd.isna(var.ASE_PVAL) and var.zyg == 1:
                    ase_items = {
                        "ASE_PVAL": _fmt(var.ASE_PVAL),
                        "REF_COUNT": var.ASE_REF_COUNT,
                        "ALT_COUNT": var.ASE_ALT_COUNT,
                        "ALT_RATIO": _fmt(var.ASE_ALT_RATIO),
                    }

                tasks.append({
                    "label": f"{sample_id}/{gene}/{var.varId}",
                    "output_path": variant_dir / f"{gene}_VarId_{var.varId}.json",
                    "prompt": build_prompt_RNA_agent_variant(
                        gene=gene,
                        fraser_items=fraser_items,
                        outrider_items=outrider,
                        ase_items=ase_items,
                        gene_constraints=constraints,
                        inheritance_flags=inheritance,
                        clingen_dosage=clingen_text,
                    ),
                })

    return tasks, n_no_rna, n_reused

def _get_available_cpus():
    """Cores this process may use; respects cgroups, taskset and SLURM."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1

def RunAgent(samplesheet, work_dir, llm_config, references, ref_ver,
             overwrite=False, use_ncbi=True,
             ncbi_email=None, ncbi_key=None, quant_workers=None):
    """
    Run the RNA Agent over the candidates of every sample.

    Two stages. Allele support is first counted straight from each BAM, since
    Integration.py reads it back to check calls against the actual pileup. The
    model is then asked for a gene-level and a variant-level reading of every
    candidate carrying RNA evidence:

        Agents/RNA/AlleleQuantification/<sampleID>.feather
        Agents/RNA/GeneLevelEval/<sampleID>/<gene>.json
        Agents/RNA/VariantLevelEval/<sampleID>/<gene>_VarId_<varId>.json

    Prompts are collected into one flat list and dispatched through a pool
    rather than looped over sample by sample, and a prompt the model cannot
    answer costs that one gene or variant. Re-running retries only what is
    missing, which matters when each item is a model call.

    A sample whose candidates carry no RNA evidence is skipped quietly - a
    DNA-only patient in a mixed cohort is ordinary.

    Args:
        samplesheet: must have sampleID and candidates_path; rna_bam_path is
            used for allele quantification when present.
        work_dir: pipeline work directory.
        llm_config: dict with model_name, ollama_url, temperature, num_parallel.
        clingen_dosage: path to the ClinGen dosage-sensitivity CSV.
        overwrite: redo work that already has output.
        use_ncbi: look up the ClinVar burden per junction region. Off by
            default; it adds a rate-limited network round trip each.
        clinvar_assembly, ncbi_email, ncbi_key: passed to that lookup.
        quant_workers: samples quantified at once in the first stage.

    Returns:
        samplesheet with an 'rna_agent_root' column, also written to
        <work_dir>/samplesheet_with_paths.csv.
    """
    work_dir = Path(work_dir)
    assembly = {"hg38": "GRCh38", "hg19": "GRCh37"}.get(str(ref_ver).lower())
    if assembly is None:
        raise ValueError(
            f"Unsupported ref_ver: {ref_ver}. Expected 'hg38' or 'hg19'."
        )

    if "candidates_path" not in samplesheet.columns:
        raise ValueError(
            "samplesheet missing 'candidates_path' column. "
            "Run DiagnosticEngine.Candidates first."
        )

    if quant_workers is None:
        # Pileup touches one coordinate at a time, so memory stays flat and the
        # limit is random-read IOPS rather than RAM - unlike FRASER's counting,
        # where concurrency had to be sized against the largest BAM.
        quant_workers = max(1, min(len(samplesheet), _get_available_cpus(), 8))

    output_root = work_dir / "Agents" / "RNA"
    output_root.mkdir(parents=True, exist_ok=True)

    print("1. BAM-based Allele Support Quantification ...")
    AlleleQuantification(samplesheet, work_dir, overwrite=overwrite,
                         max_workers=quant_workers)

    print("2. Loading ClinGen Data ...")
    clingen_path = Path(references.rarecollab.clingen_dosage)
    if not clingen_path.exists():
        raise FileNotFoundError(f"ClinGen dosage file not found: {clingen_path}")
    clingen_raw = pd.read_csv(clingen_path)
    required = ["GENE SYMBOL", "HAPLOINSUFFICIENCY", "TRIPLOSENSITIVITY"]
    missing = [c for c in required if c not in clingen_raw.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} NOT found in {clingen_path}")
    clingen = clingen_raw[required].set_index("GENE SYMBOL")

    print("3. Generating prompts ...")
    tasks, n_no_rna, n_reused = _build_tasks(
        samplesheet, output_root, clingen,
        use_ncbi, assembly, ncbi_email, ncbi_key, overwrite
    )
    n_gene = sum(1 for t in tasks if t["label"].endswith("(gene)"))
    print(f"   {len(tasks)} prompt(s): {n_gene} gene-level, "
          f"{len(tasks) - n_gene} variant-level")
    if n_reused:
        print(f"   {n_reused} already answered; pass overwrite=True to redo them")
    if n_no_rna:
        print(f"   {n_no_rna} sample(s) had no RNA evidence among their candidates")

    if not tasks:
        print("Nothing left for the RNA Agent to evaluate.")
        samplesheet = samplesheet.copy()
        samplesheet["rna_agent_root"] = str(output_root)
        return samplesheet

    workers = llm_config.get("num_parallel", 1)
    print(f"4. Evaluating RNA evidence (model={llm_config['model_name']}) ...")
    print(f"   LLM workers: {workers}")

    done = skipped = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_rna_llm_one, task, llm_config, overwrite)
                   for task in tasks]
        with tqdm(total=len(futures), desc="Assessing RNA Evidence") as pbar:
            for fut in as_completed(futures):
                try:
                    # One unparseable answer costs one item. Raising here would
                    # discard every other answer in flight, and each is a model
                    # call that has to be paid for again.
                    if fut.result() == "skipped":
                        skipped += 1
                    else:
                        done += 1
                except Exception as exc:
                    fail += 1
                    print(f"\n[ERROR] RNA LLM: {type(exc).__name__}: {exc}")
                pbar.update(1)
                pbar.set_postfix(done=done, reused=skipped, failed=fail)

    if fail:
        print(f"\nWARNING: {fail} prompt(s) produced no usable answer; "
              f"re-running will retry only those.")

    samplesheet = samplesheet.copy()
    samplesheet["rna_agent_root"] = str(output_root)

    samplesheet_path = work_dir / "samplesheet_with_paths.csv"
    tmp = work_dir / ".samplesheet_with_paths.csv.tmp"
    samplesheet.to_csv(tmp, index=False)
    os.replace(tmp, samplesheet_path)
    print(f"Updated samplesheet: {samplesheet_path}")

    print("--RNA Agent DONE--\n")
    return samplesheet