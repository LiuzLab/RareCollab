#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import json
import requests
import time
import pysam

from pathlib import Path
from collections import defaultdict
from tqdm.notebook import tqdm

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
        f"- pValueGene: {fraser_pvalue_gene} (smaller values indicate stronger evidence of any aberrant splicing across this gene; "
        "≈0.01–0.1 = moderate, ≤0.01 = strong, >0.1 = usually weak or no signal)."
        if fraser_pvalue_gene is not None
        else "(no FRASER gene-level pValueGene available for this gene)."
    )
    od_block = "\n".join(od_lines) if od_lines else "(no OUTRIDER signals retained)"

    prompt = f"""You are an RNA-seq interpretation assistant.

Gene: {gene}

Context:
- FRASER2 aggregates junction-level splicing signals into a gene-level p-value (`pValueGene`). Smaller `pValueGene` means stronger evidence for any aberrant splicing in this gene as a whole. As a rough guide, pValueGene ≤ 0.01 is strong evidence, 0.01–0.1 is moderate, and > 0.1 is usually weak or no evidence.
- OUTRIDER identifies expression outliers at the gene level using z-scores; large negative z-scores (e.g., ≤ -1.5) suggest decreased expression, while large positive z-scores (e.g., ≥ +1.5) suggest increased expression. Larger |z| indicates stronger expression outlier (e.g., |z| ≈ 1.5–3 moderate, |z| ≥ 3 large).
- ASE (allele-specific expression) tests imbalance between the two alleles, but ASE results are not provided for this gene-level summary.

This gene-level summary is generated when at least one of the following is present for this gene:
- FRASER2 gene-level evidence: pValueGene ≤ 0.1
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
- For splicing-based events (Abnormal Splicing), rely on whether pValueGene is small enough to indicate genuine splicing outliers in this gene. Note that pValueGene does not specify the exact exon/intron pattern, only that splicing is abnormal somewhere in the gene.
- For expression-based events (Moderately/Largely Decreased Expression, Increased Expression), use the magnitude and sign of OUTRIDER_zScore (e.g., |z| ≈ 1.5–3 = moderate, |z| ≥ 3 = large).

Then, considering the FRASER2 gene-level pValueGene, the OUTRIDER gene-level outlier signal, the gene constraint (pLI/o-e), and the inheritance tendency noted above, provide an overall confidence level:
- Conclusion ∈ {{No RNA Evidence, Weak RNA Evidence, Strong RNA Evidence}}.

Return exactly one line JSON object with keys "Reasoning", "Event" and "Conclusion". No markdown. No code fences. No extra keys.
Reasoning: <succinct rationale grounded in FRASER2 pValueGene, OUTRIDER values, gene constraint, and inheritance>
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

def clinvar_region_metrics(chrom: str, start: int, end: int, assembly: str, chrom_region_cache, NCBI_EMAIL, NCBI_KEY):
    params = {"db": "clinvar", 
              "retmode": "json",
              "tool": "rna_region_ratio"}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_KEY:
        params["api_key"] = NCBI_KEY
    EUTILS_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    NCBI_SLEEP_SEC = 0.34
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
    return chrom_region_cache[key]

def read_bam(bam_path: str, bai_path: str):
    if pd.isna(bam_path):
        return None
    bam_path = str(bam_path)
    try:
        if pd.isna(bai_path):
            bam = pysam.AlignmentFile(bam_path, "rb")
        else:
            bam = pysam.AlignmentFile(bam_path, "rb", index_filename=str(bai_path))
    except OSError:
        return None
    return bam

def compute_ref_alt_stats_from_bam(bam, chrom, pos_1based, ref, alt):
    ref, alt = ref.upper(), alt.upper()
    pos0 = pos_1based - 1
    nan3 = (np.nan, np.nan, np.nan)

    # SNP: 一次 pileup 同时算 ref / alt
    if len(ref) == 1 and len(alt) == 1:
        counts = {"A": 0, "C": 0, "G": 0, "T": 0}
        try:
            for col in bam.pileup(
                chrom,
                pos0,
                pos0 + 1,
                truncate=True,
                stepper="nofilter",
                min_mapping_quality=0,
                min_base_quality=0,
            ):
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

    # 非 SNP: 分别算 ref / alt，但不再调用额外函数
    out = []
    for seq in (ref, alt):
        if not seq:
            out.append(nan3)
            continue

        L = len(seq)

        if L == 1:
            cnt = 0
            try:
                for col in bam.pileup(
                    chrom,
                    pos0,
                    pos0 + 1,
                    truncate=True,
                    stepper="nofilter",
                    min_mapping_quality=0,
                    min_base_quality=0,
                ):
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
            a, c, g, t = bam.count_coverage(
                chrom,
                pos0,
                pos0 + L,
                quality_threshold=0,
            )
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

def AlleleQuantification(work_path, BAM_root_path):
    input_path = work_path + '/Diagnostic_results/Candidates/'
    if not Path(input_path).exists():
        raise ValueError(f"Input file NOT detected, please run 'DiagnosticEngine.Candidates' first and check the work_path ... ")

    #Global variables:
    Cand_Path = {p.name.split('_nomcand.feather')[0]:p for p in Path(input_path).iterdir()}
    BAM_Path = defaultdict(dict)
    cand_keys = list(Cand_Path.keys())
    for p in Path(BAM_root_path).iterdir():
        matched_keys = [k for k in cand_keys if k in p.name]
        if len(matched_keys) != 1:
            print(f"{p.name} has 0 or 2+ matched files... skip")
            continue

        sample_id = matched_keys[0]

        if p.name.endswith('.bam'):
            BAM_Path[sample_id]['BAM'] = p
        elif p.name.endswith('.bai'):
            BAM_Path[sample_id]['BAI'] = p

    BAM_Path = dict(BAM_Path)

    output_path = work_path + '/Agents/RNA/AlleleQuantification/'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    #BAM-based Allele Support Quantification：
    pbar = tqdm(Cand_Path.items(), desc="BAM-based Allele Support Quantification", total=len(Cand_Path))
    for sample_id, sample_path in pbar:
        bam_path = BAM_Path[sample_id]['BAM']
        bai_path = BAM_Path[sample_id]['BAI']
        data = pd.read_feather(sample_path)
        bam = read_bam(bam_path, bai_path)
        output_df = data[['varId','geneSymbol','transcript_id']].reset_index(drop = True)
        save_path = output_path + sample_id + '.feather'
        if bam is None:
            continue
        else:
            sub_res = []
            for row in data.itertuples(index=True):
                chrom = row.Chromosome
                pos = row.Pos
                _, _, ref, alt = row.varId.split('_')
                (ref_max, ref_mean, ref_min), (alt_max, alt_mean, alt_min) = compute_ref_alt_stats_from_bam(bam, chrom, pos, ref, alt)
                sub_res.append([ref_max, ref_mean, ref_min, alt_max, alt_mean, alt_min])
            sub_res = pd.DataFrame(sub_res, columns = ['ref_count_max','ref_count_mean','ref_count_min','alt_count_max','alt_count_mean','alt_count_min'])
            output_df = pd.concat([output_df, sub_res], axis = 1)
            output_df.to_feather(save_path)


def RunAgent(work_path, MODEL_NAME, OLLAMA_URL, TEMPERATURE, CLINGEN_DOSAGE, UseNCBI = False, CLINVAR_ASSEMBLY = "GRCh38",NCBI_EMAIL = None, NCBI_KEY = None):
    #1. Confirm data (Candidate list):
    input_path = work_path + '/Diagnostic_results/Candidates/'
    if not Path(input_path).exists():
        raise ValueError(f"Input file NOT detected, please run 'DiagnosticEngine.Candidates' first and check the work_path ... ")

    #Global variables:
    Cand_Path = {p.name.split('_nomcand.feather')[0]:p for p in Path(input_path).iterdir()}
    chrom_region_cache = defaultdict(tuple)
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"

    #2. Load ClinGen Data:
    print('Loading ClinGen Data ...')
    CLINGEN_raw = pd.read_csv(CLINGEN_DOSAGE)
    cols = CLINGEN_raw.columns
    if 'GENE SYMBOL' not in cols:
        raise ValueError(f"Column id:'GENE SYMBOL' NOT Found in {CLINGEN_DOSAGE}")
    if 'HAPLOINSUFFICIENCY' not in cols:
        raise ValueError(f"Column id:'HAPLOINSUFFICIENCY' NOT Found in {CLINGEN_DOSAGE}")
    if 'TRIPLOSENSITIVITY' not in cols:
        raise ValueError(f"Column id:'TRIPLOSENSITIVITY' NOT Found in {CLINGEN_DOSAGE}")
    CLINGEN = CLINGEN_raw[['GENE SYMBOL','HAPLOINSUFFICIENCY','TRIPLOSENSITIVITY']]
    CLINGEN = CLINGEN.set_index('GENE SYMBOL')

    output_root = work_path + '/Agents/RNA/'
    Path(output_root).mkdir(parents=True, exist_ok=True)

    #Curate Prompts and LLM Reviewing ...
    print(f"Generating Prompts and Evaluating Variants by RNA Agent ...")
    pbar = tqdm(Cand_Path.items(), desc="Accessing RNA Evidence", total=len(Cand_Path))
    for sample_id, sample_path in pbar:
        data = pd.read_feather(sample_path)
        output_genelevel = output_root + 'GeneLevelEval/' + sample_id
        output_variantlevel = output_root + 'VariantLevelEval/' + sample_id
        Path(output_genelevel).mkdir(parents=True, exist_ok=True)
        Path(output_variantlevel).mkdir(parents=True, exist_ok=True)
        data['Is_Splicing_variant'] = data['Fraser_pvaluesBetaBinomial_jaccard'] <= 0.1
        data['Is_Splicing_gene'] = data['Fraser_GenePvalue'] <= 0.1
        data['Is_outlier'] = data['Outrider_zScore'].abs() >= 1.5
        data['Is_ASE'] = data['ASE_PVAL'] <= 0.1
        data['RNA_any'] = data['Is_Splicing_variant'] | data['Is_Splicing_gene'] | data['Is_outlier'] | data['Is_ASE']
        data = data[data['RNA_any']]
        gene_grouped = data.groupby("geneSymbol", dropna=False)
        g_i, total_genes = 1, len(gene_grouped)

        for gene, subgene_df in gene_grouped:
            subgene_df = subgene_df.copy()
            if gene.upper() in CLINGEN.index:
                hi, ts = CLINGEN.loc[gene.upper(),'HAPLOINSUFFICIENCY'], CLINGEN.loc[gene.upper(),'TRIPLOSENSITIVITY']
                clingen_dosage = f"\n- Haploinsufficiency: {hi}\n- Triplosensitivity: {ts}" + "\n"
            else:
                clingen_dosage = ""

            # Gene-level constraints & inheritance flags
            constraints = {"gnomadGenePLI": subgene_df["gnomadGenePLI"].dropna().iloc[0].item(), 
                        "gnomadGeneOELof": subgene_df["gnomadGeneOELof"].dropna().iloc[0].item(),
                        "gnomadGeneOELofUpper": subgene_df["gnomadGeneOELofUpper"].dropna().iloc[0].item()}
            has_recessive = int(min(subgene_df["recessive"]))
            has_dominant  = int(min(subgene_df["dominant"]))
            inheritance_flags = {"recessive": has_recessive, "dominant": has_dominant}
            #RNA Significant Gene Rows:
            sig_gene_rows = subgene_df[subgene_df["Is_outlier"] | subgene_df["Is_Splicing_gene"]]
            if not sig_gene_rows.empty:
                pbar.set_postfix(GeneLevel = f"{gene} ({g_i}/{total_genes})",VariantLevel = f"Pending")
                json_output_gene = output_genelevel + "/" + gene + '.json'
                if not Path(json_output_gene).exists():
                    fraser_GenePvalue = float(f"{sig_gene_rows['Fraser_GenePvalue'].iloc[0].item():.4g}")
                    outrider_items = {"RawCounts": sig_gene_rows['Outrider_rawcounts'].iloc[0],
                                    "RawCount_ZScore": float(f"{sig_gene_rows['Outrider_RawZscore'].iloc[0].item():.4g}"),
                                    "OUTRIDER_ZScore": float(f"{sig_gene_rows['Outrider_zScore'].iloc[0].item():.4g}"),
                                    "OUTRIDER_pValue": float(f"{sig_gene_rows['Outrider_pValue'].iloc[0].item():.4g}")}
                    
                    prompt_text_gene = build_prompt_RNA_agent_gene(gene = str(gene),
                                                                fraser_pvalue_gene = fraser_GenePvalue,
                                                                outrider_items = outrider_items,
                                                                gene_constraints = constraints,
                                                                inheritance_flags = inheritance_flags,
                                                                clingen_dosage = clingen_dosage)
                    payload = {"model": MODEL_NAME,
                            "prompt": prompt_text_gene,
                            "stream": False,
                            "options": {"temperature": float(TEMPERATURE)}}
                    obj = None
                    for attemp in range(3):
                        r = requests.post(url, json=payload, timeout=600)
                        r.raise_for_status()
                        data = r.json()
                        llm_output = data.get("response", "").strip()
                        try:
                            obj = json.loads(llm_output)
                            if not {"Reasoning", "Event", "Conclusion"}.issubset(obj):
                                continue
                            break
                        except Exception:
                            continue
                    if obj is None:
                        raise RuntimeError(f"Failed to parse tsv file after 3 tries. Last output: {llm_output}")
                    with Path(json_output_gene).open("w", encoding="utf-8") as f: json.dump(obj, f)
            #RNA Significant Variant rows:
            sig_var_rows = subgene_df[subgene_df["Is_Splicing_variant"] | subgene_df["Is_ASE"]]
            if not sig_var_rows.empty:
                #Gene level outrider is the same to all variants:
                if f"{sig_var_rows['Outrider_pValue'].iloc[0].item():.4g}" != 'nan':
                    outrider_items = {"RawCounts": sig_var_rows['Outrider_rawcounts'].iloc[0],
                                    "RawCount_ZScore": float(f"{sig_var_rows['Outrider_RawZscore'].iloc[0].item():.4g}"),
                                    "OUTRIDER_ZScore": float(f"{sig_var_rows['Outrider_zScore'].iloc[0].item():.4g}"),
                                    "OUTRIDER_pValue": float(f"{sig_var_rows['Outrider_pValue'].iloc[0].item():.4g}")}
                else:
                    outrider_items = None
                v_i, total_v = 1, len(sig_var_rows)
                for var_row in sig_var_rows.itertuples(index=False):
                    pbar.set_postfix(GeneLevel = f"{gene} ({g_i}/{total_genes})",VariantLevel = f"{v_i}/{total_v}")
                    json_output_variant = output_variantlevel + "/" + gene + "_VarId_" + var_row.varId + '.json'
                    v_i += 1
                    if not Path(json_output_variant).exists():
                        if UseNCBI:
                            try:
                                p_lp, b_lb, ratio = clinvar_region_metrics(var_row.Chromosome[3:], 
                                                                        int(var_row.Fraser_junction_start),
                                                                        int(var_row.Fraser_junction_end),
                                                                        CLINVAR_ASSEMBLY,
                                                                        chrom_region_cache,
                                                                        NCBI_EMAIL,
                                                                        NCBI_KEY)
                            except Exception:
                                p_lp, b_lb, ratio = 0, 0, None
                        else:
                            p_lp, b_lb, ratio = 0, 0, None
                        fraser_items = {
                            "Chrom": var_row.Chromosome,
                            "FRASER_junction_start": var_row.Fraser_junction_start,
                            "FRASER_junction_end": var_row.Fraser_junction_end,
                            "pvaluesBetaBinomial_jaccard": float(f"{var_row.Fraser_pvaluesBetaBinomial_jaccard:.4g}"),
                            "psi5": float(f"{var_row.Fraser_psi5:.4g}"),
                            "psi3": float(f"{var_row.Fraser_psi3:.4g}"),
                            "rawOtherCounts_psi5": float(f"{var_row.Fraser_rawOtherCounts_psi5:.4g}"),
                            "rawOtherCounts_psi3": float(f"{var_row.Fraser_rawOtherCounts_psi3:.4g}"),
                            "rawCountsJnonsplit": float(f"{var_row.Fraser_rawCountsJnonsplit:.4g}"),
                            "jaccard": float(f"{var_row.Fraser_jaccard:.4g}"),
                            "rawOtherCounts_jaccard": float(f"{var_row.Fraser_rawOtherCounts_jaccard:.4g}"),
                            "delta_jaccard": float(f"{var_row.Fraser_delta_jaccard:.4g}"),
                            "delta_psi5": float(f"{var_row.Fraser_delta_psi5:.4g}"),
                            "delta_psi3": float(f"{var_row.Fraser_delta_psi3:.4g}"),
                            "predictedMeans_jaccard": float(f"{var_row.Fraser_predictedMeans_jaccard:.4g}"),
                            "P_LP_count": p_lp,
                            "B_LB_count": b_lb,
                            "P_LP_to_B_LB_ratio": float(f"{ratio:.4g}") if ratio != None else None
                        }
                        if f"{var_row.ASE_PVAL:.4g}" != 'nan' and var_row.zyg == 1:
                            ase_items = {
                                "ASE_PVAL": float(f"{var_row.ASE_PVAL:.4g}"),
                                "REF_COUNT": var_row.ASE_REF_COUNT,
                                "ALT_COUNT": var_row.ASE_ALT_COUNT,
                                "ALT_RATIO": float(f"{var_row.ASE_ALT_RATIO}")}
                        else:
                            ase_items = None
                        
                        prompt_text_variant = build_prompt_RNA_agent_variant(gene = gene,
                                                                            fraser_items=fraser_items,
                                                                            outrider_items=outrider_items,
                                                                            ase_items=ase_items,
                                                                            gene_constraints=constraints,
                                                                            inheritance_flags=inheritance_flags,
                                                                            clingen_dosage=clingen_dosage)
                        payload = {"model": MODEL_NAME,
                                "prompt": prompt_text_variant,
                                "stream": False,
                                "options": {"temperature": float(TEMPERATURE)}}
                        obj = None
                        for attemp in range(3):
                            r = requests.post(url, json=payload, timeout=600)
                            r.raise_for_status()
                            data = r.json()
                            llm_output = data.get("response", "").strip()
                            try:
                                obj = json.loads(llm_output)
                                if not {"Reasoning", "Event", "Conclusion"}.issubset(obj):
                                    continue
                                break
                            except Exception:
                                continue
                        if obj is None:
                            raise RuntimeError(f"Failed to parse tsv file after 3 tries. Last output: {llm_output}")
                        with Path(json_output_variant).open("w", encoding="utf-8") as f: json.dump(obj, f)
            g_i += 1
    print(f"--RNA Agent's Work is DONE--")
