#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import pandas as pd
import pyranges as pr
import torch
import torch.nn as nn
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

#Default Key Parameters:
LABEL_COL = "is_causal"
ID_COL = "identifier"
VAR_COL = "varId"
BATCH_SIZE = 2**20

class DomainExpert(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), 64),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.head = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        h = self.net(x)
        logit = self.head(h).squeeze(-1)
        return h, logit

class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        d = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(d, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = int(h)
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        h = self.mlp(x)
        return self.out(h).squeeze(-1)
    
class DomainMoE(nn.Module):
    def __init__(self, domain_input: dict[str, int], fusion_hidden_dims, expert_embed_dim, use_layer_norm, expert_dropout, fusion_dropout):
        super().__init__()
        self.domain_input = domain_input
        self.fusion_hidden_dims = fusion_hidden_dims
        self.expert_embed_dim = expert_embed_dim
        self.use_layer_norm = use_layer_norm
        self.expert_dropout = expert_dropout
        self.fusion_dropout = fusion_dropout
        self.num_domains = len(domain_input)

        self.experts = nn.ModuleDict({key: DomainExpert(self.domain_input[key], dropout=self.expert_dropout, embed_dim=self.expert_embed_dim) for key in self.domain_input.keys()})

        if self.use_layer_norm:
            self.domain_lns = nn.ModuleDict({key: nn.LayerNorm(self.expert_embed_dim) for key in self.domain_input.keys()})
            self.fusion_ln = nn.LayerNorm(self.expert_embed_dim * self.num_domains)
        else:
            self.domain_lns = nn.ModuleDict({key: nn.Identity() for key in self.domain_input.keys()})
            self.fusion_ln = nn.Identity()

        fusion_in_dim = self.expert_embed_dim * self.num_domains
        self.fusion = FusionMLP(fusion_in_dim, self.fusion_hidden_dims, dropout=self.fusion_dropout)

    def forward(self, inputs: dict[str, torch.Tensor]):
        embeddings = []
        domain_logits_list = []
        for name in self.domain_input.keys():
            h, logit = self.experts[name](inputs[name])
            h = self.domain_lns[name](h)
            embeddings.append(h)
            domain_logits_list.append(logit.unsqueeze(1))

        fusion_in = torch.cat(embeddings, dim=1)
        fusion_in = self.fusion_ln(fusion_in)
        overall_logit = self.fusion(fusion_in)
        domain_logits = torch.cat(domain_logits_list, dim=1)

        return {"overall_logit": overall_logit, "domain_logits": domain_logits}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(df, label_col, domain_preprocs_states):
    """
    Build batch tensors by slicing df on indices and applying ckpt preprocessors.
    Avoids storing huge domain_X in RAM.
    """
    y_arr = df[label_col].astype(np.float32).to_numpy()
    
    def transform(df, domain_preprocs_state):
        n = len(df)
        outs = []
        for i in range(len(domain_preprocs_state['fitted'])):
            col_name = domain_preprocs_state['fitted'][i]['name']
            col_kind = domain_preprocs_state['fitted'][i]['kind']
            col_median = domain_preprocs_state['fitted'][i]['median']
            col_std = domain_preprocs_state['fitted'][i]['std']
            col_categories = domain_preprocs_state['fitted'][i]['categories']

            if col_name in df.columns:
                #If the column is found, take it:
                s = df[col_name]
                if col_kind == 'num':
                    #If it's numeric data, normalize it:
                    vals = pd.to_numeric(s, errors="coerce").fillna(col_median).values.astype(np.float32, copy=False)
                    vals = (vals - np.float32(col_median)) / np.float32(col_std)
                    outs.append(vals.reshape(-1,1))
                    continue
                elif len(col_categories) > 0: #..obsolete..
                    #If it's categorical data, one-hot-encode the feather:
                    base = s.astype("string").fillna("")
                    cat_to_idx = {c: i for i, c in enumerate(col_categories)}
                    # vectorized one-hot
                    idx = base.map(lambda x: cat_to_idx.get(str(x), -1)).to_numpy()
                    onehot = np.zeros((n, len(col_categories)), dtype=np.float32)
                    valid = idx >= 0
                    rows = np.nonzero(valid)[0]
                    cols = idx[valid].astype(np.int64, copy=False)
                    onehot[rows, cols] = 1.0
                    outs.append(onehot)
                    continue
            #Any other cases, fill-in 0s:
            print(f"{col_name} column not found")             
            outs.append(np.zeros((n, 0), dtype=np.float32))

        return np.concatenate(outs, axis=1).astype(np.float32, copy=False)

    def _collate(idxs):
        idxs_np = np.asarray(idxs, dtype=np.int64)
        batch_df = df.iloc[idxs_np]

        out = {}
        out["idx"] = torch.from_numpy(idxs_np)
        out["y"] = torch.from_numpy(y_arr[idxs_np]).view(-1)
        for name in domain_preprocs_states.keys():
            x = transform(batch_df, domain_preprocs_states[name])  # [B, in_dim]
            out[f"{name}"] = torch.from_numpy(x)

        return out

    return _collate

def MoE(samplesheet, work_dir, references=None, overwrite=False,
        RANDOM_SEED=42, RANK_ON_LOGIT=True):
    """
    Run DNA diagnostic MoE model on each sample's vartogene.feather.
    
    Reads samplesheet["vartogene_feather_path"] column for input,
    writes <sample>_MoE_scores.feather to Diagnostic_results/DNA_MoE/out/.
    
    Args:
        samplesheet: DataFrame with sampleID + vartogene_feather_path columns.
        work_dir: Base work directory.
        references: Optional, for future use (model path could come from here).
        overwrite: Re-run even if output exists.
    """
    work_dir = Path(work_dir)
    print(f'Creating work dir: ~/Diagnostic_results/DNA_MoE/')
    OUT_DIR = work_dir / "Diagnostic_results" / "DNA_MoE"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    MODEL_PATH = references.rarecollab.moe_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(RANDOM_SEED)

    # Load MoE model
    print(f'Loading MoE Model ...')
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    domain_names = ckpt["domain_names"]
    domain_input = {k: ckpt["domain_preprocs_state"][k]["out_dim"] for k in domain_names}

    print(f"Domains Experts: {domain_names}")
    model = DomainMoE(
        domain_input=domain_input,
        fusion_hidden_dims=ckpt['fusion_hidden_dims'],
        expert_embed_dim=ckpt['expert_embed_dim'],
        use_layer_norm=ckpt['use_layer_norm'],
        expert_dropout=ckpt['expert_dropout'],
        fusion_dropout=ckpt['fusion_dropout'],
    )
    model = model.to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[Model] Loaded. expert_embed_dim={ckpt['expert_embed_dim']}, "
              f"fusion_hidden_dims={ckpt['fusion_hidden_dims']}, "
              f"use_layer_norm={ckpt['use_layer_norm']}")
    except Exception as e:
        print("[ERROR] load_state_dict(strict=True) failed.")
        print(f"  - domain_names={domain_names}")
        print(f"  - expert_embed_dim={ckpt['expert_embed_dim']}, "
              f"fusion_hidden_dims={ckpt['fusion_hidden_dims']}, "
              f"use_layer_norm={ckpt['use_layer_norm']}")
        print(f"  - expert_dropout={ckpt['expert_dropout']}, "
              f"fusion_dropout={ckpt['fusion_dropout']}")
        raise e
    model.eval()

    # Build SamplePath from samplesheet (not from glob)
    if "vartogene_feather_path" not in samplesheet.columns:
        raise ValueError(
            "samplesheet missing 'vartogene_feather_path' column. "
            "Run GENERATE_SINGLETON_FEATURES first."
        )
    SampleIDs_Path = {
        row.sampleID: Path(row.vartogene_feather_path)
        for row in samplesheet.itertuples(index=False)
    }
    
    output_paths = {}

    # Evaluate samples
    for sample_id, sample_path in tqdm(SampleIDs_Path.items(),
                                        desc="MoE on Samples",
                                        total=len(SampleIDs_Path)):
        output_path = OUT_DIR / f"{sample_id}_MoE_scores.feather"
        
        # Cache hit: skip if output exists and not overwrite
        if output_path.exists() and not overwrite:
            output_paths[sample_id] = str(output_path)
            continue
        
        if not sample_path.exists():
            print(f"[WARN] vartogene feather not found for {sample_id}: {sample_path}")
            continue
        
        processed_data = pd.read_feather(sample_path)
        N = len(processed_data)

        loader = DataLoader(
            list(range(N)),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn(processed_data, LABEL_COL, ckpt["domain_preprocs_state"]),
            pin_memory=(device == "cuda"),
            persistent_workers=True,
        )

        # Preallocate outputs
        overall_logit = np.empty(N, dtype=np.float32)
        overall_prob = np.empty(N, dtype=np.float32)
        domain_scores = {name: np.empty(N, dtype=np.float32) for name in domain_names}

        # Forward
        with torch.no_grad():
            for batch in loader:
                idx = batch["idx"].numpy()
                inputs = {name: batch[f"{name}"].to(device, non_blocking=True)
                          for name in domain_names}

                with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                    out = model(inputs)
                    logit_t = out["overall_logit"]
                    prob_t = torch.sigmoid(logit_t)
                    dom_t = out["domain_logits"]

                overall_logit[idx] = logit_t.detach().cpu().numpy().astype(np.float32, copy=False)
                overall_prob[idx] = prob_t.detach().cpu().numpy().astype(np.float32, copy=False)

                dom_np = dom_t.detach().cpu().numpy().astype(np.float32, copy=False)
                for j, name in enumerate(domain_names):
                    domain_scores[name][idx] = dom_np[:, j]

        # Organize output
        output = processed_data[[
            'varId', 'identifier', 'geneSymbol', 'dominant', 'recessive', 'zyg',
            'HGVSc', 'cons_frameshift_variant', 'transcript_id', 'HGVSc_core',
            'HGVSp', 'transcript_score', 'gnomadGenePLI', 'gnomadGeneOELof',
            'gnomadGeneOELofUpper', 'is_causal',
        ]].copy()
        output["overall_logit"] = overall_logit
        output["overall_prob"] = overall_prob
        for name in domain_names:
            output[f"score_{name}"] = domain_scores[name]
            output[f"rank_{name}"] = output["varId"].map(
                output.groupby("varId", dropna=False)[f"score_{name}"]
                .max().rank(method='max', ascending=False).astype(int)
            )
        if RANK_ON_LOGIT:
            output['Diagnostic_Engine_Rank'] = output["varId"].map(
                output.groupby("varId", dropna=False)["overall_logit"]
                .max().rank(method='max', ascending=False).astype(int)
            )
        else:
            output['Diagnostic_Engine_Rank'] = output["varId"].map(
                output.groupby("varId", dropna=False)["overall_prob"]
                .max().rank(method='max', ascending=False).astype(int)
            )

        # Atomic write
        tmp = OUT_DIR / f".{sample_id}_MoE_scores.feather.tmp"
        output.to_feather(tmp)
        os.replace(tmp, output_path)
        output_paths[sample_id] = str(output_path)

    print(f"--DNA Diagnostic Engine DONE--\n")
    # Update samplesheet with output paths
    samplesheet = samplesheet.copy()
    samplesheet["moe_score_path"] = samplesheet["sampleID"].map(output_paths)
    
    # Save updated samplesheet (atomic write)
    samplesheet_path = Path(work_dir) / "samplesheet_with_paths.csv"
    tmp = Path(work_dir) / ".samplesheet_with_paths.csv.tmp"
    samplesheet.to_csv(tmp, index=False)
    os.replace(tmp, samplesheet_path)
    print(f"Updated samplesheet: {samplesheet_path}")
    
    return samplesheet

def _join_rule_labels(row_bool: pd.Series) -> str:
    names = [name for name, hit in row_bool.items() if bool(hit)]
    return ", ".join(names)


def filter_one(sample_id,  moe_path, expression_path, splicing_path, ase_path,
               output_path, overwrite=False):
    if output_path.exists() and not overwrite:
        return {
            "has_expression": expression_path is not None and expression_path.exists(),
            "has_splicing": splicing_path is not None and splicing_path.exists(),
            "has_ase": ase_path is not None and ase_path.exists(),
            "reused": True,
        }
    # Load DNA MoE scores
    data = pd.read_feather(moe_path)

    # Expression
    if expression_path is not None and expression_path.exists():
        expression_data = pd.read_feather(expression_path)
        expression_data = expression_data.rename(columns={
            'GeneSymbol': 'geneSymbol', 'pValue': 'Outrider_pValue',
            'padjust': 'Outrider_padjust', 'zScore': 'Outrider_zScore',
            'l2fc': 'Outrider_l2f', 'rawcounts': 'Outrider_rawcounts',
        })
        if 'RawZscore' in expression_data.columns:
            expression_data = expression_data.rename(columns={'RawZscore': 'Outrider_RawZscore'})
        else:
            expression_data['Outrider_RawZscore'] = np.nan
        has_RNA_expression = True
    else:
        expression_data = pd.DataFrame({
            'geneSymbol': ['None'], 'Outrider_pValue': np.nan,
            'Outrider_padjust': np.nan, 'Outrider_zScore': np.nan,
            'Outrider_l2f': np.nan, 'Outrider_rawcounts': np.nan,
            'Outrider_RawZscore': np.nan,
        })
        has_RNA_expression = False
    data = data.merge(expression_data, on='geneSymbol', how='left')

    # Variant ID parsing (always done — downstream code may use these columns)
    parts = data["varId"].astype(str).str.strip().str.split("_", n=3, expand=True)
    chr_raw = parts[0]
    pos = pd.to_numeric(parts[1], errors="coerce")
    chr_mapped = chr_raw.replace({"23": "X", "24": "Y", "x": "X", "y": "Y"})
    data["Chromosome"] = "chr" + chr_mapped
    data["Pos"] = pos
    valid_chr = data["Chromosome"].str.match(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y)$", na=False)
    data = data[valid_chr & data["Pos"].notna()].copy()
    data["Start"] = data["Pos"].astype(int)
    data["End"] = data["Start"] + 1

    # Splicing
    if splicing_path is not None and splicing_path.exists():
        splicing_data = pd.read_feather(splicing_path)
        splicing_data = splicing_data.drop(columns=['sampleID'])
        splicing_data = splicing_data.rename(columns={
            'pvaluesBetaBinomial_jaccard': 'Fraser_pvaluesBetaBinomial_jaccard',
            'psi5': 'Fraser_psi5', 'psi3': 'Fraser_psi3',
            'rawOtherCounts_psi5': 'Fraser_rawOtherCounts_psi5',
            'rawOtherCounts_psi3': 'Fraser_rawOtherCounts_psi3',
            'rawCountsJnonsplit': 'Fraser_rawCountsJnonsplit',
            'jaccard': 'Fraser_jaccard',
            'rawOtherCounts_jaccard': 'Fraser_rawOtherCounts_jaccard',
            'delta_jaccard': 'Fraser_delta_jaccard',
            'delta_psi5': 'Fraser_delta_psi5', 'delta_psi3': 'Fraser_delta_psi3',
            'predictedMeans_jaccard': 'Fraser_predictedMeans_jaccard',
        })
        has_RNA_splicing = True

        # Splicing processing
        splicing_data = splicing_data.dropna(subset=['hgnc_symbol'])
        splicing_data['geneSymbol'] = splicing_data['hgnc_symbol'].astype(str).str.split(';')
        splicing_data = splicing_data.explode('geneSymbol', ignore_index=True)

        # Gene-level
        splicing_min_gene_level = splicing_data.loc[
            splicing_data.groupby("geneSymbol")["Fraser_pvaluesBetaBinomial_jaccard"].idxmin()
        ].reset_index(drop=True)
        splicing_min_gene_level = splicing_min_gene_level[['geneSymbol', 'Fraser_pvaluesBetaBinomial_jaccard']]
        splicing_min_gene_level = splicing_min_gene_level.rename(
            columns={'Fraser_pvaluesBetaBinomial_jaccard': 'Fraser_GenePvalue'}
        )

        # Variant-level via pyranges interval join
        splicing_data_pr = splicing_data.copy()
        splicing_data_pr = splicing_data_pr.rename(
            columns={'seqnames': 'Chromosome', 'start': 'Start', 'end': 'End'}
        )
        splicing_data_pr["End"] = splicing_data_pr["End"] + 1
        gr_pt = pr.PyRanges(data[["Chromosome", "Start", "End", "varId"]])
        gr_iv = pr.PyRanges(splicing_data_pr.drop(columns=["seqnames", "start", "end"], errors="ignore"))

        hit = gr_pt.join(gr_iv).df
        hit_min = hit.loc[
            hit.groupby("varId")["Fraser_pvaluesBetaBinomial_jaccard"].idxmin()
        ].reset_index(drop=True)
        hit_min = hit_min.rename(columns={'Start_b': 'Fraser_junction_start', 'End_b': 'Fraser_junction_end'})
        hit_min = hit_min.drop(columns=['Chromosome', 'Start', 'End', 'strand', 'hgnc_symbol', 'geneSymbol'])

        data = data.merge(splicing_min_gene_level, on='geneSymbol', how='left')
        data = data.merge(hit_min, on='varId', how='left')
    else:
        # No splicing data — fill all 15 Fraser_* columns with NaN to keep
        # downstream schema consistent with the has-splicing case.
        has_RNA_splicing = False
        for col in [
            'Fraser_GenePvalue',
            'Fraser_pvaluesBetaBinomial_jaccard',
            'Fraser_psi5', 'Fraser_psi3',
            'Fraser_rawOtherCounts_psi5', 'Fraser_rawOtherCounts_psi3',
            'Fraser_rawCountsJnonsplit',
            'Fraser_jaccard', 'Fraser_rawOtherCounts_jaccard',
            'Fraser_delta_jaccard',
            'Fraser_delta_psi5', 'Fraser_delta_psi3',
            'Fraser_predictedMeans_jaccard',
            'Fraser_junction_start', 'Fraser_junction_end',
        ]:
            data[col] = np.nan

    # ASE
    if ase_path is not None and ase_path.exists():
        ASE_data = pd.read_feather(ase_path)
        ASE_data = ASE_data[['varId', 'REF', 'ALT', 'REF_COUNT', 'ALT_COUNT', 'ALT_RATIO', 'PVAL', 'IS_MAE']]
        ASE_data = ASE_data.rename(columns={
            'REF': 'ASE_REF', 'ALT': 'ASE_ALT',
            'REF_COUNT': 'ASE_REF_COUNT', 'ALT_COUNT': 'ASE_ALT_COUNT',
            'ALT_RATIO': 'ASE_ALT_RATIO', 'PVAL': 'ASE_PVAL',
        })
        has_RNA_ASE = True
    else:
        ASE_data = pd.DataFrame({
            'varId': ['None'], 'ASE_REF': None, 'ASE_ALT': None,
            'ASE_REF_COUNT': np.nan, 'ASE_ALT_COUNT': np.nan,
            'ASE_ALT_RATIO': np.nan, 'ASE_PVAL': np.nan, 'IS_MAE': 0,
        })
        has_RNA_ASE = False
    data = data.merge(ASE_data, on='varId', how='left')
    
    #Candidate Rule:
    top100 = data["Diagnostic_Engine_Rank"].le(100)
    # rule1: Strong DNA Evidence
    rule1 = data["Diagnostic_Engine_Rank"].le(20)
    # rule2: Strong In-Silico Evidence
    rule2 = (data["rank_InSilico"].le(10) & top100)
    # rule3: Potential New Disease Gene (Database)
    rule3 = (data["rank_Database"].le(10) & top100)
    # rule4: Potential New Disease Gene (Genetics)
    rule4 = (data["rank_Genetics"].le(10) & top100)
    # rule5: RNA-level Variant
    rule5 = (data["Fraser_pvaluesBetaBinomial_jaccard"].lt(1e-5) | data["Outrider_pValue"].lt(1e-5)  & top100)

    #Calculate Mask
    mask = rule1 | rule2 | rule3 | rule4 | rule5
    flag = mask & data["recessive"].eq(1)
    recessive_mask = flag.groupby(data["geneSymbol"]).transform("any")
    final_mask = mask | recessive_mask

    #Check comp het:
    sub_data = data[["geneSymbol", "varId"]].drop_duplicates(subset=["geneSymbol", "varId"]).groupby(["geneSymbol"])["varId"].nunique()
    strong_sub = data.loc[mask, ["geneSymbol", "varId"]].drop_duplicates(subset=["geneSymbol", "varId"]).groupby(["geneSymbol"])["varId"].nunique()

    ch_df = pd.DataFrame({"total": sub_data, "n_strong": strong_sub,}).fillna(0).reset_index()
    ch_df["is_compound_het_group"] = ((ch_df["n_strong"] >= 1) & (ch_df["total"] - ch_df["n_strong"] >= 1))

    candidate_data = data.merge(ch_df[["geneSymbol", "is_compound_het_group"]], on="geneSymbol", how="left")
    candidate_data["is_compound_het_group"] = candidate_data["is_compound_het_group"].fillna(False)

    rule_columns = pd.DataFrame({
    "Strong DNA Evidence": rule1,
    "Strong In-Silico Evidence": rule2,
    "Strong Database Evidence": rule3,
    "Strong Genetics Evidence": rule4,
    "Strong RNA Evidence": rule5,
    "Compound Heterozygous (Compound Het)": candidate_data["is_compound_het_group"],}, index=candidate_data.index)

    candidate_data["evidence_rules"] = rule_columns.apply(_join_rule_labels, axis=1)
    candidate_data = candidate_data[final_mask]
    candidate_data = candidate_data[candidate_data["geneSymbol"]!="-"].copy()

    tmp = output_path.parent / f".{output_path.name}.tmp"
    candidate_data.to_feather(tmp)
    os.replace(tmp, output_path)
    return {
        "has_expression": has_RNA_expression,
        "has_splicing": has_RNA_splicing,
        "has_ase": has_RNA_ASE,
        "reused": False,
    }

def Candidates(samplesheet, work_dir, config=None, overwrite=False):
    default_config = {"candidates_workers": 1}
    cfg = {**default_config, **(config or {})}

    work_dir = Path(work_dir)

    if "moe_score_path" not in samplesheet.columns:
        raise ValueError(
            "samplesheet missing 'moe_score_path' column. "
            "Run DiagnosticEngine.MoE first."
        )

    outpath = work_dir / "Diagnostic_results" / "Candidates"
    outpath.mkdir(parents=True, exist_ok=True)

    print(f"Scanning Candidates ...")

    output_paths = {}
    futures_map = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=cfg["candidates_workers"]) as ex:
        for row in samplesheet.itertuples(index=True):
            sample_id = row.sampleID
            moe_path = Path(row.moe_score_path)
            output_path = outpath / f"{sample_id}_nomcand.feather"

            # RNA paths optional — pass None if column missing or value NaN
            expr_path = _get_optional_path(row, "rna_expression_path")
            splicing_path = _get_optional_path(row, "rna_splicing_path")
            ase_path = _get_optional_path(row, "rna_ase_path")

            fut = ex.submit(
                filter_one, sample_id, moe_path,
                expr_path, splicing_path, ase_path,
                output_path, overwrite,
            )
            futures_map[fut] = (row.Index, sample_id, output_path)
        sample_info = {}

        with tqdm(total=len(futures_map), desc="Detecting Candidates") as pbar:
            for fut in as_completed(futures_map):
                row_idx, sample_id, out_path = futures_map[fut]
                try:
                    info = fut.result()
                    output_paths[row_idx] = str(out_path)
                    sample_info[sample_id] = info
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"\n[ERROR] Candidates failed for {sample_id}: "
                          f"{type(e).__name__}: {e}")
                pbar.update(1)
                pbar.set_postfix(ok=ok, fail=fail)

    if fail > 0:
        raise RuntimeError(f"Candidates failed for {fail} sample(s).")

    # Aggregate rather than per-sample: a hundred-sample cohort should not
    # produce a hundred lines, and the counts are what actually matter.
    print()
    n = len(sample_info)
    for label, key in (("DNA", None),
                       ("Expression", "has_expression"),
                       ("Splicing", "has_splicing"),
                       ("ASE", "has_ase")):
        n_yes = n if key is None else sum(
            1 for info in sample_info.values() if info[key]
        )
        print(f"  {label:<11} {n_yes}/{n} sample(s)")

    n_reused = sum(1 for info in sample_info.values() if info.get("reused"))
    if n_reused:
        print(f"  ({n_reused} existing output(s) reused; "
              f"pass overwrite=True to rebuild)")

    # Update samplesheet
    samplesheet = samplesheet.copy()
    samplesheet["candidates_path"] = None
    for row_idx, path in output_paths.items():
        samplesheet.loc[row_idx, "candidates_path"] = path

    # Atomic write samplesheet
    samplesheet_path = work_dir / "samplesheet_with_paths.csv"
    tmp = work_dir / ".samplesheet_with_paths.csv.tmp"
    samplesheet.to_csv(tmp, index=False)
    os.replace(tmp, samplesheet_path)
    print(f"Updated samplesheet: {samplesheet_path}")

    print(f"--Candidate Filtering DONE--\n")
    return samplesheet


def _get_optional_path(row, col_name):
    """Helper: safely get an optional path column from a samplesheet row."""
    if not hasattr(row, col_name):
        return None
    val = getattr(row, col_name)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str) and val.strip() == "":
        return None
    return Path(val)
