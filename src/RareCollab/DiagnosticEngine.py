#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import pandas as pd
import pyranges as pr
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
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

def MoE(work_path, RANDOM_SEED = 42, RANK_ON_LOGIT = True):
    print(f'Creating work dir: ~/Diagnostic_results/DNA_MoE/')
    input_path = work_path + '/AIM_Preprocess/VarToGene/out'
    OUT_DIR = work_path + '/Diagnostic_results/DNA_MoE/out/'
    Done_DIR = work_path + '/Diagnostic_results/DNA_MoE/done/'
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(Done_DIR).mkdir(parents=True, exist_ok=True)
    MODEL_PATH = "/home/guantongq/workspace/RNA_diagnosis/DNA_RNA_agent/clean_folder/MoE_Diagnostic_Engine/MoE_finalized.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Set Random seeds:
    set_seed(RANDOM_SEED)

    #Prepare Model:
    print(f'Loading MoE Model ...')
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    domain_names = ckpt["domain_names"]
    domain_input = {k: ckpt["domain_preprocs_state"][k]["out_dim"] for k in domain_names}

    print(f"Domains Experts: {domain_names}")
    model = DomainMoE(domain_input = domain_input,
                  fusion_hidden_dims = ckpt['fusion_hidden_dims'],
                  expert_embed_dim= ckpt['expert_embed_dim'],
                  use_layer_norm = ckpt['use_layer_norm'],
                  expert_dropout = ckpt['expert_dropout'],
                  fusion_dropout = ckpt['fusion_dropout'])
    model = model.to(device)
    #Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[Model] Loaded. expert_embed_dim={ckpt['expert_embed_dim']}, fusion_hidden_dims={ckpt['fusion_hidden_dims']}, use_layer_norm={ckpt['use_layer_norm']}")
    except Exception as e:
        print("[ERROR] load_state_dict(strict=True) failed.")
        print(f"  - domain_names={domain_names}")
        print(f"  - expert_embed_dim={ckpt['expert_embed_dim']}, fusion_hidden_dims={ckpt['fusion_hidden_dims']}, use_layer_norm={ckpt['use_layer_norm']}")
        print(f"  - expert_dropout={ckpt['expert_dropout']}, fusion_dropout={ckpt['fusion_dropout']}")
        raise e
    model.eval()

    SampleIDs_Path = {p.name.split('.')[0]:p for p in Path(input_path).iterdir()}
    
    #Evaulating Samples ...
    for sample_id, sample_path in tqdm(SampleIDs_Path.items(), desc="MoE on Samples", total=len(SampleIDs_Path)):
        print(f"", flush = True)
        done = Path(Done_DIR + f"{sample_id}.done")
        if done.exists():
            continue
        processed_data = pd.read_feather(sample_path)
        N = len(processed_data)

        #Put/Split data in loader:
        loader = DataLoader(list(range(N)),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=1,
                    collate_fn=collate_fn(processed_data, LABEL_COL, ckpt["domain_preprocs_state"]),
                    pin_memory=(device == "cuda"),
                    persistent_workers=True)
        
        #Preallocate outputs
        overall_logit = np.empty(N, dtype=np.float32)
        overall_prob = np.empty(N, dtype=np.float32)
        domain_scores = {name: np.empty(N, dtype=np.float32) for name in domain_names}

        # ---- Forward ----
        with torch.no_grad():
            for batch in loader:
                idx = batch["idx"].numpy()
                inputs = {name: batch[f"{name}"].to(device, non_blocking=True) for name in domain_names}

                with torch.amp.autocast(device_type = device, enabled=(device == "cuda")):
                    out = model(inputs)
                    logit_t = out["overall_logit"]
                    prob_t = torch.sigmoid(logit_t)
                    dom_t = out["domain_logits"]  # [B, K]

                overall_logit[idx] = logit_t.detach().cpu().numpy().astype(np.float32, copy=False)
                overall_prob[idx] = prob_t.detach().cpu().numpy().astype(np.float32, copy=False)

                dom_np = dom_t.detach().cpu().numpy().astype(np.float32, copy=False)
                for j, name in enumerate(domain_names):
                    domain_scores[name][idx] = dom_np[:, j]
        
        # Organize output:
        output = processed_data[['varId','identifier','geneSymbol','dominant','recessive','zyg','HGVSc','cons_frameshift_variant',
                                 'transcript_id','HGVSc_core','HGVSp','transcript_score','gnomadGenePLI',
                                 'gnomadGeneOELof','gnomadGeneOELofUpper','is_causal']].copy()
        output["overall_logit"] = overall_logit
        output["overall_prob"] = overall_prob
        for name in domain_names:
            output[f"score_{name}"] = domain_scores[name]
            output[f"rank_{name}"] = output["varId"].map(output.groupby("varId", dropna=False)[f"score_{name}"].max().rank(method='max',ascending=False).astype(int))
        if RANK_ON_LOGIT:
            output['Diagnostic_Engine_Rank'] = output["varId"].map(output.groupby("varId", dropna=False)["overall_logit"].max().rank(method='max',ascending=False).astype(int))
        else:
            output['Diagnostic_Engine_Rank'] = output["varId"].map(output.groupby("varId", dropna=False)["overall_prob"].max().rank(method='max',ascending=False).astype(int))
        
        

        #Save output
        output.to_feather(OUT_DIR + sample_id + "_MoE_scores.feather")
        done.write_text("done\n", encoding="utf-8")
    
    print(f"--DNA Diagnostic Engine DONE--\n")

def _join_rule_labels(row_bool: pd.Series) -> str:
    names = [name for name, hit in row_bool.items() if bool(hit)]
    return ", ".join(names)


def filter_one(sample_id, sample_path, RNA_path, outpath):
    save_file = Path(f"{outpath}/{sample_id}_nomcand.feather")
    if save_file.exists():
        return 1
    #Load data:
    data = pd.read_feather(sample_path)

    #Load RNA related Files and Merge with DNA results:
    expression_path = RNA_path + "/Expression/" + sample_id + ".feather"
    splicing_path = RNA_path + "/Splicing/" + sample_id + ".feather"
    ASE_path = RNA_path + "/ASE/" + sample_id + ".feather"

    #Expression:
    if Path(expression_path).exists():
        expression_data = pd.read_feather(expression_path)
        expression_data = expression_data.rename(columns = {'GeneSymbol':'geneSymbol', 'pValue':'Outrider_pValue', 'padjust':'Outrider_padjust', 
                                                            'zScore':'Outrider_zScore', 'l2fc':'Outrider_l2f', 'rawcounts': 'Outrider_rawcounts'})
        if 'RawZscore' in expression_data.columns:
            expression_data = expression_data.rename(columns = {'RawZscore': 'Outrider_RawZscore'})
        else:
            expression_data['Outrider_RawZscore'] = np.nan
        has_RNA_expression = True
    else:
        expression_data = pd.DataFrame({'geneSymbol':['None'], 'Outrider_pValue':np.nan, 'Outrider_padjust':np.nan,
                                'Outrider_zScore':np.nan, 'Outrider_l2f':np.nan, 'Outrider_rawcounts':np.nan,
                                'Outrider_RawZscore': np.nan})
        has_RNA_expression = False
    data = data.merge(expression_data, on = 'geneSymbol', how = 'left')

    #Splicing:
    if Path(splicing_path).exists():
        splicing_data = pd.read_feather(splicing_path)
        splicing_data = splicing_data.drop(columns = ['sampleID'])
        splicing_data = splicing_data.rename(columns = {'pvaluesBetaBinomial_jaccard':'Fraser_pvaluesBetaBinomial_jaccard','psi5':'Fraser_psi5', 'psi3':'Fraser_psi3',
                                                        'rawOtherCounts_psi5':'Fraser_rawOtherCounts_psi5', 'rawOtherCounts_psi3':'Fraser_rawOtherCounts_psi3',
                                                        'rawCountsJnonsplit':'Fraser_rawCountsJnonsplit', 'jaccard':'Fraser_jaccard',
                                                        'rawOtherCounts_jaccard':'Fraser_rawOtherCounts_jaccard', 'delta_jaccard':'Fraser_delta_jaccard',
                                                        'delta_psi5':'Fraser_delta_psi5', 'delta_psi3':'Fraser_delta_psi3',
                                                        'predictedMeans_jaccard':'Fraser_predictedMeans_jaccard'})
        has_RNA_splicing = True
    else:
        splicing_data = pd.DataFrame({'seqnames':['None'], 'start':np.nan, 'end':np.nan, 'strand':np.nan, 'hgnc_symbol':np.nan,
                                    'Fraser_pvaluesBetaBinomial_jaccard': np.nan, 'Fraser_psi5': np.nan, 'Fraser_psi3': np.nan,
                                    'Fraser_rawOtherCounts_psi5': np.nan, 'Fraser_rawOtherCounts_psi3': np.nan, 'Fraser_rawCountsJnonsplit':np.nan,
                                    'Fraser_jaccard':np.nan, 'Fraser_rawOtherCounts_jaccard':np.nan, 'Fraser_delta_jaccard':np.nan,
                                    'Fraser_delta_psi5':np.nan, 'Fraser_delta_psi3':np.nan, 'Fraser_predictedMeans_jaccard':np.nan})
        has_RNA_splicing = False

    #data processing
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

    #splicing processing:
    splicing_data = splicing_data.copy()
    splicing_data = splicing_data.dropna(subset=['hgnc_symbol'])
    splicing_data['geneSymbol'] = splicing_data['hgnc_symbol'].astype(str).str.split(';')
    splicing_data = splicing_data.explode('geneSymbol', ignore_index=True)
    #gene-level processing:
    splicing_min_gene_level = splicing_data.loc[splicing_data.groupby("geneSymbol")["Fraser_pvaluesBetaBinomial_jaccard"].idxmin()].reset_index(drop=True)
    splicing_min_gene_level = splicing_min_gene_level[['geneSymbol','Fraser_pvaluesBetaBinomial_jaccard']]
    splicing_min_gene_level = splicing_min_gene_level.rename(columns = {'Fraser_pvaluesBetaBinomial_jaccard':'Fraser_GenePvalue'})
    #variant-level processing:
    splicing_data = splicing_data.copy()
    splicing_data = splicing_data.rename(columns = {'seqnames':'Chromosome', 'start':'Start', 'end':'End'})
    splicing_data["End"] = splicing_data["End"] + 1
    gr_pt = pr.PyRanges(data[["Chromosome", "Start", "End", "varId"]])
    gr_iv = pr.PyRanges(splicing_data.drop(columns=["seqnames", "start", "end"], errors="ignore"))

    hit = gr_pt.join(gr_iv).df
    hit_min = hit.loc[hit.groupby("varId")["Fraser_pvaluesBetaBinomial_jaccard"].idxmin()].reset_index(drop=True)
    hit_min = hit_min.rename(columns = {'Start_b': 'Fraser_junction_start', 'End_b': 'Fraser_junction_end'})
    hit_min = hit_min.drop(columns = ['Chromosome', 'Start', 'End', 'strand', 'hgnc_symbol', 'geneSymbol'])

    data = data.merge(splicing_min_gene_level, on = 'geneSymbol', how = 'left')
    data = data.merge(hit_min, on = 'varId', how = 'left')

    #ASE:
    if Path(ASE_path).exists():
        ASE_data = pd.read_feather(ASE_path)
        ASE_data = ASE_data[['varId','REF','ALT','REF_COUNT','ALT_COUNT','ALT_RATIO','PVAL','IS_MAE']]
        ASE_data = ASE_data.rename(columns = {'REF':'ASE_REF','ALT':'ASE_ALT','REF_COUNT':'ASE_REF_COUNT',
                                            'ALT_COUNT': 'ASE_ALT_COUNT', 'ALT_RATIO':'ASE_ALT_RATIO', 'PVAL': 'ASE_PVAL'})
        has_RNA_ASE = True
    else:
        ASE_data = pd.DataFrame({'varId':['None'], 'ASE_REF': None, 'ASE_ALT':None, 'ASE_REF_COUNT':np.nan,
                                'ASE_ALT_COUNT':np.nan, 'ASE_ALT_RATIO':np.nan, 'ASE_PVAL':np.nan, 'IS_MAE': 0})
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

    candidate_data.to_feather(save_file)
    print(f"Sample:{sample_id}, hasDNA: True; hasExpression: {has_RNA_expression}; hasSplicing:{has_RNA_splicing}; hasASE:{has_RNA_ASE}")
    return 1

def Candidates(work_path, max_workers = 5):
    DNA_path = work_path + '/Diagnostic_results/DNA_MoE/out'
    RNA_path = work_path + '/Diagnostic_results/RNA_MoE/'
    if not Path(DNA_path).exists():
        raise ValueError(f"Prerequsite Results NOT Found - Please Run MoE First ...")

    SamplePath = {p.name.split("_MoE_scores.feather")[0]:p for p in Path(DNA_path).iterdir()}
    outpath = work_path + '/Diagnostic_results/Candidates/'
    
    #Input/Output Path:
    print(f'Creating work dir: ~/Diagnostic_results/Candidates/')
    Path(outpath).mkdir(parents=True, exist_ok=True)
    
    #Filter Candidates:
    print(f"Scanning Candidate ...")
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sample_id, sample_path in SamplePath.items():
            futures.append(ex.submit(filter_one, sample_id, sample_path, RNA_path, outpath))
        ok = fail = 0
        with tqdm(total=len(SamplePath), desc="Detecting Candidates") as pbar:
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    ok += int(ret == 1)
                    fail += int(ret == 0)
                except Exception:
                    fail += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok, fail=fail)
    print(f'--Candidate Filtering DONE--\n')
