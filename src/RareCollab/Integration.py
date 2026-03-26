#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import json
import pysam

from pathlib import Path
from collections import defaultdict
from itertools import combinations
from tqdm.notebook import tqdm

def load_LLM_res(sub_root_path, model_type, candidate_type):
    res = []
    label_reason, label_conclusion = model_type + "_Reasoning", model_type + "_Conclusion"
    loading_list = {p.name.split('.json')[0]:p for p in Path(sub_root_path).iterdir() if p.suffix == '.json'}
    for GeneName, loading_path in loading_list.items():
        with loading_path.open("r", encoding = 'utf-8') as f:Evaluation = json.load(f)
        res.append([GeneName, Evaluation['Reasoning'], Evaluation['Conclusion']])
    res = pd.DataFrame(res, columns = [candidate_type,label_reason, label_conclusion])
    return res

def load_DatabaseLLM_res(sub_root_path):
    res = []
    loading_list = {p.name.split('.json')[0]:p for p in Path(sub_root_path).iterdir() if p.suffix == '.json'}
    for GeneName, loading_path in loading_list.items():
        with loading_path.open("r", encoding = 'utf-8') as f:Evaluation = json.load(f)
        res.append([GeneName, Evaluation['reasoning'], Evaluation['conclusion'], Evaluation['zygosity']])
    res = pd.DataFrame(res, columns = ['varId', 'Database_Reasoning', 'Database_Conclusion', 'Database_Zygosity'])
    return res

def load_RNALLM_res(sub_root_path, model_type, candidate_type):
    res = []
    label_reason, label_event, label_conclusion = model_type + "_Reasoning", model_type + "_Event", model_type + "_Conclusion"
    loading_list = {p.name.split('.json')[0]:p for p in Path(sub_root_path).iterdir() if p.suffix == '.json'}
    for GeneName, loading_path in loading_list.items():
        with loading_path.open("r", encoding = 'utf-8') as f:Evaluation = json.load(f)
        res.append([GeneName, Evaluation['Reasoning'], Evaluation['Event'], Evaluation['Conclusion']])
    res = pd.DataFrame(res, columns = [candidate_type,label_reason, label_event, label_conclusion])
    return res

def best_tandem_repeat(seq: str, min_k: int = 1, max_k: int = 6, min_repeats: int = 3):
    n = len(seq)
    best_k, best_motif, best_rep = 0, "", 0

    for k in range(min_k, min(max_k, n // min_repeats) + 1):
        for i in range(n - k * min_repeats + 1):
            motif = seq[i:i+k]
            if "N" in motif:
                continue

            rep = 1
            j = i + k
            while j + k <= n and seq[j:j+k] == motif:
                rep += 1
                j += k

            if rep >= min_repeats and rep > best_rep:
                best_k, best_motif, best_rep = k, motif, rep

    return best_k, best_motif, best_rep

def TandemRepeatDetection(data, fa):
    MOTIF_MIN_REPEATS = 10     # within +/-30bp window
    W_REPEAT = 30              # +/-30bp (61bp total)
    MIN_K = 1
    MAX_K = 6
    output_df = data[['varId','Chromosome','Pos','HGVSc']]
    #output_df = output_df.loc[output_df["HGVSc"].str.contains(r"del|dup", regex=True, na=False)].reset_index(drop = True)
    sub_res = []
    for row in output_df.itertuples(index=True):
        if "del" not in row.HGVSc and "dup" not in row.HGVSc:
            sub_res.append([0, False, "", 0, 0, ""])
            continue
        chrom = row.Chromosome
        pos = row.Pos
        _, _, ref, alt = row.varId.split('_')
        length = abs(len(ref) - len(alt))
        start = max(0, (pos - 1) - W_REPEAT)
        end = (pos - 1) + W_REPEAT + 1
        try:
            seq = fa.fetch(chrom, start, end).upper()
        except KeyError:
            seq = ""
            sub_res.append([length, False, "", 0, 0, seq])
            continue
        
        k, motif, rep = best_tandem_repeat(seq, min_k=MIN_K, max_k=MAX_K, min_repeats=3)
        k, motif, rep = best_tandem_repeat(seq, min_k=MIN_K, max_k=MAX_K, min_repeats=3)
        sub_res.append([length, rep >= MOTIF_MIN_REPEATS, motif if rep > 0 else "", k if rep > 0 else 0, rep, seq])
    sub_res = pd.DataFrame(sub_res, columns = ['length','has_repeat_motif','best_repeat_motif','best_repeat_k','best_repeat_repeats','seq_pos_pm30'])
    output_df = pd.concat([output_df, sub_res], axis = 1)
    output_df = output_df.drop(columns = ['Chromosome','Pos'])
    return output_df


def PairWiseComp(data, tier):
    HPO_Score = {'Not Fit':1, 'Partial Fit':2, 'Good Fit':3, 'Stand-Alone Strong Evidence':4}
    Db_Score = {'Against':0, 'Neutral':1, 'Supporting':2, 'Convincing':3}
    RNA_Score = {"Strong RNA Evidence":2, "Weak RNA Evidence":1,}


    comp_data = data[['varId','geneSymbol','Diagnostic_Engine_Rank','HPO_Conclusion','Database_Conclusion','RNA_GeneLevel_Conclusion','RNA_VarLevel_Conclusion','frame_shift','Insilico_Conclusion','partner']].copy()
    comp_data = comp_data[data['Tier'].isin(tier)].reset_index(drop = True)
    comp_data['HPO_Score'] = comp_data['HPO_Conclusion'].map(HPO_Score).fillna(0).astype(int)
    comp_data['Db_Score'] = comp_data['Database_Conclusion'].map(Db_Score).fillna(1).astype(int)

    comp_data['RNA_gene'] = comp_data['RNA_GeneLevel_Conclusion'].map(RNA_Score).fillna(0).astype(int)
    comp_data['RNA_var'] = comp_data['RNA_VarLevel_Conclusion'].map(RNA_Score).fillna(0).astype(int)
    comp_data['RNA_Score'] = comp_data[['RNA_gene', 'RNA_var']].max(axis=1)

    comp_data['frame_shift'] = comp_data['frame_shift'].astype(int)
    comp_data['Insilico_Score'] = (comp_data['Insilico_Conclusion'] == 'Strong').astype(int)

    comp_data = comp_data.drop(columns = ['HPO_Conclusion','Database_Conclusion','RNA_GeneLevel_Conclusion','RNA_VarLevel_Conclusion','Insilico_Conclusion','RNA_gene','RNA_var'])

    #Combine Paired Variants:
    comp_dict = defaultdict(dict)
    used = set()
    score_cols = ['Diagnostic_Engine_Rank', 'frame_shift', 'HPO_Score', 'Db_Score', 'RNA_Score', 'Insilico_Score']
    lookup = {}
    for idx, row in comp_data.iterrows():
        key = (row['geneSymbol'], row['varId'])
        lookup[key] = idx

    for idx, row in comp_data.iterrows():
        var1 = row['varId']
        gene = row['geneSymbol']
        partner = row['partner']
        if (gene, partner) in used:
            continue
        if partner != '':
            partner_idx = lookup[(gene, partner)]
            partner_row = comp_data.loc[partner_idx]
            key = (gene, var1, partner)
            comp_dict[key] = {col: [int(row[col]), int(partner_row[col])] for col in score_cols}
            comp_dict[key]['paired'] = True
            comp_dict[key]['Wins'] = 0 
            used.add((gene, var1))
            used.add((gene, partner))
        else:
            key = (gene, var1)
            comp_dict[key] = {col: [row[col]] for col in score_cols}
            comp_dict[key]['paired'] = False
            comp_dict[key]['Wins'] = 0 
            used.add(key)

    for (key1, val1), (key2, val2) in combinations(comp_dict.items(), 2):
        S1 = S2 = 0    
        if val1['paired'] == True and val2['paired'] == True:
            DE_rank_1,DE_rank_2 = sum(val1['Diagnostic_Engine_Rank'])/2, sum(val2['Diagnostic_Engine_Rank'])/2
            HPO_1, HPO_2 = sum(val1['HPO_Score'])/2, sum(val2['HPO_Score'])/2
            DB_1, DB_2 = sum(val1['Db_Score'])/2, sum(val2['Db_Score'])/2
            RNA_1, RNA_2 = max(val1['RNA_Score']), max(val2['RNA_Score'])
            FS_1, FS_2 = max(val1['frame_shift']), max(val2['frame_shift'])
            IS_1, IS_2 = max(val1['Insilico_Score']), max(val2['Insilico_Score'])
        else:
            DE_rank_1,DE_rank_2 = min(val1['Diagnostic_Engine_Rank']), min(val2['Diagnostic_Engine_Rank'])
            HPO_1, HPO_2 = max(val1['HPO_Score']), max(val2['HPO_Score'])
            DB_1, DB_2 = max(val1['Db_Score']), max(val2['Db_Score'])
            RNA_1, RNA_2 = max(val1['RNA_Score']), max(val2['RNA_Score'])
            FS_1, FS_2 = max(val1['frame_shift']), max(val2['frame_shift'])
            IS_1, IS_2 = max(val1['Insilico_Score']), max(val2['Insilico_Score'])      
        #DE:
        if DE_rank_1 < DE_rank_2:
            S1 += 1
        elif DE_rank_1 > DE_rank_2:
            S2 += 1
        #HPO:
        if HPO_1 > HPO_2:
            S1 += 1
        elif HPO_1 < HPO_2:
            S2 += 1
        if HPO_1 >= 3 and HPO_2 <= 1:
            S1 += 2
        elif HPO_2 >= 3 and HPO_1 <= 1:
            S2 += 2
        #DataBase
        if DB_1 >= 2 and DB_2 <= 1:
            S1 += 1
        elif DB_2 >= 2 and DB_1 <= 1:
            S2 += 1
        #RNA:
        if RNA_1 > RNA_2:
            S1 += 1
        elif RNA_1 < RNA_2:
            S2 += 1
        if RNA_1 >= 2 and RNA_2 < 2:
            S1 += 1
        elif RNA_2 >= 2 and RNA_1 < 2:
            S2 += 1
        #Frameshift
        if FS_1 == 1 and FS_2 == 0:
            S1 += 1
        elif FS_2 == 1 and FS_1 == 0:
            S2 += 1
        #In-silico:
        if IS_1 == 1 and IS_2 == 0:
            S1 += 1
        elif IS_2 == 1 and IS_1 == 0:
            S2 += 1

        if S1 > S2:
            comp_dict[key1]['Wins'] += 1
        elif S2 > S1:
            comp_dict[key2]['Wins'] += 1
        else:
            if DE_rank_1 > DE_rank_2:
                comp_dict[key1]['Wins'] += 1
            elif DE_rank_1 < DE_rank_2:
                comp_dict[key2]['Wins'] += 1

    tmp_rows = []
    for key, val in comp_dict.items():
        gene = key[0]
        var_ids = key[1:]
        wins = val['Wins']

        for var_id in var_ids:
            tmp_rows.append({'geneSymbol': gene, 'varId': var_id, 'PairWiseScore': wins})
    res_df = pd.DataFrame(tmp_rows)
    return res_df

def Review(work_path, reference_genome, output_path):
    input_root_nomcand = work_path + "/Diagnostic_results/Candidates/"
    input_root_phenotype = work_path + "/Agents/Phenotype/AgentEvaluation/"
    input_root_insilico = work_path + "/Agents/InSilico/AgentEvaluation/"
    input_root_rna = work_path + "/Agents/RNA/"

    #Tandem Repeats Param:
    print(f"Loading reference genome - Tandem Repeats")
    if not Path(reference_genome).exists():
        raise ValueError(f"{reference_genome} NOT Found. Skip")
    assert Path(reference_genome + ".fai").exists(), f"Missing fasta index (.fai). Run: samtools faidx {reference_genome}"
    fa = pysam.FastaFile(reference_genome)

    input_root_database = work_path + '/Agents/Database/AgentEvaluation/json/'
    database_res = load_DatabaseLLM_res(input_root_database)
    sample_ids = [p.name.split("_nomcand.feather")[0] for p in Path(input_root_nomcand).iterdir()]
    
    print(f"Reviewing Cases - Pairwise Comparison")
    pbar = tqdm(sample_ids, desc="Reviewing Cases", total=len(sample_ids))
    for sampleid in pbar:
        pbar.set_postfix(sample=sampleid)
        sub_nomcand_path = input_root_nomcand + sampleid + '_nomcand.feather'
        #phenotypes:
        sub_hpo_path = input_root_phenotype + 'HPO_Agent/' + sampleid
        sub_omim_path = input_root_phenotype + 'OMIM_Agent/' + sampleid
        sub_lit_path = input_root_phenotype + 'Literature_Agent/' + sampleid
        sub_insilico_path = input_root_insilico + sampleid
        sub_rna_genelevel_path = input_root_rna + 'GeneLevelEval/' + sampleid
        sub_rna_varlevel_path = input_root_rna + 'VariantLevelEval/' + sampleid

        #Load data:
        data = pd.read_feather(sub_nomcand_path).reset_index(drop = True)
        HPO_res = load_LLM_res(sub_hpo_path, model_type = 'HPO', candidate_type = 'geneSymbol')
        OMIM_res = load_LLM_res(sub_omim_path, model_type = 'OMIM', candidate_type = 'geneSymbol')
        Lit_res = load_LLM_res(sub_lit_path, model_type = 'Literature', candidate_type = 'geneSymbol')
        Insilico_res = load_LLM_res(sub_insilico_path, model_type = 'Insilico', candidate_type = 'varId')
        RNA_gene_res = load_RNALLM_res(sub_rna_genelevel_path, model_type = 'RNA_GeneLevel', candidate_type = 'geneSymbol')
        RNA_var_res = load_RNALLM_res(sub_rna_varlevel_path, model_type = 'RNA_VarLevel', candidate_type = 'geneSymbol_VarId')
        RNA_var_res[['geneSymbol', 'varId']] = RNA_var_res['geneSymbol_VarId'].str.split('_VarId_', n=1, expand=True)
        AQ_path = work_path + '/Agents/RNA/AlleleQuantification/' + sampleid + '.feather'
        if Path(AQ_path).exists():
            AlleleQuant = pd.read_feather(AQ_path).reset_index(drop = True)
        else:
            AlleleQuant = pd.DataFrame({'varId':['None'], 'geneSymbol': None, 'transcript_id':None, 'ref_count_max':0,
                                        'ref_count_mean':0, 'ref_count_min':0, 'alt_count_max':0, 'alt_count_mean': 0, 'alt_count_min':0})
        AlleleQuant = AlleleQuant.drop(columns = ['varId','geneSymbol','transcript_id'])
        data = pd.concat([data, AlleleQuant], axis = 1)

        #Merge res:
        data = data.merge(database_res, on = 'varId', how = 'left')
        data = data.merge(HPO_res, on  = 'geneSymbol', how = 'left')
        data = data.merge(OMIM_res, on  = 'geneSymbol', how = 'left')
        data = data.merge(Lit_res, on  = 'geneSymbol', how = 'left')
        data = data.merge(Insilico_res, on = 'varId', how = 'left')
        data = data.merge(RNA_gene_res, on = 'geneSymbol', how = 'left')
        data = data.merge(RNA_var_res, on = ['varId', 'geneSymbol'], how = 'left')

        #TandemRepeats:
        TandemRepeats = TandemRepeatDetection(data = data, fa = fa)
        data['has_repeat_motif'] = TandemRepeats['has_repeat_motif']

        #Classify variants into 2 tiers:
        #Tier 1 must be True for all 5 following criteria:
        data['misscall_rna_flag'] = True
        data.loc[(data['ref_count_mean'] > 20) & (data['alt_count_min'] == 0), 'misscall_rna_flag'] = False

        data['new_diseasegene_flag'] = True
        data.loc[(data['Literature_Conclusion'] == 'Not Fit') &
                (data['Database_Conclusion'].isin({'Neutral', 'Against'}) |
                data['Database_Conclusion'].isna()),'new_diseasegene_flag'] = False

        data['omim_flag'] = True
        data.loc[data['OMIM_Conclusion'] == 'Impossible', 'omim_flag'] = False

        data['strong_nom_flag'] = True
        data.loc[data['evidence_rules'] == 'Compound Heterozygous (Compound Het)', 'strong_nom_flag'] = False
        data.loc[data['evidence_rules'].isin({'Strong RNA Evidence','Strong RNA Evidence, Compound Heterozygous (Compound Het)'}) & 
                (data['RNA_GeneLevel_Conclusion'] != 'Strong RNA Evidence') & (data['RNA_VarLevel_Conclusion'] != 'Strong RNA Evidence'),'strong_nom_flag'] = False
        
        data['Tier'] = 2
        data.loc[data['misscall_rna_flag'] & (~data['has_repeat_motif']) & data['new_diseasegene_flag'] & data['omim_flag'] & data['strong_nom_flag'], 'Tier'] = 1

        data = data.sort_values(by=["Tier", "Diagnostic_Engine_Rank"], ascending=[True, True]).reset_index(drop = True)
        data['new_rank'] = range(1, len(data) + 1)

        #Find out if Tier1 has compound het gene:
        data['frame_shift'] = False
        data.loc[(data['cons_frameshift_variant'] == 1) & (data['HGVSp'].str.contains("Ter", na=False)), 'frame_shift'] = True

        data['find_compound_het'] = False
        data['partner'] = ''

        genehash = set()
        for i in range(len(data)):
            if data.loc[i,'geneSymbol'] in genehash:
                continue
            genehash.add(data.loc[i,'geneSymbol'])
            if ((data.loc[i,'recessive'] == 1) | (data.loc[i,'Database_Zygosity'] in {'compound_heterozygous', 'compound heterozygous'})) & (data.loc[i,'zyg'] == 1) & (data.loc[i,'Database_Zygosity'] != 'heterozygous'):
                Enough_reads = False
                ratio_direction = 0
                ref_mean = data.loc[i, 'ref_count_mean'] 
                alt_mean = data.loc[i, 'alt_count_mean']
                if ref_mean >= 50 and alt_mean >= 50:
                    Enough_reads = True
                    ratio = ref_mean / alt_mean
                    if ratio > 1.2:
                        ratio_direction = 1
                    elif ratio < 0.8:
                        ratio_direction = -1
                        
                het_candidates = data.loc[(data['geneSymbol'] == data.loc[i, 'geneSymbol']) & (data['varId'] != data.loc[i, 'varId']), 
                                        ['varId','geneSymbol','Database_Conclusion','RNA_GeneLevel_Conclusion','RNA_VarLevel_Conclusion',
                                        'frame_shift','new_rank','ref_count_mean','alt_count_mean']]
                het_candidates = het_candidates.loc[het_candidates['Database_Conclusion'].isin(set(['Supporting','Convincing'])) |
                                                    (het_candidates['RNA_GeneLevel_Conclusion'] == 'Strong RNA Evidence') |
                                                    (het_candidates['RNA_VarLevel_Conclusion'] == 'Strong RNA Evidence') |
                                                    het_candidates['frame_shift'] | (het_candidates['new_rank'] <= 50),]
                if len(het_candidates) == 0:
                    partner_varid = None
                else:
                    partner_varid = None
                    for candgene in het_candidates.itertuples(index=True):
                        if Enough_reads & (ratio_direction != 0) & (candgene.ref_count_mean >= 50) & (candgene.alt_count_mean >= 50):
                            candgene_ratio = candgene.ref_count_mean / candgene.alt_count_mean
                            if ratio_direction == 1 and candgene_ratio > 1.2:
                                continue
                            if ratio_direction == -1 and candgene_ratio < 0.8:
                                continue
                        partner_varid = candgene.varId
                        partner_j = candgene.Index
                        break
                if partner_varid:
                    data.loc[i, 'find_compound_het'] = True
                    data.loc[i, 'partner'] = partner_varid
                    data.loc[partner_j, 'find_compound_het'] = True
                    data.loc[partner_j, 'partner'] = data.loc[i, 'varId']
                    data.loc[partner_j, 'Tier'] = data.loc[i, 'Tier']
                
        #reorder:
        for i in range(len(data)):
            zyg_ok = data.loc[i, "Database_Zygosity"] in {"homozygous", "compound heterozygous", "compound_heterozygous"}
            if ((data.loc[i, 'zyg'] == 1) & 
                    (data.loc[i,'Chromosome'] != 'chrX') &
                    (not data.loc[i,'find_compound_het']) &
                    (zyg_ok | (data.loc[i,'recessive'] == 1) & (data.loc[i,'dominant'] == 0) & (data.loc[i, "Database_Zygosity"] != 'heterozygous'))):
                data.loc[i, 'Tier'] = 3

        data = data.sort_values(by=["Tier", "Diagnostic_Engine_Rank"], ascending=[True, True]).reset_index(drop = True)
        PW_Res = PairWiseComp(data, tier = [1])
        data = data.merge(PW_Res, on=['geneSymbol', 'varId'], how='left')
        data = data.sort_values(by=["Tier", "PairWiseScore"], ascending=[True, False]).reset_index(drop = True)
        data.to_csv(output_path + '/' + sampleid + '.csv', index = False)
    print(f"-- Review Done --")
