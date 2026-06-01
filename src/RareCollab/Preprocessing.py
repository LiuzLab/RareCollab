#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pronto
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

#Default Key Params:
PredictionFilePath = '/prediction/conf_4Model/'
predictionFileName = '*_default_predictions.csv'
ScoreFilePath = '/merged/scores.txt.gz'
desired_cols = {"geneSymbol", "varId", "HGVSc", "HGVSp"}


def RNA(work_path: str, splicing_path: str, expression_path: str, ase_path: str) -> None:
    if '.feather' in splicing_path:
        junction = pd.read_feather(splicing_path)
    elif '.csv' in splicing_path:
        junction = pd.read_csv(splicing_path)
    else:
        raise ValueError(f"Format Error - Splicing Result - Only take feather or csv file")
    if '.feather' in expression_path:
        expression = pd.read_feather(expression_path)
    elif '.csv' in expression_path:
        expression = pd.read_csv(expression_path)
    else:
        raise ValueError(f"Format Error - Expression Result - Only take feather or csv file")
    if '.feather' in ase_path:
        ase = pd.read_feather(ase_path)
    elif '.csv' in ase_path:
        ase = pd.read_csv(ase_path)
    else:
        raise ValueError(f"Format Error - Expression Result - Only take feather or csv file")
    
    #expression outlier:
    expression_cols = ['sampleID','GeneSymbol','pValue','padjust','zScore','l2fc','rawcounts']
    if 'RawZscore' in expression.columns:
        expression_cols.append('RawZscore')
    expression = expression[expression_cols]

    #aberrant splicing:
    #gene level:
    junction_col = ['sampleID','seqnames','start','end','strand','hgnc_symbol','pvaluesBetaBinomial_jaccard','psi5', 'psi3',
                       'rawOtherCounts_psi5','rawOtherCounts_psi3', 'rawCountsJnonsplit','jaccard',
                       'rawOtherCounts_jaccard', 'delta_jaccard', 'delta_psi5', 'delta_psi3','predictedMeans_jaccard']
    junction = junction[junction_col]

    #ASE
    is_chr = ase['CHROM'].str.match(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y)$", case=False, na=False)
    is_num = ase['CHROM'].str.match(r"^([1-9]|1[0-9]|2[0-4])$", na=False)
    ase = ase[is_chr | is_num].copy()
    ase['CHROM'] = ase['CHROM'].str.replace(r"^chr", "", regex=True, case=False)
    ase['CHROM'] = ase['CHROM'].replace({"X": "23", "x": "23", "Y": "24", "y": "24"})
    ase['varId'] = ase['CHROM'].astype(str) + '_' + ase['POS'].astype(str) + '_' + ase['REF'] + "_" + ase['ALT']

    save_path = work_path + '/Diagnostic_results/RNA_MoE/'

    print(f"Processing Expression Data ...")
    expression_save_path = save_path + 'Expression/'
    Path(expression_save_path).mkdir(parents=True, exist_ok=True)
    for curr_sample in list(set(expression['sampleID'])):
        sub_expression = expression[expression['sampleID'] == curr_sample]
        sub_expression.to_feather(expression_save_path + curr_sample + '.feather')
    
    print(f"Processing Splicing Data ...")
    junction_save_path = save_path + 'Splicing/'
    Path(junction_save_path).mkdir(parents=True, exist_ok=True)
    for curr_sample in list(set(junction['sampleID'])):
        sub_junction = junction[junction['sampleID'] == curr_sample]
        sub_junction.to_feather(junction_save_path + curr_sample + '.feather')


    print(f"Processing ASE Data ...")
    ase_save_path = save_path + 'ASE/'
    Path(ase_save_path).mkdir(parents=True, exist_ok=True)
    for curr_sample in list(set(ase['sampleID'])):
        sub_ase = ase[ase['sampleID'] == curr_sample]
        sub_ase.to_feather(ase_save_path + curr_sample + '.feather')

    print('-- RNA Preprocessing Done --')
    return

