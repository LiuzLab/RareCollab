#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pronto
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm

#Default Key Params:
PredictionFilePath = '/prediction/conf_4Model/'
predictionFileName = '*_default_predictions.csv'
ScoreFilePath = '/merged/scores.txt.gz'
desired_cols = {"geneSymbol", "varId", "HGVSc", "HGVSp"}

def CurateInputPath(aim_path: str) -> dict[str, tuple[Path, Path]]:
    #Scan Folders(Each folder is a sample):
    SampleIDs = [p.name for p in Path(aim_path).iterdir()]

    #Check the Input data in each folder:
    InputPath = defaultdict(tuple)
    for sample_id in SampleIDs:
        PredictionFile = list(Path(aim_path +'/'+ sample_id + PredictionFilePath).glob(predictionFileName))[0]
        if not PredictionFile:
            raise ValueError(f"Prediction CSV of {sample_id} NOT Found at {PredictionFile}")
        ScoreFile = Path(aim_path + '/' + sample_id + ScoreFilePath)
        if not ScoreFile.is_file():
            raise ValueError(f"Score File of {sample_id} NOT Found at {ScoreFile}")
        InputPath[sample_id] = (PredictionFile, ScoreFile)
    return InputPath

def load_GT_list(path:str, SamplePath:dict):
    GT_list = pd.read_csv(path, dtype = 'string')
    #Check colnames:
    for c in ['SampleID', 'chr', 'pos', 'ref', 'alt']:
        if c not in GT_list.columns:
            raise ValueError(f"column {c} NOT Found in Ground Truth list")
    #Check ID:
    print(f"{len(set(GT_list['SampleID']).intersection(SamplePath.keys()))}/{len(SamplePath)} sample(s) have known casual variant")
    if GT_list['chr'].dropna().str.startswith("chr").any():  # from UCSC to ENSEMBL
        GT_list['chr'] = GT_list['chr'].str[3:]
        GT_list['chr'] = GT_list['chr'].mask(GT_list['chr'] == "M", "MT")
    GT_list['varId'] = GT_list['chr'] + '_' + GT_list['pos'] + '_' + GT_list['ref'] + '_' + GT_list['alt']
    return GT_list

def Process_Pred_Score(sample_id: str, pred_path: Path, score_path: Path, GT_list, protein_coding_genes, mane_trascripts):
    #Load predicted features:
    Predictions = pd.read_csv(pred_path, engine="pyarrow")
    Predictions = Predictions.rename(columns={Predictions.columns[0]: "varId"})
    if Predictions.shape[0] < 1 or Predictions.shape[1] < 1:
        raise ValueError(f"Prediction file for {sample_id} is not correct - Too few variants or too few features") 
    
    #Process Score File:
    #Define the columns needed in the score file:
    header_cols = set(pd.read_csv(score_path, sep="\t", compression="infer", nrows=0).columns)
    use_cols = sorted(desired_cols & header_cols)
    #check varId:
    if len(use_cols) != len(desired_cols):
        tmp = set(desired_cols) - set(use_cols)
        raise ValueError(f"Column(s): {tmp} NOT found in {score_path}") 
    #Read Score in data stream:
    dtype_map = {c: "string" for c in use_cols}
    Variant_Symbol = pd.read_csv(score_path, sep="\t", compression="infer", usecols=use_cols,dtype=dtype_map)
    #Only keep Protein Coding:
    Variant_Symbol = Variant_Symbol[Variant_Symbol['geneSymbol'].isin(protein_coding_genes)].copy()
    #Split HGVSc to be 1.transcript id and 2.detail:
    tmp = Variant_Symbol["HGVSc"].str.split(":", n=1, expand = True)
    Variant_Symbol["transcript_id"] = tmp[0].str.replace(r"\..*$", "", regex=True).fillna("-")
    Variant_Symbol["HGVSc_core"] = tmp[1].fillna("-")
    #Caluate a disease-likely score:
    Variant_Symbol['is_mane'] = Variant_Symbol['transcript_id'].isin(mane_trascripts)
    Variant_Symbol['has_hgvsp'] = ~Variant_Symbol['HGVSp'].isin({'-'})
    Variant_Symbol['has_hgvsc'] = ~Variant_Symbol['HGVSc'].isin({'-'})
    Variant_Symbol["transcript_score"] = 4*Variant_Symbol["is_mane"].astype(int) + 2*Variant_Symbol["has_hgvsp"].astype(int) + Variant_Symbol["has_hgvsc"].astype(int)
    #For each variant-gene pair, keep the one with the highest score
    idx = Variant_Symbol.groupby(['varId','geneSymbol'])["transcript_score"].idxmax()
    Variant_Symbol = Variant_Symbol.loc[idx].reset_index(drop=True)

    Variant_Symbol["is_causal"] = 0
    #Mark casual variants if have ground truth:
    if GT_list is not None:
        casuals = set(GT_list.loc[GT_list['SampleID'] == sample_id,'varId'])
        Variant_Symbol.loc[Variant_Symbol['varId'].isin(casuals),'is_causal'] = 1
    #Check the final dataframe:
    if len(Variant_Symbol) < 1:
        print(f"No intersection between pred and socre - {sample_id}")
    Merged = Predictions.merge(Variant_Symbol, on = 'varId', how = 'inner')
    
    return Merged

def vtg_process_one(sample_id, pred_path, score_path, GT_list, protein_coding_genes, mane_trascripts, output_dir: str, done_dir: str) -> None:
    done = Path(done_dir + f"{sample_id}.done")
    if done.exists():
        return 1
    MergedFile = Process_Pred_Score(sample_id, pred_path, score_path, GT_list, protein_coding_genes, mane_trascripts)
    MergedFile.to_feather(output_dir + sample_id + '.feather')
    done.write_text("done\n", encoding="utf-8")
    return 1

def VarToGene(work_path: str, aim_path: str, max_workers: int, HasGroundTruth: bool, GroudTruthList: str, GENECODE_ANNOT_path:str, MANE_TRANSCRIPT_path:str) -> None:
    # GroudTruthList SHOULD contain 5 columns - SampleID, chr, pos, ref, alt
    # SampleID should be the same as AIM output folder names
    # chr should be Ensembl format without chr* (eg: chr1, chr2; UCSC format will not be accepted)
    # pos - position; ref - reference; alt - alternative
    OUTPUT_DIR = work_path + '/AIM_Preprocess/VarToGene/out/'
    Done_DIR = work_path + '/AIM_Preprocess/VarToGene/done/'

    #1. Create output dir:
    print(f'Creating work dir: ~/AIM_Preprocess/VarToGene/')
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(Done_DIR).mkdir(parents=True, exist_ok=True)

    #2. Get the path of required input files:
    print(f'Scanning DNA Features ...')
    SamplePath = CurateInputPath(aim_path)

    #3.Get the Ground Truth list (if any):
    if HasGroundTruth:
        print(f'Checking Ground Truth list ...')
        GT_list = load_GT_list(GroudTruthList, SamplePath)
    else:
        print(f'No Ground Truth list. Skip.')
        GT_list = None
    
    #4.Load ProteinCoding genes and MANE Transcripts:
    print(f'Loading necessary documents ...')
    gencode = pd.read_feather(GENECODE_ANNOT_path)
    mane = pd.read_feather(MANE_TRANSCRIPT_path)
    
    protein_coding_genes = set(gencode.loc[gencode['gene_type'] == 'protein_coding','gene_name'])
    mane_trascripts = set(mane['transcript_id'].str.split('.').str[0].dropna())

    #5. Process samples in parallel:
    print(f'Sample Processing ...')
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sample_id, (pred_path, score_path) in SamplePath.items():
            futures.append(ex.submit(vtg_process_one, sample_id, pred_path, score_path, GT_list, protein_coding_genes, mane_trascripts, OUTPUT_DIR, Done_DIR))
        ok = fail = 0
        with tqdm(total=len(SamplePath), desc="Sample Processsing") as pbar:
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    ok += int(ret == 1)
                    fail += int(ret == 0)
                except Exception:
                    fail += 1
                pbar.update(1)
                pbar.set_postfix(Processed=ok, Fail=fail)
    print(f'--Variant To Gene Symbol DONE--\n')
'''
def AnnotatePhenotype(HPO_patient_path:str, HPO_lib_path:str, aim_path:str, work_path:str):
    SampleIDs = set([p.name for p in Path(aim_path).iterdir()])
    print(f"Decoding HPO Documents ...")
    HPO_dict = pronto.Ontology(HPO_lib_path)

    print(f"Decoding HPO Documents ...")
    patient_hpo_list = []
    for hpo_file in Path(HPO_patient_path).iterdir():
        sample_id = hpo_file.name.split('.hpo')[0]
        if sample_id not in SampleIDs:
            print(f"sample - {sample_id} not found in AIM output - Skip")
            continue
        with hpo_file.open("r", encoding="utf-8") as f:
            hpo_ids = [line.strip() for line in f if line.strip()]
        #If there's no valid HPO terms for the paitent: assign 'All' to it and set 'with_hpo' = 0
        if (len(hpo_ids) == 1 and hpo_ids[0] == 'HP:0000001') or len(hpo_ids) < 1:
            patient_hpo_list.append([sample_id,'HP:0000001','All',0])
        else:
            for hpo_id in hpo_ids:
                if hpo_id in HPO_dict:
                    patient_hpo_list.append([sample_id, hpo_id, HPO_dict[hpo_id].name,1])
    if len(patient_hpo_list) < 1:
        patient_hpo_list.append([sample_id,'HP:0000001','All',0])
        print(f"No vaild HPO Terms (eg: HP:xxxxxxx) found in {sample_id}. Using HP:0000001 as default term.")
    merged_hpo = pd.DataFrame(patient_hpo_list, columns = [['sample_id','hpo_id', 'hpo_terms', 'with_hpo']])
    
    output_path = work_path + "/AIM_Preprocess/Phenotype/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_hpo.to_feather(output_path+'patient_hpo_terms.feather')
    print(f'--Annotate Phenotype DONE--\n')
    return

'''

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

