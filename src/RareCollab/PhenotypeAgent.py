#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
import pronto
import requests
import xml.etree.ElementTree as ET
import time
import re
from pathlib import Path
from collections import defaultdict
from tqdm.notebook import tqdm
from Bio import Entrez, Medline
from io import StringIO

#↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓
#↓-↓-Preprocessing-↓-↓
def AnnotatePhenotype(HPO_patient_path:str, HPO_lib_path:str, AnnotatePhenotype_path:str, SampleIDs:set):
    Path(AnnotatePhenotype_path).mkdir(parents=True, exist_ok=True)
    save_file = Path(AnnotatePhenotype_path + '/Patients_HPO_Term.json')
    if save_file.exists():
        with open(save_file, "r", encoding="utf-8") as f:
            tmp = json.load(f)
        return tmp
    print(f"Decoding HPO Documents ...")
    HPO_dict = pronto.Ontology(HPO_lib_path)

    patient_hpo_dict = defaultdict(list)
    for hpo_file in Path(HPO_patient_path).iterdir():
        sample_id = hpo_file.name.split('.hpo')[0]
        if sample_id not in SampleIDs:
            print(f"sample - {sample_id} not found in AIM output - Skip")
            continue
        with hpo_file.open("r", encoding="utf-8") as f:
            hpo_ids = [line.strip() for line in f if line.strip()]
        #If there's no valid HPO terms for the paitent: assign 'All' to it and set 'with_hpo' = 0
        if (len(hpo_ids) == 1 and hpo_ids[0] == 'HP:0000001') or len(hpo_ids) < 1:
            patient_hpo_dict[sample_id] = [('HP:0000001','All',0)]
        else:
            for hpo_id in hpo_ids:
                if hpo_id in HPO_dict:
                    patient_hpo_dict[sample_id].append((hpo_id, HPO_dict[hpo_id].name,1))
        if len(patient_hpo_dict[sample_id]) < 1:
            patient_hpo_dict[sample_id] = [(sample_id,'HP:0000001','All',0)]
            print(f"No vaild HPO Terms (eg: HP:xxxxxxx) found in {sample_id}. Using HP:0000001 as default term.")
    
    with Path(save_file).open("w", encoding="utf-8") as f: json.dump(patient_hpo_dict, f)
    print(f'--Annotate Phenotype DONE--\n')
    return patient_hpo_dict

def p_to_label(p: float) -> str:
    if p == 1.0: return "Obligate"
    if p == 0.0: return "Excluded"
    if 0.80 <= p < 1.0: return "Very frequent"
    if 0.30 <= p < 0.80: return "Frequent"
    if 0.05 <= p < 0.30: return "Occasional"
    if 0.01 <= p < 0.05: return "Very rare"
    return "Very rare"

def to_num_den(freq: str):
    TERM_TO_P = {
        "HP:0040280": 1.00,   # Obligate
        "HP:0040281": 0.90,   # Very frequent
        "HP:0040282": 0.55,   # Frequent
        "HP:0040283": 0.17,   # Occasional
        "HP:0040284": 0.025,  # Very rare
        "HP:0040285": 0.00,   # Excluded
    }
    if freq == "-":
        return 0.0, 0.0, 0
    if freq.startswith("HP:"):
        den = 6.0
        num = TERM_TO_P[freq] * den
        return num, den, 1
    if freq.endswith("%"):
        den = 6.0
        num = float(freq[:-1])/100 * den
        return num, den, 1
    m, n = freq.split("/")
    return float(int(m)), float(int(n)), 1

def build_gene2hpo(df: pd.DataFrame) -> dict[str, str]:
    tmp = df[["gene_symbol", "hpo_id", "hpo_name", "frequency"]].copy()
    parsed = tmp["frequency"].apply(to_num_den)
    tmp["num"] = parsed.map(lambda t: t[0])
    tmp["den"] = parsed.map(lambda t: t[1])
    tmp["has"] = parsed.map(lambda t: t[2])

    grp = tmp.groupby(["gene_symbol", "hpo_id"], as_index=False).agg(
        hpo_name=("hpo_name", "first"),
        num=("num", "sum"),
        den=("den", "sum"),
        has=("has", "max"),
    )

    grp["entry"] = grp.apply(lambda r: f"{r.hpo_name} (Frequency not specified)" if (r.has == 0 or r.den == 0) else f"{r.hpo_name} ({p_to_label(r.num / r.den)})", axis=1)

    return grp.groupby("gene_symbol")["entry"].apply(lambda s: ", ".join(s)).to_dict()

def Preprocessing(work_path, HPO_patient_path, HPO_lib_path, HPO_genes_path, OMIM_path):
    root_path = work_path + '/Agents/Phenotype/'
    AnnotatePhenotype_path = root_path + '/HPO/'
    #1. Confirm data (Candidate list):
    input_path = work_path + '/Diagnostic_results/Candidates/'
    if not Path(input_path).exists():
        raise ValueError(f"Input file NOT detected, please run 'DiagnosticEngine.Candidates' first and check the work_path ... ")

    SamplePath = {p.name.split('_nomcand.feather')[0]:p for p in Path(input_path).iterdir()}
    #Merge Sample Data:
    print('Loading Patient Candidate Gene Files ...')
    Merged_Data_Path = Path(root_path + '/PatientCandGene/MergedFile.feather')
    if Merged_Data_Path.exists():
        print('Merged Data detected. Loading ...')
        Merged_Data = pd.read_feather(Merged_Data_Path)
    else:
        print('Merging Data ...')
        Path(root_path + '/PatientCandGene/').mkdir(parents=True, exist_ok=True)
        Merged_Data = []
        for p in SamplePath.values():
            df = pd.read_feather(p, columns = ['identifier','geneSymbol'])
            df = df.drop_duplicates().reset_index(drop = True)
            Merged_Data.append(df)
        Merged_Data = pd.concat(Merged_Data, ignore_index=True, copy=False)
        Merged_Data.to_feather(Merged_Data_Path)
    print('-- Patient HPO Data -- Done')    
    
    #Get Patient HPO Terms:
    print('Preparing HPO Documents ...')
    save_patient_hpo_term_dict = Path(AnnotatePhenotype_path + '/Patient_HPO_Dict.json')
    if save_patient_hpo_term_dict.exists():
        print(f"-- Patient HPO Files are prepared --")
    else:
        patient_dict = AnnotatePhenotype(HPO_patient_path = HPO_patient_path, 
                                        HPO_lib_path = HPO_lib_path, 
                                        AnnotatePhenotype_path = AnnotatePhenotype_path, 
                                        SampleIDs = SamplePath.keys())
        #PatientID:HPO TERMs:
        patient_hpo_term_dict = {sample_id: ", ".join(t[1] for t in tuples)for sample_id, tuples in patient_dict.items()}
        with Path(save_patient_hpo_term_dict).open("w", encoding="utf-8") as f: json.dump(patient_hpo_term_dict, f)
        print(f'--Patient HPO Files are prepared--\n')

    #HPO MAP:
    print('Processing HPO Gene Mapping ...')
    save_hpo_map = Path(AnnotatePhenotype_path + '/HPO_MAP.json')
    if save_hpo_map.exists():
        print(f"-- HPO MAP is prepared --")
    else:
        df = pd.read_csv(HPO_genes_path, sep="\t", dtype=str)
        hpo_map = build_gene2hpo(df)
        with Path(save_hpo_map).open("w", encoding="utf-8") as f: json.dump(hpo_map, f)
        print(f'--Patient HPO Files are prepared--\n')

    #Process OMIM Data:
    omim_save_file = AnnotatePhenotype_path + '/OMIM_MAP.json'
    if Path(omim_save_file).exists():
        print(f"OMIM Documents are prepared")
    else:
        print(f"Processing OMIM Documents ...")
        df = pd.read_table(Path(OMIM_path))
        df = df[['gene_symbol', 'Phenotypes', 'Clinical Features']].copy()
        df = df.dropna(subset = ["Clinical Features"]).reset_index(drop = True)
        OMIM_Gene_MAP  = (df.assign(_cf=lambda d: d["Clinical Features"].str.replace("\n\n", "\n", regex=False),
                                    _txt=lambda d: (d["gene_symbol"] + " causes " + d["Phenotypes"].astype(str) + ".\nClinical Features: " + d["_cf"]).str.strip())
                                    .groupby("gene_symbol")["_txt"]
                                    .apply(list)
                                    .to_dict()
                                    )
        with Path(omim_save_file).open("w", encoding="utf-8") as f: json.dump(OMIM_Gene_MAP, f)
        print(f'--OMIM Documents are prepared--\n')

    return
#↑-↑-Preprocessing-↑-↑
#↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑

#↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓
#↓-↓-↓-HPO Agent-↓-↓-↓
def build_prompt_HPO_agent(gene: str, patient_terms: str, hpo_gene_des: str) -> str:
    patient_block = patient_terms if patient_terms else "No patient phenotype terms provided."
    hpo_block = (hpo_gene_des if (hpo_gene_des and hpo_gene_des.strip()) else f"No documented phenotype from HPO for {gene}.")

    prompt = f"""You are a clinical genomics assistant. Your goal is to assess how well a candidate gene's known phenotypes match a patient's phenotype profile.

### Inputs
- Gene: {gene}

**Patient Phenotype (HPO terms):**
{patient_block}

**Gene Phenotype — HPO-derived (unique terms; Frequency Term when available):**
{hpo_block}

### Instructions
Definition of Vaild Frequency Term:
- Obligate: Always present, i.e. in 100% of the cases.
- Very frequent: Present in 80-99% of the cases.
- Frequent: Present in 30-79% of the cases.
- Occasional: Present in 5-29% of the cases.
- Very rare: Present in 1-4% of the cases.
- Excluded: Present in 0% of the cases.

#### A. General constraints
1) Use ONLY the information above. Do not invent additional phenotypes or disease mechanisms beyond what is provided.
2) Weigh both:
   - Positive matches: overlapping or very similar features, especially when they appear frequently in the gene phenotypes.
   - Unexplained key patient features: important patient findings that are not represented in the gene phenotypes.
   Treat unexplained features as “unexplained” rather than automatic contradictions, unless the gene phenotypes text explicitly indicate the opposite pattern.
3) Keep reasoning concise but specific: explicitly mention overlapping and unexplained features.

#### B. Think in phenotype axes / systems
4) Organize your reasoning by **major phenotype axes / systems**, rather than just counting raw terms. Typical axes include (but are not limited to):
   - Neuromuscular / motor axis (hypotonia, muscle weakness, myopathy, motor delay, fatigability, contractures, gait disturbance, etc.)
   - Central nervous system / brain structure axis (microcephaly, macrocephaly, brain malformations, seizures, ataxia, encephalopathy, etc.)
   - Cognitive / developmental / behavioral axis (global developmental delay, intellectual disability, autism, behavioral abnormalities, regression, etc.)
   - Craniofacial / dysmorphic axis (facial dysmorphism, micrognathia/prognathia, ear anomalies, palate anomalies, etc.)
   - Ocular axis (optic atrophy, retinal disease, cataracts, ophthalmoplegia, nystagmus, etc.)
   - Auditory axis (hearing loss or related ear findings).
   - Skeletal / limb / contracture axis (limb malformations, scoliosis/kyphosis, joint contractures, talipes, bone anomalies, fractures, etc.)
   - Growth axis (failure to thrive, short stature, overgrowth, abnormal weight gain/loss, slender/stocky build, etc.)
   - Cardiac axis (cardiomyopathy, arrhythmia, conduction defects, structural heart disease, etc.)
   - Respiratory axis (respiratory insufficiency, hypoventilation, recurrent infections, diaphragm weakness, etc.)
   - Gastrointestinal / feeding / nutrition axis (feeding difficulties, poor appetite, vomiting, malabsorption, tube feeding, etc.)
   - Endocrine / metabolic axis.
   - Genitourinary / genital axis.
   - Hematologic / immune / skin-hair axis.
   - Any other axis clearly suggested by the provided terms.

5) For each relevant axis:
   - Summarize key patient features on that axis.
   - Summarize overlapping gene features on that axis (mention frequency if present).
   - State whether this axis is **strongly explained**, **partially explained**, or **not explained** by the gene.

#### C. Label definitions
6) Use the following precise rules for the overall verdict:

- **Stand-Alone Strong Evidence**
  - Based on the provided phenotypes alone (HPO-derived terms), the gene explains the patient's phenotype in a **highly specific, convincing, diagnosis-like way**.
  - There is **strong, specific overlap across the main disease axes**, including multiple distinctive matching features (not just generic findings), and minimal important unexplained features.
  - No strong contradictions are evident based on the provided phenotypes.

- **Good Fit**
  - The gene phenotypes explain **most of the patient's key features across the main disease axes**.
  - There is strong, specific overlap on at least one major axis, and other major axes are either compatible or only missing a few features.
  - No strong contradictions are evident based on the provided phenotypes.

- **Partial Fit**
  - The gene phenotypes clearly and specifically explain **at least one major disease axis** of the patient, with **multiple specific overlapping features** on that axis (not just one or two very generic findings).
  - There may be important patient features or axes that remain unexplained by the gene phenotypes; these could be due to another gene, modifier, or unrelated causes.
  - Partial Fit should be used when there is a **coherent, disease-like pattern of overlap** on at least one axis, even if other axes are unexplained.

- **Not Fit**
  - There is **no major axis** with a clear, specific pattern of overlap between patient and gene phenotypes.
  - Any overlap is limited to **very nonspecific features** (for example, only “developmental delay”, “feeding difficulties”, or “hypotonia” without additional specific matching features), and no coherent disease pattern emerges.
  - Or the gene's typical phenotype pattern (based on the provided terms) is clearly incompatible with the patient's main problems, according to the information given.

7) When you are uncertain:
   - If the match is so specific that it feels like a convincing standalone diagnosis from phenotype alone, choose **Stand-Alone Strong Evidence**.
   - If there is a clear, specific pattern of overlap on at least one major axis with multiple matching features, but not fully convincing as a standalone diagnosis, lean toward **Good Fit** or **Partial Fit** depending on how many major axes are explained.
   - If overlap is only through 1-2 very generic findings without a coherent pattern, lean toward **Not Fit**.

#### D. Output formatting rules:
8) Output MUST be valid JSON. No markdown. No code fences. No extra keys. Use ASCII characters only. Use straight apostrophe ' and straight double quote " only.
   Return exactly one JSON object with keys "Reasoning" and "Conclusion".

   - "Reasoning" use bullet points grouped by major axis (e.g. "Neuromuscular axis:", "CNS/brain axis:", etc.), mapping key patient features to gene features, and stating for each axis whether it is strongly explained / partially explained / not explained.
   - "Conclusion" must be exactly one of: "Not Fit", "Partial Fit", "Good Fit", "Stand-Alone Strong Evidence".

   Output schema (must match exactly):{{"Reasoning":"<string>","Conclusion":"<Not Fit|Partial Fit|Good Fit|Stand-Alone Strong Evidence>"}}
   If you cannot comply, output exactly:{{"Reasoning":"comply_error","Conclusion":"Not Fit"}}
"""

    return prompt

def RunAgent_HPO(work_path, MODEL_NAME, OLLAMA_URL, TEMPERATURE):
    root_path = work_path + '/Agents/Phenotype/'
    AnnotatePhenotype_path = root_path + '/HPO/'
    input_path = work_path + '/Diagnostic_results/Candidates/'
    if not Path(input_path).exists():
        raise ValueError(f"Input file NOT detected, please run 'DiagnosticEngine.Candidates' first and check the work_path ... ")

    #Merged Data:
    Merged_Data_Path = Path(root_path + '/PatientCandGene/MergedFile.feather')
    if Merged_Data_Path.exists():
        print('Merged Data detected. Loading ...')
        Merged_Data = pd.read_feather(Merged_Data_Path)
    else:
        raise ValueError(f"Merged Data Not Found, please Run phenotype preprocessing first")
    
    #Paitent HPO
    save_patient_hpo_term_dict = Path(AnnotatePhenotype_path + '/Patient_HPO_Dict.json')
    if save_patient_hpo_term_dict.exists():
        print('Patient HPO Data detected. Loading ...')
        with save_patient_hpo_term_dict.open("r", encoding="utf-8") as f:patient_hpo_term_dict = json.load(f)
    else:
        raise ValueError(f"Patient HPO Data Not Found, please Run phenotype preprocessing first")
    
    #HPO MAP
    save_hpo_map = Path(AnnotatePhenotype_path + '/HPO_MAP.json')
    if save_hpo_map.exists():
        print('HPO MAP detected. Loading ...')
        with save_hpo_map.open("r", encoding="utf-8") as f:hpo_map = json.load(f)
    else:
        raise ValueError(f"HPO MAP Not Found, please Run phenotype preprocessing first")

    #Generate Prompts and Run LLM:
    print('Generating Prompts and Run LLM ...')
    LLM_Path = root_path + '/AgentEvaluation/HPO_Agent/'
    pbar = tqdm(Merged_Data.itertuples(index=False), desc="Evaluating Evidence", total=len(Merged_Data))
    for row in pbar:
        sample_id = row.identifier
        gene = row.geneSymbol
        pbar.set_postfix(sample=sample_id)
        Path(LLM_Path + '/' + sample_id).mkdir(parents=True, exist_ok=True)
        save_file_txt = Path(LLM_Path + "/" + sample_id + "/" + gene + '.txt')
        save_file_json = Path(LLM_Path + "/" + sample_id + "/" + gene + '.json')
        if save_file_json.exists():
            continue
        
        patient_terms = patient_hpo_term_dict.get(sample_id, "")
        if patient_terms == "All":
            patient_terms = ""
        hpo_gene_des = hpo_map.get(gene, "")
        
        prompt_text = build_prompt_HPO_agent(gene = gene, patient_terms = patient_terms, hpo_gene_des = hpo_gene_des)
        url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt_text,
            "stream": False,
            "options": {"temperature": float(TEMPERATURE)},
        }

        #As the reasoning is too long, and come with bullets, it cannot garantee a json output:
        #max_try = 3 times
        obj = None
        for attemp in range(3):
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            llm_output = data.get("response", "").strip()
            try:
                obj = json.loads(llm_output)
                break
            except Exception:
                continue
        if obj is None:
            raise RuntimeError(f"Failed to parse JSON after 3 tries. Last output: {llm_output}")

        conclusion = obj['Conclusion'] if 'Conclusion' in obj else 'Not Fit'
        reasoning_line = obj['Reasoning'] if 'Reasoning' in obj else 'comply_error'
        final_text = f"{reasoning_line}\nConclusion: {conclusion}\n"

        Path(save_file_txt).write_text(final_text, encoding="utf-8")
        with Path(save_file_json).open("w", encoding="utf-8") as f: json.dump(obj, f)

    print(f"--Phenotype HPO Agent's Work is DONE--\n")

#↑-↑-↑-HPO Agent-↑-↑-↑
#↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑

#↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-
#↓-↓-↓-OMIM Agent-↓-↓-↓
def build_prompt_OMIM_agent(gene: str, patient_hpo: str, omim_entries: str) -> str:
    patient_block = patient_hpo if (patient_hpo and patient_hpo.strip()) else "No patient phenotype terms provided."
    # Note: by construction, omim_entries should be non-empty here.
    omim_block = "\n\n".join([f"{i+1}.\n{s}" for i, s in enumerate(omim_entries)])
    prompt = f"""You are a clinical genomics assistant. Your task is to assess whether this gene's OMIM Clinical Features match the patient's phenotype.

Inputs
- Gene: {gene}

Patient phenotype (HPO terms):
{patient_block}

OMIM evidence (gene-linked; ONLY Clinical Features):
- Each numbered item below describes a disease/phenotype that is associated with this gene.
- The sentence "{gene} causes ..." explicitly indicates the gene-disease association for that item.

{omim_block}

Rules (must follow)
1) Use ONLY the text above. Do not add external gene knowledge.
2) Treat the patient phenotype list as the ONLY patient evidence available.
3) Do not require axis-based organization. Keep reasoning short and concrete.
4) Consider both:
   - Matches: specific overlaps between patient terms and OMIM clinical features.
   - Missing: major OMIM features not present in the patient terms.

Conclusion labels (choose exactly ONE)
- Stand-Alone Strong Evidence:
  OMIM clinical features provide a highly specific, diagnosis-like match to the patient; multiple distinctive overlaps; minimal major missing features.
- Good Fit:
  Patient matches most major OMIM clinical features; strong overlap overall; missing features are few or not central.
- Partial Fit:
  Clear, specific overlap exists for an important subset of OMIM features, but substantial major OMIM features are missing from the patient terms.
- Not Fit:
  No coherent specific overlap pattern; overlaps (if any) are only very generic and not compelling.
- Impossible:
  Use ONLY when OMIM clinical features explicitly state a phenotype is REQUIRED/INVARIABLE/ALWAYS PRESENT (i.e., a must-have core feature),
  and that must-have feature is NOT present (or clearly not represented) in the patient phenotype terms.
  Do NOT use Impossible just because something is “common” or “typical”; it must be stated as required/always present in the provided OMIM text.

Output formatting rules (STRICT):
Output MUST be valid JSON. No markdown. No code fences. No extra keys.
Use ASCII characters only. Use straight apostrophe ' and straight double quote " only.

Return exactly one JSON object with keys "Reasoning" and "Conclusion".
'Reasoning' should contain 2-6 bullet points (one line for each bullet) starting with '-', then in the middle explaining matches, missing major features, and any explicit must-have mismatch if present.
'Conclusion' must be exactly one of 'Impossible', 'Not Fit', 'Partial Fit', 'Good Fit', 'Stand-Alone Strong Evidence'

Output schema (must match exactly):{{"Reasoning":"<string>","Conclusion":"<Impossible|Not Fit|Partial Fit|Good Fit|Stand-Alone Strong Evidence>"}}
If you cannot comply, output exactly:{{"Reasoning":"comply_error","Conclusion":"Impossible"}}
"""
    return prompt

def RunAgent_OMIM(work_path, MODEL_NAME, OLLAMA_URL, TEMPERATURE, Candidates = None):
    root_path = work_path + '/Agents/Phenotype/'
    AnnotatePhenotype_path = root_path + '/HPO/'
    input_path = work_path + '/Diagnostic_results/Candidates/'
    if not Path(input_path).exists():
        raise ValueError(f"Input file NOT detected, please run 'DiagnosticEngine.Candidates' first and check the work_path ... ")

    #Merged Data:
    print('Loading Data ...')
    Merged_Data_Path = Path(root_path + '/PatientCandGene/MergedFile.feather')
    if Merged_Data_Path.exists():
        print('Merged Data detected. Loading ...')
        Merged_Data = pd.read_feather(Merged_Data_Path)
    else:
        raise ValueError(f"Merged Data Not Found, please Run phenotype preprocessing first")

    #Patient HPO:
    save_patient_hpo_term_dict = Path(AnnotatePhenotype_path + '/Patient_HPO_Dict.json')
    if save_patient_hpo_term_dict.exists():
        print('Patient HPO Data detected. Loading ...')
        with save_patient_hpo_term_dict.open("r", encoding="utf-8") as f:patient_hpo_term_dict = json.load(f)
    else:
        raise ValueError(f"Patient HPO Data Not Found, please Run phenotype preprocessing first")

    #OMIM MAP:
    save_omim_map = Path(AnnotatePhenotype_path + '/OMIM_MAP.json')
    if save_omim_map.exists():
        print('OMIM Documents detected. Loading ...')
        with save_omim_map.open("r", encoding="utf-8") as f:OMIM_Gene_MAP = json.load(f)
    else:
        raise ValueError(f"OMIM Documents Not Found, please Run phenotype preprocessing first")
    
    #Generate Prompts and Run LLM:
    print('Generating Prompts and Run LLM ...')
    LLM_Path = root_path + '/AgentEvaluation/OMIM_Agent/'
    pbar = tqdm(Merged_Data.itertuples(index=False), desc="Evaluating Evidence", total=len(Merged_Data))
    skipped_genes = 0
    for row in pbar:
        sample_id = row.identifier
        gene = row.geneSymbol
        pbar.set_postfix(sample=sample_id)
        Path(LLM_Path + '/' + sample_id).mkdir(parents=True, exist_ok=True)
        save_file_txt = Path(LLM_Path + "/" + sample_id + "/" + gene + '.txt')
        save_file_json = Path(LLM_Path + "/" + sample_id + "/" + gene + '.json')
        if save_file_json.exists():
            continue
        
        patient_terms = patient_hpo_term_dict.get(sample_id, "")
        if patient_terms == "All":
            patient_terms = ""
        omim_entries  = OMIM_Gene_MAP.get(gene, [])

        if not omim_entries:
            skipped_genes += 1
            continue

        prompt_text = build_prompt_OMIM_agent(gene, patient_terms, omim_entries)
        url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt_text,
            "stream": False,
            "options": {"temperature": float(TEMPERATURE)},
        }

        #As the reasoning is too long, and come with bullets, it cannot garantee a json output:
        #max_try = 3 times
        obj = None
        for attemp in range(3):
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            llm_output = data.get("response", "").strip()
            try:
                obj = json.loads(llm_output)
                break
            except Exception:
                continue
        if obj is None:
            raise RuntimeError(f"Failed to parse JSON after 3 tries. Last output: {llm_output}")

        conclusion = obj['Conclusion'] if 'Conclusion' in obj else 'Not Fit'
        reasoning_line = obj['Reasoning'] if 'Reasoning' in obj else 'comply_error'
        final_text = f"{reasoning_line}\nConclusion: {conclusion}\n"

        Path(save_file_txt).write_text(final_text, encoding="utf-8")
        with Path(save_file_json).open("w", encoding="utf-8") as f: json.dump(obj, f)
    if skipped_genes > 0:
        print(f'{skipped_genes} genes with no Clinical features are skipped')

    print(f"--Phenotype OMIM Agent's Work is DONE--\n")

#↑-↑-↑-OMIM Agent-↑-↑-↑
#↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-


#↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-↓-
#↓-↓-↓-Literature Agent-↓-↓-↓
def build_pubmed_term(gene: str) -> str:
    gene_q = f"\"{gene}\"[Title/Abstract]"
    kw_q = "(" + " OR ".join(
        [f"{kw}[Title/Abstract]" for kw in ["variant", "variants", "mutation",
                                            "mutations", "diagnosis", "mendelian",
                                            "cause", "causes"]]
    ) + ")"
    return f"({gene_q}) AND {kw_q}"

def esearch_pmids(term: str, retmax: int = 400, max_retries: int = 5):
    last_e = None
    for attempt in range(max_retries):
        try:
            with Entrez.esearch(db="pubmed", term=term, retmax=retmax, sort="relevance") as handle:
                res = Entrez.read(handle)
            return res.get("IdList", [])
        except Exception as e:
            last_e = e
            sleep_s = min(60, 2 ** attempt)
            time.sleep(sleep_s)
    print(f"[esearch] ERROR after max retries: {last_e} term={term}")
    return []

def _make_gene_boundary_regex(gene: str) -> re.Pattern:
    """
    Strictly match the gene name: letters/numbers/underlines etc are not allowed
    eg: 'SNUPN' match exactly '... SNUPN ...'; NOT match 'SNUPN1' or 'pre-SNUPN'.
    """
    g = re.escape(gene)
    return re.compile(rf"(?<![A-Za-z0-9_\-]){g}(?![A-Za-z0-9_\-])")

def local_strict_filter(records: list[dict[str, str]], gene: str, max_keep: int = 50):
    """
    Filter unrelated gene names in Title/Abstract (ignore case)
    return max_keep items
    """
    gpat = _make_gene_boundary_regex(gene)
    KW_REGEX = re.compile(r"\b(?:variant|variants|mutation|mutations|diagnosis|mendelian|cause|causes)\b", re.I)
    out = []
    for r in records:
        pmid = str(r.get("PMID", "")).strip()
        title = (r.get("TI") or "").strip()
        abstr = (r.get("AB") or "").strip()
        text = f"{title} {abstr}"
        if not pmid or not title:
            continue
        if not gpat.search(text):
            continue
        if not KW_REGEX.search(text):
            continue
        out.append({"PMID": pmid, "Title": title})
        if len(out) >= max_keep:
            break
    return out

def efetch_medline(pmids: list[str], chunk_size: int = 200, max_retries: int = 4):
    records = []

    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i : i + chunk_size]
        last_e = None
        for attempt in range(max_retries):
            try:
                with Entrez.efetch(
                    db="pubmed",
                    id=",".join(chunk),
                    rettype="medline",
                    retmode="text",
                ) as handle:
                    records.extend(Medline.parse(handle))
                break 
            except Exception as e:
                last_e = e
                sleep_s = min(60, 2 ** attempt)
                time.sleep(sleep_s)
        else:
            print(
                f"[efetch] ERROR after retries: {last_e} "
                f"pmids[{i}:{min(i + chunk_size, len(pmids))}] n={len(chunk)}"
                )

    return records

def prompt_for_classification(gene: str, items: list[dict[str, str]]) -> str:
    """
    Return LLM prompy with "- PMID :: Title".
    """
    header = (
        "You are a precise literature triager for rare disease.\n"
        f"Gene: {gene}\n"
        "Given a list of items (PMID, Title), OUTPUT ONLY the items that are explicitly about rare disease contexts and classify EACH into exactly one of:\n"
        "- phenotype association (links gene to disease/phenotype without mechanistic experiments),\n"
        "- functional study (mechanistic or model experiments about variant impact or gene function),\n"
        "- patient cohort (multi-patient genetic study),\n"
        "- case report (single patient or small family report),\n"
        "- review (review article).\n"
        "If uncertain, choose the most plausible single category. Do not include anything unrelated to rare disease.\n"
        "Return as TSV with exactly 3 columns and no extra commentary:\n"
        "PMID\\tTitle\\tCategory\n"
        "\nItems:\n"
    )

    lines = []
    for it in items:
        lines.append(f"- {it['PMID']}: {it['Title']}")
    lines.append("\nNow produce ONLY the TSV with header line.")
    return header + "\n".join(lines)

def fetch_pubmed_records(pmids: list[str],EUTILS_EFETCH, NCBI_EMAIL, NCBI_KEY) -> dict[str, dict[str, str]]:
    NCBI_SLEEP_SEC = 0.34
    pmids = [p for p in pmids if p and p.isdigit()]
    if not pmids:
        return {}
    out = {}
    PMID_BATCH_SIZE = 50
    for i in range(0, len(pmids), PMID_BATCH_SIZE):
        batch = pmids[i:i+PMID_BATCH_SIZE]
        recs = fetch_pubmed_batch(batch,EUTILS_EFETCH, NCBI_EMAIL, NCBI_KEY)
        out.update(recs)
        if NCBI_SLEEP_SEC:
            time.sleep(NCBI_SLEEP_SEC)
    return out

def fetch_pubmed_batch(pmids: list[str], EUTILS_EFETCH, NCBI_EMAIL, NCBI_KEY) -> dict[str, dict[str, str]]:
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
        "tool": "literature_agent",
    }
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_KEY:
        params["api_key"] = NCBI_KEY

    r = requests.get(EUTILS_EFETCH, params=params, timeout=60)
    r.raise_for_status()
    out = {}
    root = ET.fromstring(r.text)
    # XML: PubmedArticleSet/PubmedArticle/MedlineCitation/PMID; Article/ArticleTitle; Article/Abstract/AbstractText
    for art in root.findall(".//PubmedArticle"):
        pmid_node = art.find("./MedlineCitation/PMID")
        pmid = pmid_node.text.strip() if (pmid_node is not None and pmid_node.text) else None
        # Abstract can have multiple AbstractText nodes; concatenate
        abs_parts = []
        for at in art.findall("./MedlineCitation/Article/Abstract/AbstractText"):
            t = "".join(at.itertext()).strip() if at is not None else ""
            if t:
                abs_parts.append(t)
        abstract = " ".join(abs_parts).strip()

        if pmid:
            out[pmid] = {"title": "", "abstract": abstract}
    return out

def build_prompt_Literatrue_agent(gene: str, phenotype_text: str, lit_items: list[dict[str, str]]) -> str:
    """
    Build prompt:
    - Patient phenotype (HPO labels 串)
    - Gene symbol
    - Literature (PMID + Title + Abstract)
    """
    phenostr = phenotype_text if phenotype_text else "(not provided)"
    gene_line = f"Gene: {gene}"

    #literatures:
    if lit_items:
        lit_lines = []
        for it in lit_items:
            lit_lines.append( f"""- PMID: {it['pmid']} Title: {it.get('title','')} Abstract: {it.get('abstract','')}""")
        lit_block = "\n".join(lit_lines)
    else:
        lit_block = "(no qualifying papers found by category filter)"

    prompt = f"""You are a genetics literature reviewer assisting with phenotype–gene matching.

Patient phenotype (HPO labels):
{phenostr}

{gene_line}

Relevant literature (PubMed abstracts with PMIDs):
{lit_block}

Task:
Based only on the patient phenotype and the gene-associated information described in the abstracts above, assess how well the gene's **reported** phenotype matches the patient's phenotype profile.

Important constraints:
- Use ONLY the information in the text above. Do not add phenotypes or mechanisms from outside knowledge.
- Abstracts often describe only the most typical or partial phenotype for a gene. 
  → Absence of a feature in the abstracts should usually be treated as “unknown”, not as evidence against the gene.
- Give more weight to:
  * Clear overlaps in organ system / disease axis (e.g. neuromuscular disease, epileptic encephalopathy, cardiomyopathy, renal disease).
  * Specific shared findings (e.g. “congenital hypotonia”, “optic atrophy”, “hypertrophic cardiomyopathy”, “early-onset ataxia”).
- Use mismatch only when the abstracts explicitly describe a phenotype pattern that is clearly incompatible with the patient's main problems
  (for example: purely adult-onset isolated cardiomyopathy in the literature vs. isolated congenital brain malformation in the patient).

Label definitions:
- Good Fit:
  - The abstracts describe a phenotype pattern that strongly overlaps the patient's main problems
    (same key organ systems and age of onset, and several specific shared features),
  - and there are no clear conflicts based on what is written.
- Partial Fit:
  - The abstracts support a plausible overlap with the patient:
    * same main organ system or syndrome family, OR
    * several shared but not exhaustive features,
  - but the description is incomplete, more general, or only partially matches the patient.
  - This is the appropriate label when the literature suggests the gene could reasonably explain at least part of the patient's phenotype, but important details are missing.
- Not Fit:
  - The abstracts either:
    (a) describe a phenotype pattern that is clearly different from the patient's main problems (based on the text), OR
    (b) contain no meaningful phenotype information relevant to the patient (no overlap beyond extremely generic words like “patient”, “disease”, “mutation” without phenotypic content).

When uncertain between Partial Fit and Not Fit:
- If there is at least one plausible overlap at the level of organ system or phenotype cluster, lean toward **Partial Fit**.
- If there is no meaningful overlap in phenotype or organ system, lean toward **Not Fit**.

Return exactly one line JSON object with keys "Reasoning" and "Conclusion":
'Reasoning' should be a brief and concrete rationale grounded in phenotype terms and the abstracts; explicitly mention overlaps / or clear differences, and cite PMIDs in parentheses where relevant, e.g. (PMID:12345678).
'Conclusion' must be exactly one of 'Not Fit', 'Partial Fit', 'Good Fit'.

Output schema (must match exactly):{{"Reasoning":"<string>","Conclusion":"<Not Fit|Partial Fit|Good Fit>"}}
If you cannot comply, output exactly:{{"Reasoning":"comply_error","Conclusion":"Not Fit"}}
"""

    return prompt

def RunAgent_Literature(work_path, MODEL_NAME, OLLAMA_URL, TEMPERATURE, Entrez_EMAIL = None, Entrez_KEY = None, NCBI_EMAIL = None, NCBI_KEY = None):
    root_path = work_path + '/Agents/Phenotype/'
    AnnotatePhenotype_path = root_path + '/HPO/'
    EUTILS_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    input_path = work_path + '/Diagnostic_results/Candidates/'
    if not Path(input_path).exists():
        raise ValueError(f"Input file NOT detected, please run 'DiagnosticEngine.Candidates' first and check the work_path ... ")

    #Merged Data:
    print('Loading Data ...')
    Merged_Data_Path = Path(root_path + '/PatientCandGene/MergedFile.feather')
    if Merged_Data_Path.exists():
        print('Merged Data detected. Loading ...')
        Merged_Data = pd.read_feather(Merged_Data_Path)
    else:
        raise ValueError(f"Merged Data Not Found, please Run phenotype preprocessing first")


    print('Loading Gene HPO ...')
    Gene_MAP_Path = Path(root_path + 'HPO/HPO_MAP.json')
    if Gene_MAP_Path.exists():
        print('Gene HPO detected. Loading ...')
        with Gene_MAP_Path.open("r", encoding = 'utf-8') as f:GeneMap = json.load(f)
    else:
        raise ValueError(f"Gene HPO Not Found, please Run phenotype preprocessing first")

    #Only search genes that don't have HPO Terms:
    Genelist = list(set(Merged_Data['geneSymbol']))
    Genelist = list(set(Genelist) - set(GeneMap.keys()))
    print(f"{len(Genelist)} Candidate Genes prepared for Search")

    #Search:
    if Entrez_EMAIL:
        Entrez.email = Entrez_EMAIL
    if Entrez_KEY:
        Entrez.api_key = Entrez_KEY
    Gene_Pubmed_Path = root_path + '/GenePubmed/'
    Collected_Literatures_Path = root_path + '/GeneLiterature/'
    Related_Literatures_Path = root_path + '/GeneAbstract/'
    Path(Gene_Pubmed_Path).mkdir(parents = True, exist_ok = True)
    Path(Collected_Literatures_Path).mkdir(parents = True, exist_ok = True)
    Path(Related_Literatures_Path).mkdir(parents = True, exist_ok = True)

    #Collect Literature:
    print(f"Collecting Literatures ...")
    Literature_category = {"functional study","review", "case report", "patient cohort", "phenotype association"}
    pbar = tqdm(Genelist, desc="Accessing Literatures", total=len(Genelist))
    for gene in pbar:
        pbar.set_postfix(gene=gene)

        #Check if Literature related with this gene has been searched 
        search_res_path = Path(Gene_Pubmed_Path + '/' + gene + '.json')
        if Path(search_res_path).exists():
            with search_res_path.open('r', encoding = 'utf-8') as f:pmids = json.load(f)
        else:
            search_term = build_pubmed_term(gene)
            pmids = esearch_pmids(search_term, retmax=400, max_retries = 5)
            with Path(search_res_path).open("w", encoding="utf-8") as f: json.dump(pmids, f)
        
        #If we found some paper related to this gene:
        if pmids:
            #Step 1: Access whether the literature worth reviewing...
            save_file_tsv = Path(Collected_Literatures_Path + gene + ".tsv")
            if save_file_tsv.exists():
                #This gene has already been accessed...
                obj = pd.read_csv(save_file_tsv, dtype = 'str')
            else:
                recs = efetch_medline(pmids = pmids)
                items = local_strict_filter(recs, gene, max_keep=50)
                prompt_text = prompt_for_classification(gene, items)
                url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
                payload = {
                    "model": MODEL_NAME,
                    "prompt": prompt_text,
                    "stream": False,
                    "options": {"temperature": float(TEMPERATURE)},
                }
                obj = None
                for attemp in range(3):
                    r = requests.post(url, json=payload, timeout=600)
                    r.raise_for_status()
                    data = r.json()
                    llm_output = data.get("response", "").strip()
                    llm_output_fixed = llm_output.replace("\\t", "\t")
                    try:
                        obj = pd.read_csv(StringIO(llm_output_fixed), sep="\t", dtype=str)
                        break
                    except Exception:
                        continue
                if obj is None:
                    raise RuntimeError(f"Failed to parse tsv file after 3 tries. Last output: {llm_output}")
                obj.to_csv(save_file_tsv, index = False)
            #Step 2: Search the paper abstract:
            save_file_json = Path(Related_Literatures_Path + gene + ".json")
            if save_file_json.exists():
                #The literatures of this gene have been searched ...
                continue
            if len(obj) == 0:
                #This gene has no rare-disease related literature...
                continue
            obj["Category"] = obj["Category"].str.lower().str.strip()
            obj = obj[obj["Category"].isin(Literature_category)]
            filtered_pmids = list(set(obj['PMID']))
            if len(filtered_pmids) == 0:
                #No valid paper id
                continue
            #Search the tile and Abstract:
            try:
                pmid2rec = fetch_pubmed_records(filtered_pmids, EUTILS_EFETCH = EUTILS_EFETCH, NCBI_EMAIL = NCBI_EMAIL, NCBI_KEY = NCBI_KEY)
            except Exception as e:
                pmid2rec = {}
                print(f"[WARN] PubMed fetch failed for {gene} ): {e}")
            for pmid, item in pmid2rec.items():
                pmid2rec[pmid]['title'] = obj.loc[obj['PMID'] == pmid, 'Title'].iloc[0]
            with Path(save_file_json).open("w", encoding="utf-8") as f: json.dump(pmid2rec, f)
        else:
            #No findings; next gene ...
            continue

    #Literature Agent:
    save_patient_hpo = Path(AnnotatePhenotype_path + '/Patient_HPO_Dict.json')
    if save_patient_hpo.exists():
        print('HPO MAP detected. Loading ...')
        with save_patient_hpo.open("r", encoding="utf-8") as f:patient_hpo = json.load(f)
    else:
        raise ValueError(f"HPO MAP Not Found, please Run phenotype preprocessing first")

    Merged_Data = Merged_Data[Merged_Data['geneSymbol'].isin(set(Genelist))].reset_index(drop = True)
    LLM_Path = root_path + '/AgentEvaluation/Literature_Agent/'
    print(f"Reviewing by Literature Agent ...")
    pbar = tqdm(Merged_Data.itertuples(index=False), desc="Agent Evaluating New Disease Genes", total=len(Merged_Data))
    for row in pbar:
        sample_id = row.identifier
        gene = row.geneSymbol
        pbar.set_postfix(sample=sample_id, gene=gene)
        gene_abstract_path = Path(root_path + '/GeneAbstract/' + gene + '.json')
        if Path(gene_abstract_path).exists():
            Path(LLM_Path + '/' + sample_id).mkdir(parents=True, exist_ok=True)
            save_file_txt = Path(LLM_Path + "/" + sample_id + "/" + gene + '.txt')
            save_file_json = Path(LLM_Path + "/" + sample_id + "/" + gene + '.json')
            if save_file_json.exists():
                continue
            with gene_abstract_path.open("r", encoding="utf-8") as f:gene_abstract = json.load(f)
            phenotype_text = patient_hpo.get(sample_id, "")
            lit_items = []
            for pmid, content in gene_abstract.items():
                lit_items.append({"pmid": pmid, "title": content.get("title", ""), "abstract": content.get("abstract", "")})
            prompt_text = build_prompt_Literatrue_agent(gene=gene, phenotype_text=phenotype_text, lit_items=lit_items)
            url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt_text,
                "stream": False,
                "options": {"temperature": float(TEMPERATURE)},
            }
            #As the reasoning is too long, and come with bullets, it cannot garantee a json output:
            #max_try = 3 times
            obj = None
            for attemp in range(3):
                r = requests.post(url, json=payload, timeout=600)
                r.raise_for_status()
                data = r.json()
                llm_output = data.get("response", "").strip()
                try:
                    obj = json.loads(llm_output)
                    break
                except Exception:
                    continue
            if obj is None:
                raise RuntimeError(f"Failed to parse JSON after 3 tries. Last output: {llm_output}")
            
            conclusion = obj['Conclusion'] if 'Conclusion' in obj else 'Not Fit'
            reasoning_line = obj['Reasoning'] if 'Reasoning' in obj else 'comply_error'
            final_text = f"{reasoning_line}\nConclusion: {conclusion}\n"

            Path(save_file_txt).write_text(final_text, encoding="utf-8")
            with Path(save_file_json).open("w", encoding="utf-8") as f: json.dump(obj, f)
        else:
            gene_abstract = None
            #This gene has no rare-disease related literature...
            continue

    print(f"--Phenotype Literature Agent's Work is DONE--\n")

#↑-↑-↑-Literature Agent-↑-↑-↑
#↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-↑-

