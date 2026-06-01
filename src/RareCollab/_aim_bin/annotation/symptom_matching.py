## UPDATE
#     Changes made by Chaozhong noted as #CL

from annotation.variant_models import *


def omimSymMatch(varObj, omimHPOScoreDf, inFileType):
    """
    Find OMIM symptom match score.
    
    OPTIMIZATION: expects omimHPOScoreDf to be indexed by Pheno_ID (call
    apply_curate_scores after omimHPOScoreDf.set_index('Pheno_ID', drop=False)).
    Uses O(1) dict-style lookup instead of repeated boolean filters.
    """
    simScore = 0
    if "phenotypes" in varObj.omimGeneDict.keys():
        for pheno in varObj.omimGeneDict["phenotypes"]:
            if "phenotypeMimNumber" in pheno.keys():
                PminNum = pheno["phenotypeMimNumber"]
                # Single O(1) lookup instead of three O(N) boolean filters.
                if PminNum in omimHPOScoreDf.index:
                    row = omimHPOScoreDf.loc[PminNum]
                    # If multiple rows match (duplicate Pheno_ID), take first.
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    simScore = row["Similarity_Score"]
                    disease_name = row["Disease_Name"]
                else:
                    simScore = 0
                    disease_name = None
            else:
                simScore = 0
                disease_name = None

            if simScore >= 0.2:
                varObj.symptomScore[PminNum] = simScore
                symptomList = disease_name.split(";")
                symptomList = [i.strip().upper() for i in symptomList]
                varObj.symptomName[PminNum] = symptomList
                varObj.omimSymMatchFlag = 1
            else:
                continue
    varObj.omimSymptomSimScore = simScore


def hgmdSymMatch(varObj, hgmdHPOScoreAccSortedDf, hgmdHPOScoreGeneSortedDf):
    """
    Find HGMD symptom match score
    Param:
    omimHPOScoreDf: this is read from user input OMIM symptom mathc file. For example:/six_tera/chaozhong/module_1/out/UDN/HPOsimi_UDNUDN630665_simi_0.tsv
    inFileType:the type of input file. Will be used later.
    Return:
    Calculate the variant symptom score
    """

    # print('\nin HGMDSymMatch')
    # print('\tvar:', varObj.varId_dash)
    hgmdSymptomSimScore = "-"
    if varObj.hgmd_id in hgmdHPOScoreAccSortedDf.index:
        varScore = hgmdHPOScoreAccSortedDf.loc[varObj.hgmd_id].Similarity_Score

        varObj.hgmdSymptomScore = varScore
        if varScore >= 0.2:
            varObj.hgmdSymMatchFlag = 1
        hgmdSymptomSimScore = varScore
    elif varObj.hgmdGeneFound:
        geneScore = hgmdHPOScoreGeneSortedDf.loc[varObj.geneSymbol].Similarity_Score

        if geneScore >= 0.2:
            varObj.hgmdSymMatchFlag = 1
        hgmdSymptomSimScore = geneScore

    varObj.hgmdSymptomSimScore = hgmdSymptomSimScore
    # print('hgmdSymMatch results:')
    # print('\thgmdSymMatchFlag:', varObj.hgmdSymMatchFlag)
    # print('\thgmdSymptomSimScore:', varObj.hgmdSymptomSimScore)


def clinVarSymMatch(varObj, inFileType):
    """
    Find clinvar symptom match score
    Param:
    Return:
    Calculate the variant symptom score
    """
    # print('\nin clinvarSymMatch')
    if varObj.clinVarVarFound:
        # pheno = varObj.clinVarVarDict['condition'].strip().upper()
        # print('clinvar condition:', varObj.clinvarCondition)
        if type(varObj.clinvarCondition) is str:
            pheno = varObj.clinvarCondition.strip().upper()
            for PminNum in varObj.symptomName.keys():
                if (
                    pheno in varObj.symptomName[PminNum]
                    or "#%s %s" % (PminNum, pheno) in varObj.symptomName[PminNum]
                ):
                    varObj.clinVarSymMatchFlag = 1
        else:
            varObj.clinVarSymMatchFlag = 0

    # Need more checking on this one per gene condition
    if varObj.clinVarGeneFound:
        for var in varObj.clinVarGeneDict:
            pheno = var["condition"].strip().upper()
            for PminNum in varObj.symptomName.keys():
                if (
                    pheno in varObj.symptomName[PminNum]
                    or "#%s %s" % (PminNum, pheno) in varObj.symptomName[PminNum]
                ):
                    varObj.clinVarGeneSymMatchFlag = 1

    # print('\tvarObj.clinVarVarSymMatchFlag:', varObj.clinVarSymMatchFlag)
