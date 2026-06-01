import re
import pandas as pd

from .variant_models import Variant
from .clinical_flatfile_sources import (
    getClinVarUsingMarrvelFlatFile,
    getHGMDUsingFlatFile,
)


def getAnnotateInfoRow_3_1(row, genomeRef):
    varObj = Variant()
    transcriptId = row.Feature
    optFlag = 0
    if row.iloc[0].find("/") != -1:
        optFlag = 1

    if optFlag == 0:
        s = row.iloc[0].split("_")
        chrom = s[0]
        pos = int(s[1])
        ref = s[2]
        alt = s[3]
    elif optFlag == 1:
        s = row.iloc[0].split("_")
        chrom = s[0]
        pos = int(s[1])
        s = s[2].split("/")
        ref = s[0]
        alt = s[1]

    if "-" in row.iloc[1]:
        s = row.iloc[1].split(":")
        tmp = s[1]
        s = tmp.split("-")
        start = int(s[0])
        stop = int(s[1])
    else:
        s = row.iloc[1].split(":")
        start = int(s[1])
        stop = int(s[1])

    if chrom == "X":
        chrom = 23
    elif chrom == "Y":
        chrom = 24
    elif chrom == "MT":
        chrom = 25
    elif re.search(r"GL", chrom):
        chrom = 26
    chrom = int(chrom)

    if genomeRef == "hg38":
        varObj.hg38Chrom = chrom
        varObj.hg38Pos = pos
        varObj.chrom = chrom
        varObj.pos = pos
        varObj.start = start
        varObj.stop = stop
    else:
        varObj.hg19Chrom = chrom
        varObj.hg19Pos = pos
        varObj.chrom = chrom
        varObj.pos = pos
        varObj.start = start
        varObj.stop = stop

    geneSymbol = row.SYMBOL
    varObj.geneSymbol = geneSymbol
    varObj.CADD_phred = row.CADD_phred
    varObj.CADD_PHRED = row.CADD_PHRED

    varObj.ref = ref
    varObj.alt = alt
    varObj.varId_dash = "-".join([str(chrom), str(start), ref, alt])
    varId = "_".join([str(chrom), str(pos), ref, alt, transcriptId])
    varObj.varId = varId
    if "ZYG" in row:
        varObj.zyg = row.ZYG
    varObj.geneEnsId = row.Gene
    varObj.rsId = row.Existing_variation
    varObj.GERPpp_RS = row.GERPpp_RS
    varObj.featureType = row.Feature_type
    varObj.gnomadAF = row.gnomAD_AF
    varObj.gnomadAFg = row.gnomADg_AF
    varObj.CLIN_SIG = row.CLIN_SIG
    varObj.LRT_Omega = row.LRT_Omega
    varObj.LRT_score = row.LRT_score
    varObj.phyloP100way_vertebrate = row.phyloP100way_vertebrate
    varObj.IMPACT = row.IMPACT
    varObj.Consequence = row.Consequence
    varObj.HGVSc = row.HGVSc
    varObj.HGVSp = row.HGVSp
    varObj.GERPpp_NR = row.GERPpp_NR
    varObj.DANN_score = row.DANN_score
    varObj.FATHMM_pred = row.FATHMM_pred
    varObj.FATHMM_score = row.FATHMM_score
    varObj.GTEx_V8_gene = row.GTEx_V8_gene
    varObj.GTEx_V8_tissue = row.GTEx_V8_tissue
    varObj.Polyphen2_HDIV_score = row.Polyphen2_HDIV_score
    varObj.Polyphen2_HVAR_score = row.Polyphen2_HVAR_score
    varObj.REVEL_score = row.REVEL_score
    varObj.SIFT_score = row.SIFT_score

    varObj.clinvar_AlleleID = row.clinvar
    varObj.clinvar_clnsig = row.clinvar_CLNSIG
    varObj.clinvar_CLNREVSTAT = row.clinvar_CLNREVSTAT
    varObj.clinvar_CLNSIGCONF = row.clinvar_CLNSIGCONF
    varObj.clin_code = row.clinvar_CLNSIG

    varObj.fathmm_MKL_coding_score = row.fathmm_MKL_coding_score
    varObj.LRT_score = row.LRT_score
    varObj.LRT_Omega = row.LRT_Omega
    varObj.phyloP100way_vertebrate = row.phyloP100way_vertebrate
    varObj.M_CAP_score = row.M_CAP_score
    varObj.MutationAssessor_score = row.MutationAssessor_score
    varObj.MutationTaster_score = row.MutationTaster_score
    varObj.ESP6500_AA_AC = row.ESP6500_AA_AC
    varObj.ESP6500_AA_AF = row.ESP6500_AA_AF
    varObj.ESP6500_EA_AC = row.ESP6500_EA_AC
    varObj.ESP6500_EA_AF = row.ESP6500_EA_AF

    varObj.VARIANT_CLASS = row.VARIANT_CLASS
    varObj.Feature = row.Feature
    varObj.hom = row.gnomADg_controls_nhomalt
    varObj.hgmd_id = row.hgmd
    varObj.hgmd_symbol = row.hgmd_GENE
    varObj.hgmd_rs = row.hgmd_RANKSCORE
    varObj.hgmd_PHEN = row.hgmd_PHEN
    varObj.hgmd_CLASS = row.hgmd_CLASS

    if row.clinvar_CLNSIGCONF != "-":
        clin_dict = dict()
        for ro in row.clinvar_CLNSIGCONF.split("|_"):
            temp = ro.split("(")
            clin_dict[temp[0]] = int(temp[1][0])
        PLP_sum = clin_dict.get("Pathogenic", 0) + clin_dict.get(
            "Likely_pathogenic", 0
        )
        varObj.clin_dict = clin_dict
        varObj.clin_PLP = PLP_sum
        varObj.clin_PLP_perc = PLP_sum / sum(clin_dict.values())
    else:
        if "benign" in row.clinvar_clnsig.lower():
            varObj.clin_PLP_perc = 0
        elif "pathogenic" in row.clinvar_clnsig.lower():
            varObj.clin_PLP_perc = 1
        else:
            varObj.clin_PLP_perc = "-"
        varObj.clin_PLP = "-"
        varObj.clin_dict = "-"

    if row.SpliceAI_pred != "-":
        varObj.spliceAI = row.SpliceAI_pred
        temp = row.SpliceAI_pred.split("|")
        varObj.spliceAImax = max(
            float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4])
        )
    else:
        varObj.spliceAI = "-"
        varObj.spliceAImax = "-"

    # OPTIMIZATION: return the varObj itself (not vars(varObj)) so downstream
    # f2-f6 can access fields via attribute access, the same way they always have.
    # We dict-ify only at the very end, once per row, instead of once per f-step.
    return varObj


def getAnnotateInfoRow_3_2(varObj, decipherSortedDf):
    # get decipher: 0.6s
    decipherDictList = []
    decipherDeletionObsList = []
    decipherStudyList = []
    decipherVarFound = 0
    deletionObs = "-"
    chromVal = int(varObj.chrom)
    posVal = int(varObj.pos)
    startVal = int(varObj.start)
    stopVal = int(varObj.stop)

    if (chromVal, startVal, stopVal) in decipherSortedDf:
        vals = decipherSortedDf.loc[(chromVal, startVal, stopVal)]
        decipherVarFound = 1
        deletionObs = vals.iloc[0]["deletion.obs"]
        decipherDeletionObsList.append(deletionObs)

    retList = [decipherDictList, decipherDeletionObsList, decipherStudyList, decipherVarFound]
    return {
        "decipherDictList": retList[0],
        "decipherDeletionObsList": retList[1],
        "decipherStudyList": retList[2],
        "decipherVarFound": retList[3],
    }


def getAnnotateInfoRow_3_3(varObj, gnomadMetricsGeneSortedDf):
    if varObj.geneSymbol in gnomadMetricsGeneSortedDf.index:
        val = gnomadMetricsGeneSortedDf.loc[varObj.geneSymbol]
        gnomadGeneZscore = val["mis_z"]
        gnomadGenePLI = val["pLI"]
        gnomadGeneOELof = val["oe_lof"]
        gnomadGeneOELofUpper = val["oe_lof_upper"]
    else:
        gnomadGeneZscore = "-"
        gnomadGenePLI = "-"
        gnomadGeneOELof = "-"
        gnomadGeneOELofUpper = "-"

    retList = [gnomadGeneZscore, gnomadGenePLI, gnomadGeneOELof, gnomadGeneOELofUpper]
    return {
        "gnomadGeneZscore": retList[0],
        "gnomadGenePLI": retList[1],
        "gnomadGeneOELof": retList[2],
        "gnomadGeneOELofUpper": retList[3],
    }


def getAnnotateInfoRow_3_4(varObj, omimGeneSortedDf):
    inputSnpList = []
    if "," in varObj.rsId:
        inputSnpList = varObj.rsId.split(",")
    else:
        inputSnpList = varObj.rsId
    varFound = 0
    geneFound = 0
    omimDict = {}
    omimGeneDict = {}
    omimAlleleDict = {}
    phenoList = []
    phenoInhList = []
    phenoMimList = []

    if varObj.geneSymbol in omimGeneSortedDf.index:
        geneFound = 1
        omimGeneDict = omimGeneSortedDf.loc[varObj.geneSymbol]
        snpList = []
        for a in omimGeneDict["allelicVariants"]:
            if "dbSnps" in a:
                snpList.append(a["dbSnps"])
        set1 = set(inputSnpList)
        set2 = set(snpList)
        if set1.intersection(set2):
            varFound = 1
        else:
            varFound = 0

        for a in omimGeneDict["phenotypes"]:
            pheno = a["phenotype"]
            if "phenotypeMimNumber" in a:
                phenoMim = a["phenotypeMimNumber"]
            else:
                phenoMim = "-"
            if "phenotypeInheritance" in a:
                phenoInh = a["phenotypeInheritance"]
            else:
                phenoInh = "-"
            phenoList.append(pheno)
            phenoInhList.append(phenoInh)
            phenoMimList.append(str(phenoMim))

    omimRet = [
        varFound, geneFound, omimDict, omimGeneDict, omimAlleleDict,
        phenoList, phenoInhList, phenoMimList,
    ]
    return {
        "omimVarFound": omimRet[0],
        "omimGeneFound": omimRet[1],
        "omimDict": omimRet[2],
        "omimGeneDict": omimRet[3],
        "omimAlleleDict": omimRet[4],
        "phenoList": omimRet[5],
        "phenoInhList": omimRet[6],
        "phenoMimList": omimRet[7],
    }


def getAnnotateInfoRow_3_5(varObj, clinvarGeneDf, clinvarAlleleDf):
    clinVarRet = getClinVarUsingMarrvelFlatFile(varObj, clinvarAlleleDf, clinvarGeneDf)
    clinVarRet[10] = varObj.clinvar_clnsig

    return {
        "clinVarVarFound": clinVarRet[0],
        "clinVarVarDict": clinVarRet[1],
        "clinVarGeneFound": clinVarRet[2],
        "clinVarGeneDict": clinVarRet[3],
        "clinvarTotalNumVars": clinVarRet[4],
        "clinvarNumP": clinVarRet[5],
        "clinvarNumLP": clinVarRet[6],
        "clinvarNumLB": clinVarRet[7],
        "clinvarNumB": clinVarRet[8],
        "clinvarTitle": clinVarRet[9],
        "clinvarSignDesc": clinVarRet[10],
        "clinvarCondition": clinVarRet[11],
    }


def getAnnotateInfoRow_3_6(varObj, hgmdHPOScoreGeneSortedDf):
    hgmdRet = getHGMDUsingFlatFile(varObj, hgmdHPOScoreGeneSortedDf)
    return {
        "hgmdVarFound": hgmdRet[0],
        "hgmdGeneFound": hgmdRet[1],
        "hgmdVarPhenIdList": hgmdRet[2],
        "hgmdVarHPOIdList": hgmdRet[3],
        "hgmdVarHPOStrList": hgmdRet[4],
    }


def getAnnotateInfoRows_3(
        vepDf,
        genomeRef,
        clinvarGeneDf,
        clinvarAlleleDf,
        omimGeneSortedDf,
        omimAlleleList,
        hgmdHPOScoreGeneSortedDf,
        decipherSortedDf,
        gnomadMetricsGeneSortedDf,
):
    """
    OPTIMIZATION: iterate vepDf ONCE. For each row, call f1 to build varObj,
    then call f2-f6 on that varObj (avoiding 5 extra full-DataFrame apply
    sweeps). Behavior is identical to the original 6-pass version: each f_i
    produces the same dict it always did, and the merged row dict is
    assembled in the same order, so column order in the output DataFrame
    matches f1's columns first then f2..f6.
    """
    #print("=== USING SINGLE-PASS getAnnotateInfoRows_3 (NEW VERSION) ===")

    results = []
    for _, row in vepDf.iterrows():
        # f1 returns a Variant object (NOT vars(varObj) dict) so f2-f6
        # can use the same attribute-access pattern they were written for.
        varObj = getAnnotateInfoRow_3_1(row, genomeRef)

        # The "f1 output dict" — same content as the legacy first apply call.
        row_dict = vars(varObj).copy()

        # f2..f6 each return a small dict; merge into row_dict.
        row_dict.update(getAnnotateInfoRow_3_2(varObj, decipherSortedDf))
        row_dict.update(getAnnotateInfoRow_3_3(varObj, gnomadMetricsGeneSortedDf))
        row_dict.update(getAnnotateInfoRow_3_4(varObj, omimGeneSortedDf))
        row_dict.update(getAnnotateInfoRow_3_5(varObj, clinvarGeneDf, clinvarAlleleDf))
        row_dict.update(getAnnotateInfoRow_3_6(varObj, hgmdHPOScoreGeneSortedDf))

        results.append(row_dict)

    return pd.DataFrame(results)