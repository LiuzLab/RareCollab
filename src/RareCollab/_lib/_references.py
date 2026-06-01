"""
Reference dataclasses for RareCollab.

Defined in a module file (not in the notebook) so that
ProcessPoolExecutor's 'spawn' worker processes can pickle/unpickle
AimReferences via standard import.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SingularityImages:
    vep: Path


@dataclass(frozen=True)
class PhrankReferences:
    dagfile: Path
    disease_annotation: Path
    gene_annotation: Path
    disease_gene: Path


@dataclass(frozen=True)
class OmimReferences:
    omim_hgmd_phen: Path
    omim_obo: Path
    omim_genemap2: Path
    omim_pheno: Path


@dataclass(frozen=True)
class GnomadReferences:
    genome_vcf: Path
    genome_tbi: Path
    exome_vcf: Path
    exome_tbi: Path


@dataclass(frozen=True)
class VepReferences:
    cache_dir: Path
    plugins_dir: Path
    custom_gnomad: Path
    custom_clinvar: Path
    custom_hgmd: Path
    plugin_revel: Path
    plugin_spliceai_snv: Path
    plugin_spliceai_indel: Path
    plugin_cadd: Path
    plugin_dbnsfp: Path
    index_files: list

@dataclass(frozen=True)
class RareCollabReferences:
    gencode_annot: Path
    mane_transcript: Path
    moe_model: Path
    clinvar_feather: Path  
    omim_disease: Path
    hpo_lib: Path
    hpo_genes: Path

@dataclass(frozen=True)
class AimReferences:
    chrmap_file: Path
    ensembl_to_location_file: Path
    ensembl_to_symbol_file: Path

    ref_annot_dir: Path
    ref_var_tier_dir: Path
    ref_merge_expand_dir: Path
    ref_mod5_diffusion_dir: Path

    phrank: PhrankReferences
    omim: OmimReferences
    gnomad: GnomadReferences
    vep: VepReferences
    rarecollab: RareCollabReferences
    aim_bin_dir: Path


@dataclass(frozen=True)
class FastaReferences:
    fasta: Path
    fai: Path
    dict: Path