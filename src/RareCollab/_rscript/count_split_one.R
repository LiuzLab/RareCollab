#!/usr/bin/env Rscript
#
# count_split_one.R -- count split reads for ONE sample, then exit.
#
# Run as a separate process per sample rather than through countRNAData's
# bplapply. R's heap grows but never shrinks, so a worker that handles several
# samples in sequence ends up holding the high-water mark of the most expensive
# one plus accumulated fragmentation: a single worker was observed at 178 GB on
# a cohort whose largest BAM is 18 GB. A process that exits after one sample
# returns all of it to the OS, capping peak memory at one sample's worth and
# making it a function of the largest BAM rather than of the cohort.
#
# Also isolates failures - a BAM that kills the process no longer takes the
# whole batch with it - and gives honest per-sample progress.
#
# Usage:
#   Rscript count_split_one.R <sample_anno.tsv> <working_dir> <analysis_name> \
#                             <sampleID> <keep_scaffolds:TRUE|FALSE>

suppressPackageStartupMessages({
    library(FRASER)
    library(data.table)
    library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
    stop("Expected 5 arguments: anno, working_dir, analysis_name, sampleID, ",
         "keep_scaffolds")
}
anno_path       <- args[1]
working_dir     <- args[2]
analysis_name   <- args[3]
sample_id       <- args[4]
keep_scaffolds  <- as.logical(args[5])

# Nothing to parallelise across inside a single sample's count; the outer
# process pool provides the concurrency, and letting this fork as well would
# multiply memory in exactly the way this script exists to avoid.
register(SerialParam())

anno <- fread(anno_path)
anno[, strand := as.integer(strand)]
anno[, pairedEnd := as.logical(pairedEnd)]

if (!sample_id %in% anno$sampleID) {
    stop("Sample not present in the annotation: ", sample_id)
}

fds <- FraserDataSet(colData = anno, workingDir = working_dir)
name(fds) <- analysis_name

counts <- countSplitReads(
    sampleID = sample_id,
    fds = fds,
    NcpuPerSample = 1,
    keepNonStandardChromosomes = keep_scaffolds
)

cat(sprintf("OK %s %d junctions\n", sample_id, length(counts)))