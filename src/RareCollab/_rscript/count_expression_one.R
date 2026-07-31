#!/usr/bin/env Rscript
#
# count_expression_one.R -- count reads per gene for ONE sample, then exit.
#
# Run as a separate process per sample, matching count_split_one.R. Unlike
# FRASER's split-read counting this is not memory-hungry - featureCounts is a C
# implementation that streams the BAM rather than materialising alignments as R
# objects - but the isolation is kept anyway: a sample that fails then costs
# one sample rather than the batch, and the exit status says what happened.
#
# Gene counts have a property FRASER's non-split counts do not: they depend
# only on the BAM and the annotation, never on cohort membership. Adding or
# removing a patient never invalidates them, so this cache survives everything.
#
# Usage:
#   Rscript count_expression_one.R <sample_anno.tsv> <gene_saf.tsv> \
#                                  <cache_dir> <sampleID> <n_threads> <tmp_dir>

suppressPackageStartupMessages({
    library(data.table)
    library(Rsubread)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
    stop("Expected 6 arguments: anno, gene_saf, cache_dir, sampleID, ",
         "n_threads, tmp_dir")
}
anno_path  <- args[1]
saf_path   <- args[2]
cache_dir  <- args[3]
sample_id  <- args[4]
n_threads  <- as.integer(args[5])
tmp_dir    <- args[6]

anno <- fread(anno_path)
row <- anno[sampleID == sample_id]
if (nrow(row) != 1) {
    stop("Sample not found exactly once in the annotation: ", sample_id)
}

saf <- fread(saf_path)
required <- c("GeneID", "Chr", "Start", "End", "Strand")
if (!all(required %in% names(saf))) {
    stop("Annotation is not in SAF format; needs: ",
         paste(required, collapse = ", "))
}

dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)
out_file <- file.path(cache_dir, paste0("geneCounts-", sample_id, ".tsv.gz"))

# featureCounts uses the same strand encoding FRASER does - 0 unstranded,
# 1 forward, 2 reverse - so the column passes through unchanged.
res <- featureCounts(
    files = row$bamFile,
    annot.ext = as.data.frame(saf),
    isGTFAnnotationFile = FALSE,
    isPairedEnd = as.logical(row$pairedEnd),
    strandSpecific = as.integer(row$strand),

    # Rows sharing a GeneID are summed, so this works unchanged whether the SAF
    # holds one interval per gene or one per exon.
    useMetaFeatures = TRUE,

    # A read landing in two annotated genes is ambiguous rather than evidence
    # for both; counting it twice would inflate exactly the overlapping loci
    # that are hardest to interpret.
    allowMultiOverlap = FALSE,

    countMultiMappingReads = FALSE,
    nthreads = n_threads,
    verbose = FALSE,

    # Rsubread scatters scratch files into the working directory unless told
    # otherwise, and a subprocess launched from a notebook inherits the user's
    # source tree as its working directory. FRASER's own featureCounts call
    # sets this for exactly the same reason.
    tmpDir = normalizePath(tmp_dir)
)

counts <- data.table(
    GeneID = rownames(res$counts),
    count  = as.integer(res$counts[, 1]),
    length = as.integer(res$annotation$Length)
)

fwrite(counts, out_file, sep = "\t")

assigned <- res$stat[res$stat$Status == "Assigned", 2]
total <- sum(res$stat[, 2])
cat(sprintf("OK %s %d genes, %d/%d reads assigned (%.1f%%)\n",
            sample_id, nrow(counts), assigned, total,
            100 * assigned / max(total, 1)))