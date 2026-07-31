#!/usr/bin/env Rscript
#
# run_outrider.R -- the R half of RareCollab's OUTRIDER step.
#
# Per-sample gene counts are produced beforehand by count_expression_one.R,
# one process each; this script reads them back, fits one model across the
# whole cohort, and writes one table per case.
#
# Like FRASER, OUTRIDER is a cohort method: the autoencoder learns what normal
# expression looks like across samples, so a single sample has nothing to be an
# outlier against.
#
# Usage:
#   Rscript run_outrider.R <sample_anno.tsv> <cache_dir> <out_dir> \
#                          <n_threads> <fpkm_cutoff>

suppressPackageStartupMessages({
    library(OUTRIDER)
    library(data.table)
    library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
    stop("Expected 5 arguments: anno, cache_dir, out_dir, n_threads, fpkm_cutoff")
}
anno_path   <- args[1]
cache_dir   <- args[2]
out_dir     <- args[3]
n_threads   <- as.integer(args[4])
fpkm_cutoff <- as.numeric(args[5])

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
register(MulticoreParam(workers = n_threads))

say <- function(...) message(format(Sys.time(), "[%H:%M:%S] "), ...)


# ---------------------------------------------------------------------------
# Assemble the count matrix
# ---------------------------------------------------------------------------

anno <- fread(anno_path)
is_case <- as.logical(anno$isCase)
case_ids <- anno$sampleID[is_case]
say(nrow(anno), " sample(s): ", sum(is_case), " case, ", sum(!is_case),
    " background")

say("Reading per-sample gene counts from ", cache_dir)
per_sample <- lapply(anno$sampleID, function(sid) {
    f <- file.path(cache_dir, paste0("geneCounts-", sid, ".tsv.gz"))
    if (!file.exists(f)) stop("Missing gene counts for sample: ", sid)
    fread(f)
})

# Every sample was counted against the same SAF, so the gene sets must match
# exactly. If they do not, the caches were built from different annotations and
# silently merging them would misalign genes across samples.
gene_ids <- per_sample[[1]]$GeneID
for (i in seq_along(per_sample)) {
    if (!identical(per_sample[[i]]$GeneID, gene_ids)) {
        stop("Sample ", anno$sampleID[i], " was counted against a different ",
             "gene set. Delete the gene-count cache and re-run so every ",
             "sample uses the current annotation.")
    }
}

count_mat <- do.call(cbind, lapply(per_sample, function(dt) dt$count))
rownames(count_mat) <- gene_ids
colnames(count_mat) <- anno$sampleID
gene_lengths <- per_sample[[1]]$length

say("  ", nrow(count_mat), " genes x ", ncol(count_mat), " samples")

ods <- OutriderDataSet(countData = count_mat, colData = as.data.frame(anno))
# filterExpression works in FPKM, which needs a length per gene. featureCounts
# already reports one - the summed width of the intervals sharing a GeneID -
# so no second annotation pass is needed.
mcols(ods)$basepairs <- gene_lengths


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

# This filter is on expression, not significance. It has to happen: a gene with
# no reads anywhere has no negative-binomial model to fit. It is also safe for
# the case that matters most - a gene switched off in one patient - because a
# gene passes when its 95th percentile across samples clears the cutoff, so
# something expressed in the controls is kept regardless of the patient.
say("Filtering genes below ", fpkm_cutoff, " FPKM")
ods <- filterExpression(ods, fpkmCutoff = fpkm_cutoff, filterGenes = TRUE)
say("  ", nrow(ods), " gene(s) passed")
if (nrow(ods) < 100) {
    stop("Only ", nrow(ods), " genes passed the expression filter. Either the ",
         "annotation does not match the BAMs, or the libraries are far ",
         "shallower than expected.")
}

say("Estimating latent space dimension")
# estimateBestQ gained the Optimal Hard Thresholding shortcut in a later
# release; fall back to a grid search where this OUTRIDER predates it. The grid
# stays well under the sample count because q cannot exceed it, and a small
# cohort overfits quickly.
q_grid <- seq(2, max(2, min(10, ncol(ods) - 2)), 2)
ods <- tryCatch(
    estimateBestQ(ods, useOHT = TRUE),
    error = function(e) {
        say("  OHT unavailable (", conditionMessage(e), "); using a grid search")
        estimateBestQ(ods, useOHT = FALSE, params = q_grid)
    }
)
q <- getBestQ(ods)
say("  q = ", q)

say("Fitting the expression model")
ods <- OUTRIDER(ods, q = q, BPPARAM = MulticoreParam(workers = n_threads))

say("Saving OutriderDataSet")
saveRDS(ods, file.path(out_dir, "ods-object.RDS"))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
#
# Every surviving gene is written, not just the significant ones. Downstream
# the MoE weighs evidence rather than consuming a verdict, so a p-value that
# happens to be unremarkable is still worth more than a missing row. At ~8k
# genes per case the cost of keeping everything is about a megabyte.
#
# One file per case, never a combined table: holding every sample at once here,
# again in the file and a third time when Python reads it back, is fine for
# five patients and fatal for a few hundred.
#
# Column names are fixed by Preprocessing.RNA(). Note GeneSymbol rather than
# the hgnc_symbol the splicing table uses - the two genuinely differ, so they
# are matched as-is rather than harmonised.

say("Building the full results table")
res_all <- as.data.table(results(ods, all = TRUE))
setnames(res_all, "geneID", "GeneSymbol")

needed <- c("sampleID", "GeneSymbol", "pValue", "padjust",
            "zScore", "l2fc", "rawcounts")
missing <- setdiff(needed, names(res_all))
if (length(missing) > 0) {
    stop("OUTRIDER results are missing expected column(s): ",
         paste(missing, collapse = ", "))
}

say("Writing per-sample expression tables for ", length(case_ids), " sample(s)")
for (sid in case_ids) {
    dt <- res_all[sampleID == sid]
    sample_dir <- file.path(dirname(out_dir), sid)
    dir.create(sample_dir, showWarnings = FALSE, recursive = TRUE)
    fwrite(dt, file.path(sample_dir, "expression.tsv.gz"), sep = "\t")
}

say("Done")