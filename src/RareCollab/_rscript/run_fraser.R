#!/usr/bin/env Rscript
#
# run_fraser.R -- the R half of RareCollab's FRASER 2.0 step.
#
# Written into the cohort work folder by Features.RunFRASER() and invoked with
# Rscript. Kept free of hard-coded paths so the copy left behind in the work
# folder can be re-run by hand for debugging.
#
# Split-read counting is NOT done here - count_split_one.R handles that, one
# process per sample, because R never returns heap to the OS and a long-lived
# worker accumulates the peak of every sample it has touched.
#
# Usage:
#   Rscript run_fraser.R <sample_anno.tsv> <gene_ranges.tsv> <working_dir> \
#                        <out_dir> <n_threads> <analysis_name>

suppressPackageStartupMessages({
    library(FRASER)
    library(data.table)
    library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
    stop("Expected 6 arguments: anno, gene_ranges, working_dir, out_dir, ",
         "n_threads, analysis_name")
}
anno_path     <- args[1]
gene_ranges   <- args[2]
working_dir   <- args[3]
out_dir       <- args[4]
n_threads     <- as.integer(args[5])
analysis_name <- args[6]

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Counting runs serially. Split reads are already cached by count_split_one.R,
# so this pass only reads them back; non-split counting still happens in this
# process, and serialising it keeps the peak at one sample rather than letting
# an ever-growing R heap accumulate several high-water marks. The fit further
# down re-registers a parallel backend, being CPU-bound but memory-cheap.
register(SerialParam())

say <- function(...) message(format(Sys.time(), "[%H:%M:%S] "), ...)


# ---------------------------------------------------------------------------
# Gene symbol annotation
# ---------------------------------------------------------------------------

annotateFromGencode <- function(fds, tsv_path) {
    # Annotate from the same GENCODE build the DNA arm uses, rather than from a
    # Bioconductor annotation package. Two reasons: it avoids depending on a
    # package whose serialised format is tied to the user's R version, and more
    # importantly it keeps gene symbols byte-identical across the DNA and RNA
    # arms. Downstream the MoE links RNA evidence to DNA candidates by symbol,
    # and a vocabulary mismatch there fails silently rather than loudly.
    genes <- fread(tsv_path)
    anno <- makeGRangesFromDataFrame(genes, keep.extra.columns = TRUE)
    seqlevelsStyle(anno) <- seqlevelsStyle(fds)[1]

    for (type in c("j", "theta")) {
        gr <- rowRanges(fds, type = type)
        anno_t <- anno
        if (any(strand(gr) == "*")) strand(anno_t) <- "*"

        suppressWarnings(hits <- findOverlaps(gr, anno_t))
        dt <- data.table(
            from    = from(hits),
            feature = mcols(anno_t[to(hits)])[["gene_name"]]
        )
        missing <- setdiff(seq_along(gr), unique(from(hits)))
        if (length(missing) > 0) {
            dt <- rbind(dt, data.table(from = missing, feature = NA_character_))
        }
        # A junction spanning several genes keeps all of them, ';'-joined,
        # which is what FRASER's own annotateRanges produces.
        dt <- dt[, .(feature = paste(unique(feature), collapse = ";")),
                 by = "from"][order(from)]
        mcols(fds, type = type)[["hgnc_symbol"]] <- dt$feature
    }
    fds
}


# ---------------------------------------------------------------------------
# Build the dataset
# ---------------------------------------------------------------------------

say("Reading sample annotation: ", anno_path)
anno <- fread(anno_path)

# FRASER's validity checks are strict about these two: 'strand' must be integer
# 0/1/2 and 'pairedEnd' must be logical. Either slips through fread as the
# wrong type easily and then fails obscurely inside FraserDataSet().
anno[, strand := as.integer(strand)]
anno[, pairedEnd := as.logical(pairedEnd)]

is_case <- anno$isCase
case_ids <- anno$sampleID[is_case]
say(nrow(anno), " sample(s): ", sum(is_case), " case, ", sum(!is_case),
    " background")

fds <- FraserDataSet(colData = anno, workingDir = working_dir)
name(fds) <- analysis_name

say("Assembling counts (split reads come from cache; non-split runs now)")
# keepNonStandardChromosomes=FALSE drops decoys, alts and unplaced contigs.
# They carry no interpretable splicing for a rare-disease readout and inflate
# both the junction map and memory. FRASER re-applies this when reading a
# cached count, so caches built either way stay consistent.
fds <- countRNAData(fds, keepNonStandardChromosomes = FALSE)

say("Computing PSI and Intron Jaccard Index")
fds <- calculatePSIValues(fds)

say("Filtering introns")
fds <- filterExpressionAndVariability(fds,
        minExpressionInOneSample = 20, minDeltaPsi = 0, filter = TRUE)
say("  ", nrow(fds), " intron(s) passed")

# Annotation has to precede the fit: calculatePadjValues, which FRASER() calls
# internally, needs symbols in place to produce gene-level p-values.
say("Annotating gene symbols from ", basename(gene_ranges))
fds <- annotateFromGencode(fds, gene_ranges)
n_named <- sum(!is.na(mcols(fds, type = "j")$hgnc_symbol))
say("  ", n_named, "/", nrow(fds), " intron(s) matched a protein-coding gene")
if (n_named == 0) {
    stop("No intron overlapped any gene. This almost always means the BAM and ",
         "the annotation disagree on chromosome naming or genome build.")
}

# The fit is CPU-bound but memory-cheap, unlike counting, so give it the cores.
register(MulticoreParam(workers = n_threads))

say("Estimating latent space dimension")
fds <- estimateBestQ(fds, type = "jaccard", plot = FALSE)
q <- bestQ(fds, type = "jaccard")
say("  q = ", q)

say("Fitting the splicing model")
fds <- FRASER(fds, q = c(jaccard = q))

say("Saving FraserDataSet")
saveFraserDataSet(fds, dir = working_dir, name = analysis_name)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
#
# Written one sample at a time, never combined. A single table would hold every
# case in memory here, again in the file, and a third time when Python reads it
# back - fine for five patients, fatal for a few hundred. Per-sample files also
# let one patient's results be opened on their own.
#
# Only cases are exported. Background controls exist to give the autoencoder
# enough samples to fit; carrying them through would multiply the output and
# make Preprocessing.RNA() emit per-sample files for controls, polluting the
# downstream MoE.
#
# Column names are fixed by Preprocessing.RNA(): they are FRASER assay names,
# not the columns results() emits. The results table alongside is the readable
# one.

say("Writing per-sample junction tables for ", length(case_ids), " sample(s)")

gr <- rowRanges(fds, type = "j")
base <- data.table(
    seqnames    = as.character(seqnames(gr)),
    start       = start(gr),
    end         = end(gr),
    strand      = as.character(strand(gr)),
    hgnc_symbol = mcols(gr)$hgnc_symbol
)

plain_assays <- c("psi5", "psi3", "jaccard",
                  "delta_psi5", "delta_psi3", "delta_jaccard",
                  "rawOtherCounts_psi5", "rawOtherCounts_psi3",
                  "rawOtherCounts_jaccard", "rawCountsJnonsplit")

missing_assays <- setdiff(plain_assays, assayNames(fds))
if (length(missing_assays) > 0) {
    stop("FRASER did not produce expected assay(s): ",
         paste(missing_assays, collapse = ", "))
}

# These two go through the accessors rather than by assay name: the p-value
# assay picks up a suffix when rho filtering is in play, so a hardcoded name
# would silently come back empty.
pvals <- pVals(fds, type = "jaccard")
mus   <- predictedMeans(fds, type = "jaccard")

for (sid in case_ids) {
    dt <- copy(base)
    dt[, sampleID := sid]
    for (a in plain_assays) {
        dt[[a]] <- as.vector(assay(fds, a)[, sid])
    }
    dt[["pvaluesBetaBinomial_jaccard"]] <- as.vector(pvals[, sid])
    dt[["predictedMeans_jaccard"]]      <- as.vector(mus[, sid])

    sample_dir <- file.path(dirname(out_dir), sid)
    dir.create(sample_dir, showWarnings = FALSE, recursive = TRUE)
    fwrite(dt, file.path(sample_dir, "junctions.tsv.gz"), sep = "\t")
    rm(dt)
    gc()
}

say("Extracting the results table")
res <- as.data.table(results(fds, padjCutoff = 0.1, deltaPsiCutoff = 0.1))
res <- res[sampleID %in% case_ids]
res_out <- file.path(out_dir, "fraser_results.tsv")
fwrite(res, res_out, sep = "\t")
say("Wrote ", nrow(res), " significant event(s) -> ", basename(res_out))

say("Done")