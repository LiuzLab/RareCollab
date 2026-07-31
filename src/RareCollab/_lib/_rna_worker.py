"""
Machinery behind the RNA functions in Features.

Everything here is called from Features.ProcessBAM / RunFRASER / RunOutrider /
RunASE / PrepareRNAEvidence. Nothing is public API.

Unlike the other _lib workers this uses threads rather than processes: the work
is subprocess orchestration and file I/O, not Python compute, so there is no
pickling constraint and no need for ProcessPoolExecutor.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from tqdm import tqdm

from ._cpu import get_available_cpus


# ---------------------------------------------------------------------------
# Bundled resources and small shared vocabularies
# ---------------------------------------------------------------------------

def rscript_path(name):
    """
    Locate a bundled R script.

    Resolved from this file, not the working directory: the package runs from
    wherever it was installed and a notebook's cwd has nothing to do with it.
    The scripts sit beside _lib, hence the second .parent.
    """
    path = Path(__file__).resolve().parent.parent / "_rscript" / name
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled R script missing: {path}\n"
            f"  The package was built without its data files. pyproject.toml "
            f"needs:\n    [tool.setuptools.package-data]\n"
            f'    "RareCollab._rscript" = ["*.R"]'
        )
    return path


def phase_label(line):
    """Extract the label from a '[HH:MM:SS] ...' progress line, else None."""
    match = re.match(r"^\[\d{2}:\d{2}:\d{2}\] (.+)", line)
    return match.group(1).strip() if match else None


def main_chroms(style):
    """
    Autosomes and X, in the requested naming convention.

    chrY het calls are almost always mapping artefacts and mitochondrial DNA is
    not diploid, so neither carries an allelic ratio worth interpreting. Keeping
    the list short also sidesteps scaffold naming: Ensembl's KI270728.1 is
    UCSC's chr1_KI270728v1_random, and no rename bridges that.
    """
    bare = [str(i) for i in range(1, 23)] + ["X"]
    return [f"chr{c}" for c in bare] if style == "UCSC" else bare


def build_fraser_anno(cohort, with_seqlevelstyle):
    """
    Build the colData table FRASER and featureCounts both expect.

    Shared because they use the same encoding: strand as integer 0/1/2 and
    pairedEnd as an R logical. FRASER validates both types strictly and fails
    obscurely on a double or a string. Only FRASER wants SeqLevelStyle.
    """
    strand_map = {"unstranded": 0, "forward": 1, "reverse": 2}
    paired_map = {"yes": "TRUE", "no": "FALSE"}

    anno = pd.DataFrame({
        "sampleID": cohort["sampleID"].astype(str),
        "bamFile": cohort["rna_path"].astype(str),
        "pairedEnd": cohort["pairedEnd"].map(paired_map),
        "strand": cohort["strand"].map(strand_map).astype(int),
        "isCase": (cohort["sample_type"] == "case").map({True: "TRUE", False: "FALSE"}),
    })
    if with_seqlevelstyle:
        # This column is what lets FRASER reconcile 'chr1' with '1' (see
        # checkSeqLevelStyle in its helper-functions.R); without it the styles
        # stay disjoint and the model fits on non-overlapping junctions with no
        # error. 'unknown' would break seqlevelsStyle<-, so fall back to the
        # cohort majority.
        style = cohort["SeqLevelStyle"].copy()
        if (style == "unknown").any():
            known = style[style != "unknown"]
            style = style.replace("unknown", known.mode()[0] if len(known) else "UCSC")
        anno.insert(4, "SeqLevelStyle", style.values)
    return anno.reset_index(drop=True)


def rna_kinds():
    """analysis kind -> (rna_cohort column, subdirectory, samplesheet column)."""
    return {
        "splicing": ("fraser_path", "Splicing", "rna_splicing_path"),
        "expression": ("outrider_path", "Expression", "rna_expression_path"),
        "ase": ("ase_path", "ASE", "rna_ase_path"),
    }


def attach_output_column(rna_cohort, column, paths):
    """
    Copy of the cohort with `column` carrying each sample's output path.

    Every input row survives and anything without an output gets an empty
    string. A step that could not process a sample must not silently shrink the
    cohort for later steps: FRASER, OUTRIDER and ASE count different things and
    fail on different samples.
    """
    out = rna_cohort.copy()
    out[column] = out["sampleID"].astype(str).map(
        {str(k): str(v) for k, v in paths.items()}).fillna("")
    return out


def run_cmd(cmd, timeout=None):
    """Run a command and return CompletedProcess; never raises on non-zero."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def subprocess_detail(result):
    """
    Turn a failed CompletedProcess into one honest line.

    A process killed for memory produces no error text at all, and R writes
    message() to stderr, so the last stderr line is usually a progress note
    rather than a diagnosis. The exit status is the reliable signal: negative
    means a signal, -9 means SIGKILL.
    """
    if result.returncode < 0:
        sig = -result.returncode
        detail = f"killed by signal {sig}"
        return detail + (" (SIGKILL - almost certainly out of memory)" if sig == 9 else "")
    lines = [l for l in (result.stderr or "").splitlines()
             if "Error" in l or "cannot" in l]
    return lines[-1] if lines else f"exit {result.returncode}; see the log"


# ---------------------------------------------------------------------------
# ProcessBAM: indexing and library-layout detection
# ---------------------------------------------------------------------------

def recommend_bam_parallelism(n_bam, max_workers=None, threads_per_sample=None):
    """
    How many BAMs at once, and how many threads each.

    Unlike the CPU-bound steps elsewhere this is I/O bound: samtools index
    reads the whole file and the ceiling is disk bandwidth. On shared storage,
    more concurrent readers past a handful makes things slower, so the total
    stays well below the core count however large the node.
    """
    total = min(get_available_cpus(), 16)
    if max_workers is None:
        max_workers = max(1, min(n_bam, total // 2, 8))
    if threads_per_sample is None:
        # 4 is where `samtools index -@` stops helping: it reads sequentially,
        # so extra decompression threads just wait on disk.
        threads_per_sample = max(1, min(4, total // max(1, max_workers)))
    return max_workers, threads_per_sample


def bam_fingerprint(bam_path):
    """Cheap identity for a BAM, used to invalidate the cache when it changes."""
    st = Path(bam_path).stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def find_bam_index(bam_path):
    """
    An existing, non-stale index for `bam_path`, else None.

    samtools accepts both naming conventions, so both are checked. An index
    older than the BAM counts as absent: a regenerated BAM with a leftover
    index is a real failure mode and a silent one.
    """
    bam_path = Path(bam_path)
    suffix = ".crai" if bam_path.suffix == ".cram" else ".bai"
    bam_mtime = bam_path.stat().st_mtime
    for candidate in (Path(str(bam_path) + suffix), bam_path.with_suffix(suffix)):
        if candidate.exists():
            return candidate if candidate.stat().st_mtime >= bam_mtime else None
    return None


def ensure_bam_index(bam_path, sample_dir, threads, notes):
    """
    Guarantee an index sits next to the BAM path handed to FRASER.

    Indexing in place is preferred. If the BAM lives somewhere read-only -
    common for a curated control set on a shared mount - fall back to
    symlinking it into the sample's work folder and indexing there. Only the
    fallback is reported, since it changes the path FRASER will be given.

    Returns (bam_path_to_use, index_path, was_built).
    """
    bam_path = Path(bam_path).resolve()
    existing = find_bam_index(bam_path)
    if existing is not None:
        return str(bam_path), str(existing), False

    if os.access(bam_path.parent, os.W_OK):
        result = run_cmd(["samtools", "index", "-@", str(threads), str(bam_path)])
        if result.returncode == 0:
            index_path = find_bam_index(bam_path)
            if index_path is not None:
                return str(bam_path), str(index_path), True
        detail = (result.stderr.strip().splitlines()[-1]
                  if result.stderr.strip() else "unknown error")
        notes.append(f"in-place indexing failed ({detail}); indexed a symlink "
                     f"in the work folder instead")
    else:
        notes.append(f"BAM directory is not writable; indexed a symlink under "
                     f"{sample_dir} and will pass that path to FRASER")

    sample_dir.mkdir(parents=True, exist_ok=True)
    link_path = sample_dir / bam_path.name
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(bam_path)

    result = run_cmd(["samtools", "index", "-@", str(threads), str(link_path)])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to index BAM: {bam_path}\n"
                           f"  Tried in place and under {sample_dir}.\n"
                           f"  samtools said: {result.stderr.strip()}")
    index_path = find_bam_index(link_path)
    if index_path is None:
        raise RuntimeError(f"samtools index reported success but produced no "
                           f"index for {link_path}")
    return str(link_path), str(index_path), True


def read_bam_contigs(bam_path):
    """
    Parse the @SQ lines once: returns (style, [(name, length), ...]).

    Style is "UCSC" / "Ensembl" because those are the exact values
    GenomeInfoDb's seqlevelsStyle<- accepts. Every @SQ line is scanned for a
    recognisable main chromosome rather than trusting the first, which may be a
    scaffold.
    """
    result = run_cmd(["samtools", "view", "-H", str(bam_path)])
    if result.returncode != 0:
        return "unknown", []

    contigs = []
    for line in result.stdout.splitlines():
        if not line.startswith("@SQ"):
            continue
        name = length = None
        for field in line.split("\t"):
            if field.startswith("SN:"):
                name = field[3:]
            elif field.startswith("LN:"):
                try:
                    length = int(field[3:])
                except ValueError:
                    pass
        if name is not None:
            contigs.append((name, length or 0))
    if not contigs:
        return "unknown", []

    names = {n for n, _ in contigs}
    main = [str(i) for i in range(1, 23)] + ["X", "Y", "MT", "M"]
    if any(f"chr{c}" in names for c in main):
        return "UCSC", contigs
    if any(c in names for c in main):
        return "Ensembl", contigs
    return "unknown", contigs


def pick_sampling_regions(contigs, n_regions=8, window=20_000_000):
    """
    A spread of genomic windows to sample reads from.

    Reading only the head of a coordinate-sorted BAM takes every read from the
    first few megabases of one chromosome, where a single highly expressed gene
    - or one with heavy antisense transcription - can skew the strand estimate
    badly. The read count is rarely the problem; the locality is. Windows start
    a quarter of the way in, since telomeric and centromere-adjacent stretches
    are gene-poor.
    """
    def is_main(name):
        bare = name[3:] if name.lower().startswith("chr") else name
        return bare in {str(i) for i in range(1, 23)} | {"X"}

    usable = sorted([(n, L) for n, L in contigs if is_main(n) and L > 30_000_000],
                    key=lambda item: -item[1])
    regions = []
    for name, length in usable[:n_regions]:
        start = int(length * 0.25)
        regions.append(f"{name}:{start}-{min(length, start + window)}")
    return regions


def scan_bam(bam_path, regions, max_reads, target_informative):
    """
    Gather the evidence for both the pairedEnd and strand decisions.

    pairedEnd comes from FLAG 0x1 and is exact. strand comes from the XS:A: tag
    that STAR (with --outSAMstrandField intronMotif) and HISAT2 attach to
    spliced reads: XS holds the intron's strand as implied by its GT/AG
    dinucleotides, i.e. the true transcript direction, so comparing it against
    the read's own alignment strand reveals the protocol with no annotation.
    Only R1 is scored for paired data, since R2 is by construction opposite.
    """
    counters = {"n_reads": 0, "n_paired": 0, "n_informative": 0, "same": 0, "diff": 0}

    def scan_one(region, budget):
        # 0xD04 = unmapped + secondary + duplicate + supplementary.
        # -q 30 drops multi-mappers, whose XS assignments are unreliable.
        cmd = ["samtools", "view", "-F", "0xD04", "-q", "30", str(bam_path)]
        if region is not None:
            cmd.append(region)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
        seen = 0
        try:
            for line in proc.stdout:
                parts = line.split("\t", 11)
                if len(parts) < 6:
                    continue
                flag = int(parts[1])
                seen += 1
                counters["n_reads"] += 1
                is_paired = bool(flag & 1)
                if is_paired:
                    counters["n_paired"] += 1

                if "N" in parts[5] and (not is_paired or flag & 64):
                    tags = parts[11] if len(parts) > 11 else ""
                    pos = tags.find("XS:A:")
                    if pos != -1 and tags[pos + 5] in "+-":
                        counters["n_informative"] += 1
                        read_strand = "-" if flag & 16 else "+"
                        key = "same" if read_strand == tags[pos + 5] else "diff"
                        counters[key] += 1

                if seen >= budget:
                    break
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()

    targets = regions if regions else [None]
    budget = max(1, max_reads // len(targets))
    for i, region in enumerate(targets):
        scan_one(region, budget)
        # Stop once well supported, but only after a few regions, so the answer
        # never rests on a single locus.
        if i >= 2 and counters["n_informative"] >= target_informative:
            break

    # Region sampling assumes a whole-transcriptome library; a targeted panel,
    # or a BAM whose contigs we could not interpret, may yield almost nothing.
    if regions and counters["n_informative"] < 100:
        scan_one(None, max_reads)

    n, info = counters["n_reads"], counters["n_informative"]
    return {
        "n_reads": n, "n_paired": counters["n_paired"],
        "paired_fraction": (counters["n_paired"] / n) if n else None,
        "n_informative_spliced": info,
        "n_read1_matching_xs": counters["same"],
        "n_read1_opposing_xs": counters["diff"],
        "fraction_matching_xs": (counters["same"] / info) if info else None,
        "n_regions_sampled": len(targets),
    }


def classify_layout(scan, sample_id):
    """
    Turn raw counts into pairedEnd / strand calls. Returns (paired, strand, notes).

    Always returns a label; an awkward number is never a reason to abort a
    batch. Low confidence comes back as a note, so the user is told what was
    seen and can override in the samplesheet. Either label may be 'unknown'
    when there is genuinely no evidence.
    """
    notes = []
    frac = scan["paired_fraction"]
    if frac is None:
        notes.append("no alignments could be read; pairedEnd undetermined")
        paired_end = "unknown"
    else:
        paired_end = "yes" if frac >= 0.5 else "no"
        if 0.1 < frac < 0.9:
            notes.append(f"pairedEnd is unclear: {frac:.1%} of sampled reads carry "
                         f"the paired flag, suggesting the BAM mixes libraries or "
                         f"was filtered asymmetrically. Going with '{paired_end}'.")

    n_info = scan["n_informative_spliced"]
    if n_info < 100:
        notes.append(f"strand undetermined: only {n_info} spliced read(s) carried "
                     f"an XS:A: tag. STAR writes XS only with "
                     f"--outSAMstrandField intronMotif. Set 'strand' in the "
                     f"samplesheet (most human RNA-seq is 'reverse').")
        return paired_end, "unknown", notes
    if n_info < 500:
        notes.append(f"strand call rests on only {n_info} informative read(s); "
                     f"treat it as provisional.")

    # Bands are deliberately asymmetric and the uncertain zones fall to the
    # nearer *stranded* call, not to unstranded. Unstranded is a true coin flip,
    # so with a few thousand reads it sits within a percent or two of 0.5 and
    # anything past ~0.42/0.58 is many standard deviations away. A stranded
    # library leaks: antisense transcription, readthrough, imperfect dUTP and
    # non-canonical splice motifs routinely push it 10-20% off the ideal.
    p = scan["fraction_matching_xs"]
    if p >= 0.70:
        strand, confident = "forward", True
    elif p <= 0.30:
        strand, confident = "reverse", True
    elif 0.42 <= p <= 0.58:
        strand, confident = "unstranded", True
    else:
        strand, confident = ("forward" if p > 0.58 else "reverse"), False

    if not confident:
        notes.append(f"strand call is low confidence: {p:.1%} of {n_info} "
                     f"informative reads match the intron strand, between the "
                     f"expected ranges (>70% forward, <30% reverse, ~50% "
                     f"unstranded). Calling it '{strand}'; set 'strand' in the "
                     f"samplesheet to override.")
    return paired_end, strand, notes


def process_one_sample(row, out_root, overwrite, threads, max_reads, target_informative):
    """
    Index and inspect one BAM. Returns (record, notes).

    `notes` holds only what deserves attention - a low-confidence call, a
    disagreement with the samplesheet, a fallback path. Routine outcomes go
    into the record and are summarised in aggregate, so a hundred-sample cohort
    does not produce a hundred lines.
    """
    sample_id = str(row["sampleID"])
    bam_path = str(row["rna_path"])
    want_strand = str(row.get("strand", "auto") or "auto").lower()
    want_paired = str(row.get("pairedEnd", "auto") or "auto").lower()

    notes = []
    sample_dir = out_root / sample_id
    cache_file = sample_dir / "bam_info.json"
    fingerprint = bam_fingerprint(bam_path)

    if cache_file.exists() and not overwrite:
        try:
            cached = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            cached = None
        if cached is not None:
            same_bam = cached.get("bam_fingerprint") == fingerprint
            same_request = (cached.get("requested_strand") == want_strand
                            and cached.get("requested_pairedEnd") == want_paired)
            index_ok = Path(cached.get("index_path", "")).exists()
            if same_bam and same_request and index_ok:
                cached["from_cache"] = True
                return cached, notes
            if not same_bam:
                notes.append("BAM has changed since the cached run; re-ran detection")

    sample_dir.mkdir(parents=True, exist_ok=True)
    usable_bam, index_path, built = ensure_bam_index(bam_path, sample_dir, threads, notes)

    # pairedEnd is always measured: it is exact, it costs nothing on top of a
    # scan we are doing anyway, and what a user means by "paired-end
    # sequencing" is not always what ended up inside the BAM.
    seq_style, contigs = read_bam_contigs(usable_bam)
    scan = scan_bam(usable_bam, pick_sampling_regions(contigs),
                    max_reads, target_informative)
    detected_paired, detected_strand, detect_notes = classify_layout(scan, sample_id)
    notes.extend(detect_notes)

    if want_paired == "auto":
        paired_end, paired_source = detected_paired, "detected"
    elif want_paired != detected_paired:
        notes.append(f"samplesheet says pairedEnd={want_paired} but the BAM says "
                     f"{detected_paired} ({scan['paired_fraction']:.1%} of reads "
                     f"flagged paired). Using the BAM.")
        paired_end, paired_source = detected_paired, "detected (overrides samplesheet)"
    else:
        paired_end, paired_source = want_paired, "samplesheet (confirmed by BAM)"

    if want_strand != "auto":
        # The user has told us; whether detection succeeded is irrelevant.
        strand, strand_source = want_strand, "samplesheet"
        if detected_strand not in ("unknown", want_strand):
            notes.append(f"samplesheet says strand={want_strand}, BAM evidence "
                         f"suggests {detected_strand} "
                         f"({scan['fraction_matching_xs']:.1%} of "
                         f"{scan['n_informative_spliced']} informative reads match "
                         f"the intron strand). Using the samplesheet value.")
    else:
        strand, strand_source = detected_strand, "detected"

    record = {
        "sampleID": sample_id, "sample_type": row["sample_type"],
        "original_bam": str(Path(bam_path).resolve()), "rna_path": usable_bam,
        "index_path": index_path, "index_built": built,
        "strand": strand, "pairedEnd": paired_end,
        "strand_source": strand_source, "pairedEnd_source": paired_source,
        "detected_strand": detected_strand, "detected_pairedEnd": detected_paired,
        "requested_strand": want_strand, "requested_pairedEnd": want_paired,
        "SeqLevelStyle": seq_style, "n_contigs": len(contigs),
        "evidence": scan, "bam_fingerprint": fingerprint,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache_file.write_text(json.dumps(record, indent=2))
    record["from_cache"] = False
    return record, notes


# ---------------------------------------------------------------------------
# FRASER
# ---------------------------------------------------------------------------

def count_cached_samples(cohort_dir, analysis_name):
    """
    How many per-sample counting results exist, as a progress signal.

    Preferred over FRASER's own messages: BiocParallel's forked workers buffer
    message() output until a whole task chunk finishes, so a bar driven by them
    lags by tens of minutes. Globs are prefixed deliberately - the non-split
    cache stores .h5 not .RDS, and shares its directory with a
    spliceSiteCoordinates.RDS that is not a sample.
    """
    cohort_dir = Path(cohort_dir)
    split_dir = cohort_dir / "cache" / "splitCounts"
    nonsplit_dir = cohort_dir / "cache" / "nonSplicedCounts" / analysis_name
    done = 0
    if split_dir.exists():
        done += sum(1 for _ in split_dir.glob("splitCounts-*.RDS"))
    if nonsplit_dir.exists():
        done += sum(1 for _ in nonsplit_dir.glob("nonSplicedCounts-*.h5"))
    return done


def stream_r_output(proc, log, n_samples, cohort_dir, analysis_name):
    """
    Relay the R subprocess through one in-place progress bar.

    Phase markers only change the bar's description, so a multi-hour run leaves
    one line rather than a scrolling wall. Everything reaches the log; only
    genuine errors interrupt.
    """
    bar = tqdm(total=2 * n_samples, unit="sample", desc="starting", ascii=True,
               bar_format="{desc:<38} {bar} {n_fmt}/{total_fmt} [{elapsed}]")
    stop = threading.Event()

    def watch():
        while not stop.wait(10):
            done = min(count_cached_samples(cohort_dir, analysis_name), bar.total)
            if done > bar.n:
                bar.update(done - bar.n)
            else:
                bar.refresh()   # keeps the elapsed clock moving

    threading.Thread(target=watch, daemon=True).start()
    errors = []
    try:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            label = phase_label(line)
            if label:
                bar.set_description(label.split(" (")[0][:38])
            elif "Error" in line or "error:" in line:
                errors.append(line.rstrip())
                bar.write(f"  {line.rstrip()}")
    finally:
        stop.set()
        bar.set_description("finished" if not errors else "failed")
        bar.close()
    proc.wait()
    return errors


def invoke_r(cmd, log_path, n_samples, cohort_dir, analysis_name, append=False):
    """Run one attempt, streaming output to the log. Returns (rc, seconds)."""
    started = time.time()
    with open(log_path, "a" if append else "w") as log:
        log.write("\n" + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, cwd=cohort_dir)
        stream_r_output(proc, log, n_samples, cohort_dir, analysis_name)
    return proc.returncode, time.time() - started


def split_count_cache_file(cohort_dir, sample_id):
    return Path(cohort_dir) / "cache" / "splitCounts" / f"splitCounts-{sample_id}.RDS"


def count_split_one(sample_id, anno_path, cohort_dir, analysis_name,
                    keep_scaffolds, timeout):
    """
    Count split reads for one sample in its own R process.

    Returns (sample_id, ok, detail). Never raises: a failure costs one sample,
    not the batch.
    """
    cmd = ["Rscript", "--vanilla", str(rscript_path("count_split_one.R")),
           str(anno_path), str(cohort_dir), analysis_name, str(sample_id),
           "TRUE" if keep_scaffolds else "FALSE"]
    log_path = Path(cohort_dir) / "logs" / f"count-{sample_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-sample scratch, and the subprocess runs from there: anything an R
    # library writes to "." lands here instead of in whatever directory the
    # notebook was started from.
    scratch = Path(cohort_dir) / "logs" / "_scratch" / str(sample_id)
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, cwd=scratch)
    except subprocess.TimeoutExpired:
        log_path.write_text(f"timed out after {timeout}s\n")
        return sample_id, False, f"timed out after {timeout / 3600:.1f}h"

    log_path.write_text(" ".join(cmd) + "\n\n" + (result.stdout or "") + (result.stderr or ""))
    if result.returncode != 0:
        return sample_id, False, subprocess_detail(result)
    if not split_count_cache_file(cohort_dir, sample_id).exists():
        return sample_id, False, "exited cleanly but wrote no cache file"
    return sample_id, True, ""


def count_split_reads_parallel(anno, anno_path, cohort_dir, analysis_name,
                               max_workers=3, keep_scaffolds=False, timeout=6 * 3600):
    """
    Count split reads for every sample, one process each.

    Concurrency is managed here rather than by BiocParallel for two reasons.
    R's heap grows but never shrinks, so a worker handling several samples in
    sequence holds the high-water mark of the most expensive one plus
    fragmentation - one worker was seen at 178 GB on a cohort whose largest BAM
    is 18 GB, and a process that exits per sample gives all of it back. And a
    forked worker that runs out of memory reports only "wrong args for
    environment subassignment" from BiocParallel's reducer, naming neither the
    sample nor the cause, whereas a separate process has a real exit status.

    Cached samples are skipped, so this resumes cleanly.
    """
    sample_ids = list(anno["sampleID"].astype(str))
    todo = [s for s in sample_ids if not split_count_cache_file(cohort_dir, s).exists()]
    if not todo:
        print(f"Split reads: all {len(sample_ids)} sample(s) already counted.")
        return []

    print(f"Counting split reads for {len(todo)} sample(s) "
          f"({len(sample_ids) - len(todo)} already cached), "
          f"{max_workers} process(es) at a time.")

    failures = []
    bar = tqdm(total=len(todo), unit="sample", desc="split reads", ascii=True,
               bar_format="{desc:<38} {bar} {n_fmt}/{total_fmt} [{elapsed}]")
    # Without a heartbeat the bar renders once then freezes: as_completed blocks
    # until something finishes and a sample can take half an hour.
    stop = threading.Event()
    threading.Thread(target=lambda: [bar.refresh() for _ in
                                     iter(lambda: stop.wait(15), True)],
                     daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(count_split_one, sid, anno_path, cohort_dir,
                                   analysis_name, keep_scaffolds, timeout)
                       for sid in todo]
            for future in as_completed(futures):
                sid, ok, detail = future.result()
                if not ok:
                    failures.append((sid, detail))
                    bar.write(f"  FAILED {sid}: {detail}")
                bar.update(1)
    finally:
        stop.set()
        bar.close()

    # A sample killed for memory is the most retryable failure there is: the
    # BAM is fine, it just ran alongside two others. Sequential retries give it
    # the whole budget, which is often all it needed. Genuinely broken inputs
    # fail again, and now their R-level error survives to reach the log.
    if failures and max_workers > 1:
        retry_ids = [sid for sid, _ in failures]
        print(f"\nRetrying {len(retry_ids)} failed sample(s) one at a time, so "
              f"each gets the full memory budget: {retry_ids}")
        still_failing = []
        bar = tqdm(total=len(retry_ids), unit="sample", desc="retry (serial)",
                   ascii=True,
                   bar_format="{desc:<38} {bar} {n_fmt}/{total_fmt} [{elapsed}]")
        for sid in retry_ids:
            _, ok, detail = count_split_one(sid, anno_path, cohort_dir,
                                            analysis_name, keep_scaffolds, timeout)
            bar.write(f"  recovered {sid}" if ok else f"  FAILED again {sid}: {detail}")
            if not ok:
                still_failing.append((sid, detail))
            bar.update(1)
        bar.close()
        failures = still_failing
    return failures


def write_gene_ranges(gencode_path, out_path):
    """
    Export protein-coding gene ranges from the GENCODE feather as a plain TSV.

    Fallback for a reference bundle that does not ship gene_ranges.tsv, or
    ships it read-only. protein_coding matches FRASER's own annotateRanges
    default: without the filter a junction inside a coding gene routinely also
    overlaps a lncRNA or pseudogene and the symbol comes back ';'-joined, which
    then fails to match the plain symbols used downstream.
    """
    genes = pd.read_feather(gencode_path)
    required = {"seqname", "start", "end", "strand", "gene_name", "gene_type"}
    missing = required - set(genes.columns)
    if missing:
        raise ValueError(f"GENCODE annotation is missing expected column(s): "
                         f"{sorted(missing)}\n  Looked in: {gencode_path}")

    genes = genes[genes["gene_type"] == "protein_coding"]
    genes = (genes[["seqname", "start", "end", "strand", "gene_name"]]
             .rename(columns={"seqname": "chr"}).dropna()
             .sort_values(["chr", "start", "end"]).reset_index(drop=True))
    if genes.empty:
        raise ValueError(f"No protein-coding genes found in {gencode_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    genes.to_csv(out_path, sep="\t", index=False)
    print(f"  generated {len(genes)} protein-coding gene range(s)")
    return out_path


def prepare_fraser_cohort(rna_cohort):
    """
    Turn the ProcessBAM table into something FRASER will accept.

    Two things are settled here rather than in R, because FRASER signals both
    by aborting rather than adapting: a sample whose layout is unknown cannot
    be counted, and FraserDataSet() raises outright on a cohort mixing
    unstranded with stranded samples, since unstranded yields '*' junctions
    that cannot be reconciled with '+'/'-' ones. A control failing either test
    is dropped - it is padding and there are others - while a case cannot be.
    """
    cohort = rna_cohort.copy()

    unknown = (cohort["strand"] == "unknown") | (cohort["pairedEnd"] == "unknown")
    if unknown.any():
        bad_cases = cohort.loc[unknown & (cohort["sample_type"] == "case"), "sampleID"]
        if len(bad_cases) > 0:
            raise RuntimeError(f"Library layout could not be determined for case "
                               f"sample(s): {bad_cases.tolist()}\n"
                               f"  Set 'strand' and/or 'pairedEnd' explicitly in "
                               f"the samplesheet. Most human RNA-seq is "
                               f"strand='reverse'.")
        dropped = cohort.loc[unknown, "sampleID"].tolist()
        print(f"  dropping {len(dropped)} control(s) with unknown layout: {dropped}")
        cohort = cohort.loc[~unknown].copy()

    case_strands = set(cohort.loc[cohort["sample_type"] == "case", "strand"])
    if "unstranded" in case_strands and case_strands & {"forward", "reverse"}:
        raise RuntimeError(f"The case samples themselves mix stranded and "
                           f"unstranded libraries ({sorted(case_strands)}).\n"
                           f"  FRASER cannot model them together; run them as "
                           f"separate cohorts.")

    cases_are_stranded = bool(case_strands & {"forward", "reverse"})
    incompatible = ((cohort["strand"] == "unstranded") if cases_are_stranded
                    else cohort["strand"].isin(["forward", "reverse"])) \
        & (cohort["sample_type"] == "control")
    if incompatible.any():
        dropped = cohort.loc[incompatible, "sampleID"].tolist()
        print(f"  dropping {len(dropped)} control(s) incompatible with the cases' "
              f"library type: {dropped}")
        cohort = cohort.loc[~incompatible].copy()

    if int((cohort["sample_type"] == "case").sum()) == 0:
        raise RuntimeError("No usable case sample remains; nothing to analyse.")

    if len(cohort) < 50:
        print(f"  WARNING: only {len(cohort)} sample(s) in the cohort. DROP "
              f"recommends at least 50 of the same tissue and protocol; with "
              f"fewer, the latent space stays small and p-values run conservative.")

    if len(set(cohort["strand"]) & {"forward", "reverse"}) > 1:
        print(f"  NOTE: the cohort mixes forward and reverse libraries "
              f"{cohort['strand'].value_counts().to_dict()}. FRASER counts each "
              f"BAM with its own strandMode so this is handled correctly, but it "
              f"remains a technical covariate for the autoencoder to absorb.")

    return build_fraser_anno(cohort, with_seqlevelstyle=True), cohort.reset_index(drop=True)


# ---------------------------------------------------------------------------
# OUTRIDER
# ---------------------------------------------------------------------------

def write_gene_saf(gene_ranges_tsv, out_path):
    """
    Convert the gene-range TSV into the SAF layout featureCounts expects.

    SAF sums every row sharing a GeneID, so this works unchanged whether the
    source holds one interval per gene or one per exon. GeneID is the symbol,
    which makes it the rowname of the count matrix and therefore the
    'GeneSymbol' column downstream expects, with no later join.
    """
    genes = pd.read_csv(gene_ranges_tsv, sep="\t")
    saf = pd.DataFrame({"GeneID": genes["gene_name"], "Chr": genes["chr"],
                        "Start": genes["start"], "End": genes["end"],
                        "Strand": genes["strand"]}).dropna()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saf.to_csv(out_path, sep="\t", index=False)
    return out_path, saf["GeneID"].nunique()


def expr_count_cache_file(cache_dir, sample_id):
    return Path(cache_dir) / f"geneCounts-{sample_id}.tsv.gz"


def count_expression_one(sample_id, anno_path, saf_path, cache_dir, sample_dir,
                         threads, timeout):
    """
    Count one sample in its own R process. Returns (id, ok, detail).

    Scratch goes under the sample's folder and the subprocess runs from there.
    tmpDir covers featureCounts itself, but Rsubread also writes a
    .Rsubread_UserProvidedAnnotation_pid* file relative to the working
    directory, which tmpDir does not govern - and a subprocess launched from a
    notebook inherits the user's source tree as its working directory.
    """
    scratch = Path(sample_dir) / "_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    cmd = ["Rscript", "--vanilla", str(rscript_path("count_expression_one.R")),
           str(anno_path), str(saf_path), str(cache_dir), str(sample_id),
           str(threads), str(scratch)]
    log_path = Path(cache_dir).parent.parent / "logs" / f"expr-{sample_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, cwd=scratch)
    except subprocess.TimeoutExpired:
        log_path.write_text(f"timed out after {timeout}s\n")
        return sample_id, False, f"timed out after {timeout / 3600:.1f}h"

    log_path.write_text(" ".join(cmd) + "\n\n" + (result.stdout or "") + (result.stderr or ""))
    if result.returncode == 0 and expr_count_cache_file(cache_dir, sample_id).exists():
        # Cleared only on success: featureCounts leaves partial output behind
        # when it fails, and that is worth having when working out why.
        shutil.rmtree(scratch, ignore_errors=True)
        return sample_id, True, ""
    if result.returncode != 0:
        return sample_id, False, subprocess_detail(result)
    return sample_id, False, "exited cleanly but wrote no counts"


def count_expression_parallel(anno, anno_path, saf_path, cache_dir, root,
                              max_workers, threads_each, timeout=4 * 3600):
    """
    Count genes for every sample, one process each.

    Gene counts depend only on the BAM and the annotation, never on who else is
    in the cohort, so this cache is never invalidated by adding or removing a
    patient - unlike FRASER's non-split counts. Changing the annotation does
    invalidate it, which run_outrider.R catches by comparing gene sets.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = list(anno["sampleID"].astype(str))
    todo = [s for s in sample_ids if not expr_count_cache_file(cache_dir, s).exists()]
    if not todo:
        print(f"Gene counts: all {len(sample_ids)} sample(s) already counted.")
        return []

    print(f"Counting genes for {len(todo)} sample(s) "
          f"({len(sample_ids) - len(todo)} already cached), "
          f"{max_workers} process(es) x {threads_each} thread(s).")

    failures = []
    bar = tqdm(total=len(todo), unit="sample", desc="gene counts", ascii=True,
               bar_format="{desc:<38} {bar} {n_fmt}/{total_fmt} [{elapsed}]")
    stop = threading.Event()
    threading.Thread(target=lambda: [bar.refresh() for _ in
                                     iter(lambda: stop.wait(15), True)],
                     daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(count_expression_one, sid, anno_path, saf_path,
                                   cache_dir, Path(root) / sid, threads_each, timeout)
                       for sid in todo]
            for future in as_completed(futures):
                sid, ok, detail = future.result()
                if not ok:
                    failures.append((sid, detail))
                    bar.write(f"  FAILED {sid}: {detail}")
                bar.update(1)
    finally:
        stop.set()
        bar.close()
    return failures


# ---------------------------------------------------------------------------
# ASE
# ---------------------------------------------------------------------------

def bh_fdr(pvals):
    """
    Benjamini-Hochberg, vectorised.

    The cumulative minimum runs over the p-values in sorted order and is only
    then scattered back. Applying it in the original order silently skips the
    monotonicity step, which does not merely perturb the q-values: on a
    60,000-site table it turned 320 significant hits into 59,996, because a
    running minimum over an arbitrarily ordered suffix almost always picks up
    one of the smallest adjusted values.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    adj = pvals[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(adj, 0.0, 1.0)
    return q


def vcf_style(vcf_path):
    """UCSC or Ensembl contig naming, from the header, else the first data row."""
    result = run_cmd(["bcftools", "view", "-h", str(vcf_path)])
    if result.returncode == 0:
        ensembl = set(main_chroms("Ensembl")) | {"MT", "Y"}
        for line in result.stdout.splitlines():
            if line.startswith("##contig=<ID="):
                name = line.split("ID=", 1)[1].split(",", 1)[0].rstrip(">")
                if name.startswith("chr"):
                    return "UCSC"
                if name in ensembl:
                    return "Ensembl"

    result = run_cmd(["bcftools", "view", "-H", str(vcf_path)])
    if result.returncode == 0 and result.stdout:
        first = result.stdout.split("\n", 1)[0].split("\t", 1)[0]
        return "UCSC" if first.startswith("chr") else "Ensembl"
    return "unknown"


def vcf_samples(vcf_path):
    result = run_cmd(["bcftools", "query", "-l", str(vcf_path)])
    return [s for s in result.stdout.split() if s] if result.returncode == 0 else []


def is_gvcf(vcf_path):
    """
    Whether this needs genotyping before its calls can be used.

    Detected from the header, not the filename: the .g.vcf convention is only a
    convention, and a gVCF named .vcf.gz would otherwise sail through and
    produce a het set full of <NON_REF> reference blocks.
    """
    result = run_cmd(["bcftools", "view", "-h", str(vcf_path)])
    if result.returncode != 0:
        return False
    return "##GVCFBlock" in result.stdout or "ID=NON_REF" in result.stdout


def ensure_vcf_index(vcf_path):
    """Index a VCF if it has none. Returns (ok, detail)."""
    vcf_path = Path(vcf_path)
    for suffix in (".tbi", ".csi", ".idx"):
        if Path(str(vcf_path) + suffix).exists():
            return True, ""
    result = (run_cmd(["bcftools", "index", "-t", "-f", str(vcf_path)])
              if str(vcf_path).endswith(".gz")
              else run_cmd(["gatk", "IndexFeatureFile", "-I", str(vcf_path)]))
    if result.returncode != 0:
        return False, f"could not index {vcf_path.name}: {result.stderr.strip()[-200:]}"
    return True, ""


def rename_vcf_contigs(vcf_path, target_style, out_dir):
    """
    Write a copy of the VCF with contigs renamed to `target_style`.

    Only the VCF is ever rewritten. The BAM's naming is authoritative because
    rewriting a BAM header streams out a full copy - hundreds of gigabytes for
    a cohort - while a VCF is three orders of magnitude smaller. The map is
    derived here rather than read from a file, so it always matches the
    conventions this module uses.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ucsc = main_chroms("UCSC") + ["chrY", "chrM"]
    ens = main_chroms("Ensembl") + ["Y", "MT"]
    pairs = zip(ens, ucsc) if target_style == "UCSC" else zip(ucsc, ens)

    map_file = out_dir / "chrom_rename.txt"
    map_file.write_text("".join(f"{a}\t{b}\n" for a, b in pairs))
    out_vcf = out_dir / "renamed.vcf.gz"
    result = run_cmd(["bcftools", "annotate", "--rename-chrs", str(map_file),
                      "-O", "z", "-o", str(out_vcf), str(vcf_path)])
    if result.returncode != 0:
        raise RuntimeError(f"bcftools annotate failed while renaming contigs: "
                           f"{result.stderr.strip()[-300:]}")
    ok, detail = ensure_vcf_index(out_vcf)
    if not ok:
        raise RuntimeError(detail)
    return out_vcf, f"contigs renamed to {target_style} naming"


def compute_mae(counts_path, sample_id, min_depth, ratio_thr, fdr_alpha):
    """
    Turn ASEReadCounter output into per-site allelic ratios and an MAE flag.

    Every reported site is kept; IS_MAE is a column, not a filter. Downstream
    weighs the raw allelic ratio, and a site failing the MAE screen is still
    evidence.

    The null is a fair coin, which ignores reference bias: reads carrying the
    alternate allele align slightly less readily, so the true null sits a
    little below 0.5 and the residual skew is towards REF. The flag is a screen
    rather than a verdict, so the approximation is tolerable; correcting it
    would be the thing to change here.

    Detection is depth-sensitive but not by a fixed cutoff. At depth 10 the most
    extreme admissible ratio is 0/10, whose two-sided p is 0.002; whether that
    clears BH depends on how many other extreme sites the sample has, since the
    threshold is adaptive. A 1/10 site, at p = 0.02, realistically never does.
    """
    df = pd.read_csv(counts_path, sep="\t")
    rename = {"contig": "CHROM", "position": "POS", "refAllele": "REF",
              "altAllele": "ALT", "refCount": "REF_COUNT",
              "altCount": "ALT_COUNT", "totalCount": "TOTAL_COUNT"}
    missing = set(rename) - set(df.columns)
    if missing:
        raise RuntimeError(f"ASEReadCounter output is missing expected column(s): "
                           f"{sorted(missing)}")
    df = df.rename(columns=rename)
    df = df[df["TOTAL_COUNT"] >= min_depth].copy()

    columns = ["sampleID", "CHROM", "POS", "REF", "ALT", "REF_COUNT", "ALT_COUNT",
               "TOTAL_COUNT", "ALT_RATIO", "PVAL", "QVAL", "IS_MAE"]
    if df.empty:
        return pd.DataFrame(columns=columns), 0

    alt = df["ALT_COUNT"].to_numpy(dtype=np.int64)
    total = df["TOTAL_COUNT"].to_numpy(dtype=np.int64)

    # An exact binomial test cannot be vectorised, but (alt, total) repeats
    # heavily - tens of thousands of rows collapse to a few thousand distinct
    # pairs - so each distinct pair is tested once.
    pairs, inverse = np.unique(np.column_stack([alt, total]), axis=0,
                               return_inverse=True)
    unique_p = np.array([binomtest(int(k), int(n), 0.5, alternative="two-sided").pvalue
                         for k, n in pairs])
    pvals = unique_p[inverse]
    qvals = bh_fdr(pvals)
    ratio = alt / total

    out = pd.DataFrame({
        "sampleID": sample_id, "CHROM": df["CHROM"].to_numpy(),
        "POS": df["POS"].to_numpy(), "REF": df["REF"].to_numpy(),
        "ALT": df["ALT"].to_numpy(), "REF_COUNT": df["REF_COUNT"].to_numpy(),
        "ALT_COUNT": alt, "TOTAL_COUNT": total, "ALT_RATIO": ratio,
        "PVAL": pvals, "QVAL": qvals,
        "IS_MAE": (((ratio <= ratio_thr) | (ratio >= 1 - ratio_thr))
                   & (qvals < fdr_alpha)).astype(int),
    })
    return out, int(out["IS_MAE"].sum())


def run_ase_one(row, out_root, fasta_references, overwrite, min_depth, min_mapq,
                min_baseq, ratio_thr, fdr_alpha, timeout):
    """
    Full ASE pipeline for one sample. Returns
    (sample_id, ok, detail, notes, n_sites, n_mae).
    """
    sample_id = str(row["sampleID"])
    bam, vcf = str(row["rna_path"]), str(row["vcf_path"])
    style = str(row.get("SeqLevelStyle", "UCSC"))
    notes, log = [], []

    sample_dir = Path(out_root) / sample_id
    tmp_dir = sample_dir / "_tmp"
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(out_root) / "_cohort" / "logs" / f"ase-{sample_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path, out_path = sample_dir / "ase_counts.tsv", sample_dir / "ase.feather"

    if out_path.exists() and not overwrite:
        existing = pd.read_feather(out_path)
        return (sample_id, True, "", ["reused existing result"],
                len(existing), int(existing["IS_MAE"].sum()))

    def record(step, result):
        log.append(f"$ {' '.join(step)}\n{result.stdout}{result.stderr}\n")

    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # The reference follows the BAM, never the other way round.
        if style == "Ensembl":
            ref, target_style = fasta_references.fasta, "Ensembl"
        else:
            ref, target_style = fasta_references.fasta_ucsc, "UCSC"
            if style == "unknown":
                notes.append("BAM contig naming undetermined; assuming UCSC")

        vcf_in = Path(vcf)
        found_style = vcf_style(vcf_in)
        if found_style == "unknown":
            return sample_id, False, "could not determine the VCF's contig naming", notes, 0, 0
        if found_style != target_style:
            notes.append(f"VCF uses {found_style} contig naming but the BAM uses "
                         f"{target_style}; a renamed copy was written to {tmp_dir}")
            vcf_in, _ = rename_vcf_contigs(vcf_in, target_style, tmp_dir)
        else:
            ok, detail = ensure_vcf_index(vcf_in)
            if not ok:
                return sample_id, False, detail, notes, 0, 0

        if is_gvcf(vcf_in):
            notes.append("input is a gVCF; genotyped before use")
            genotyped = tmp_dir / "genotyped.vcf.gz"
            step = ["gatk", "GenotypeGVCFs", "-R", str(ref), "-V", str(vcf_in),
                    "-O", str(genotyped)]
            result = run_cmd(step, timeout)
            record(step, result)
            if result.returncode != 0:
                return (sample_id, False,
                        f"GenotypeGVCFs failed: {result.stderr.strip()[-200:]}",
                        notes, 0, 0)
            vcf_in = genotyped

        samples = vcf_samples(vcf_in)
        if not samples:
            return sample_id, False, "no samples found in the VCF", notes, 0, 0
        if len(samples) == 1:
            vcf_sample = samples[0]
        else:
            # A joint-called family VCF is common in rare disease, and taking
            # the first column would silently use a parent's genotypes.
            matches = ([s for s in samples if s == sample_id]
                       or [s for s in samples if s in sample_id or sample_id in s])
            if len(matches) != 1:
                return (sample_id, False,
                        f"the VCF holds {len(samples)} samples {samples} and none "
                        f"uniquely matches {sample_id}; a single-sample VCF or a "
                        f"matching sample name is required", notes, 0, 0)
            vcf_sample = matches[0]
            notes.append(f"multi-sample VCF; using genotypes for {vcf_sample}")

        intervals = [arg for chrom in main_chroms(target_style) for arg in ("-L", chrom)]
        het_vcf = tmp_dir / "hets.vcf.gz"
        step = ["gatk", "SelectVariants", "-R", str(ref), "-V", str(vcf_in),
                "-O", str(het_vcf), "--select-type-to-include", "SNP",
                "--restrict-alleles-to", "BIALLELIC", "--exclude-filtered",
                "-sn", vcf_sample,
                "-select", f"vc.getGenotype('{vcf_sample}').isHet()"] + intervals
        result = run_cmd(step, timeout)
        record(step, result)
        if result.returncode != 0:
            return (sample_id, False,
                    f"SelectVariants failed: {result.stderr.strip()[-200:]}",
                    notes, 0, 0)

        step = ["gatk", "ASEReadCounter", "-R", str(ref), "-I", bam,
                "-V", str(het_vcf), "-O", str(counts_path),
                "--min-mapping-quality", str(min_mapq),
                "--min-base-quality", str(min_baseq),
                "--min-depth-of-non-filtered-base", str(min_depth),
                "--output-format", "TABLE"]
        result = run_cmd(step, timeout)
        record(step, result)
        if result.returncode != 0:
            return (sample_id, False,
                    f"ASEReadCounter failed: {result.stderr.strip()[-200:]}",
                    notes, 0, 0)

        table, n_mae = compute_mae(counts_path, sample_id, min_depth,
                                   ratio_thr, fdr_alpha)
        table.to_feather(out_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return sample_id, True, "", notes, len(table), n_mae

    except subprocess.TimeoutExpired:
        return sample_id, False, f"timed out after {timeout / 3600:.1f}h", notes, 0, 0
    except Exception as exc:
        return sample_id, False, str(exc), notes, 0, 0
    finally:
        log_path.write_text("\n".join(log))


# ---------------------------------------------------------------------------
# PrepareRNAEvidence
# ---------------------------------------------------------------------------

def prepare_splicing(table):
    cols = ["sampleID", "seqnames", "start", "end", "strand", "hgnc_symbol",
            "pvaluesBetaBinomial_jaccard", "psi5", "psi3",
            "rawOtherCounts_psi5", "rawOtherCounts_psi3", "rawCountsJnonsplit",
            "jaccard", "rawOtherCounts_jaccard", "delta_jaccard",
            "delta_psi5", "delta_psi3", "predictedMeans_jaccard"]
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise ValueError(f"splicing table is missing column(s) {missing}")
    return table[cols]


def prepare_expression(table):
    """
    Trim to what filter_one renames.

    Necessary rather than tidy: filter_one renames these seven and then merges
    the whole frame, so anything extra from results(all=TRUE) would ride along
    unrenamed. RawZscore is optional - filter_one fills NaN when absent.
    """
    cols = ["sampleID", "GeneSymbol", "pValue", "padjust", "zScore", "l2fc", "rawcounts"]
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise ValueError(f"expression table is missing column(s) {missing}")
    if "RawZscore" in table.columns:
        cols = cols + ["RawZscore"]
    return table[cols]


def prepare_ase(table):
    """
    Add the variant identifier filter_one joins on.

    The DNA side writes varId as CHROM_POS_REF_ALT with chromosomes as bare
    numbers and X as 23, Y as 24 - filter_one reverses that mapping when it
    rebuilds coordinates. Sites on contigs outside that scheme are dropped
    because they could never match anything on the DNA side.
    """
    missing = [c for c in ("CHROM", "POS", "REF", "ALT") if c not in table.columns]
    if missing:
        raise ValueError(f"ASE table is missing column(s) {missing}")

    table = table.copy()
    chrom = table["CHROM"].astype(str)
    keep = (chrom.str.match(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y)$", case=False, na=False)
            | chrom.str.match(r"^([1-9]|1[0-9]|2[0-4])$", na=False))
    table = table.loc[keep].copy()

    table["CHROM"] = (table["CHROM"].astype(str)
                      .str.replace(r"^chr", "", regex=True, case=False)
                      .replace({"X": "23", "x": "23", "Y": "24", "y": "24"}))
    table["varId"] = (table["CHROM"].astype(str) + "_" + table["POS"].astype(str)
                      + "_" + table["REF"].astype(str) + "_" + table["ALT"].astype(str))

    cols = ["sampleID", "varId", "CHROM", "POS", "REF", "ALT", "REF_COUNT",
            "ALT_COUNT", "TOTAL_COUNT", "ALT_RATIO", "PVAL", "QVAL", "IS_MAE"]
    return table[[c for c in cols if c in table.columns]]


def prepare_one(sample_id, sources, save_root, overwrite):
    """
    Reshape whatever this sample has.

    Returns (sample_id, written, absent, error). An analysis never run, or whose
    output has since been removed, goes into `absent` rather than raising: a
    patient with ASE but no splicing should still get their ASE through, and
    filter_one already fills the missing side with NaN.
    """
    reshape = {"splicing": prepare_splicing, "expression": prepare_expression,
               "ase": prepare_ase}
    kinds = rna_kinds()
    written, absent = {}, []
    try:
        for kind, src in sources.items():
            if not src or not Path(src).exists():
                absent.append(kind)
                continue
            out = save_root / kinds[kind][1] / f"{sample_id}.feather"
            if out.exists() and not overwrite:
                written[kind] = out
                continue
            table = reshape[kind](pd.read_feather(src))
            out.parent.mkdir(parents=True, exist_ok=True)
            table.reset_index(drop=True).to_feather(out)
            written[kind] = out
        return sample_id, written, absent, None
    except Exception as exc:
        return sample_id, written, absent, str(exc)