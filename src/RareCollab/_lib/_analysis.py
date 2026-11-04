"""
Where one analysis's outputs live.

A patient measured against two background cohorts has two sets of FRASER and
OUTRIDER results, and therefore two candidate lists, two agent readings and two
reports. Every one of those is written one file per patient, so without a
folder per analysis the second run either overwrites the first or skips it as
already done. This module is the single place that decides what that folder is
called, so the steps that need it cannot drift apart.

ASE is deliberately not covered here. It is a pileup over the patient's own
reads and has one answer whichever cohort sits alongside, so it is written once
and shared.
"""

from pathlib import Path

import pandas as pd

DEFAULT_ANALYSIS = "my_work"
ANALYSIS_COLUMN = "analysis_name"


def analysis_subdir(root, analysis_name):
    """
    Where one analysis's per-sample outputs live under `root`.

    The default analysis writes straight into `root`, so a user who never
    passes analysis_name sees exactly the layout earlier versions produced and
    never has to learn the concept exists. A named analysis gets a folder of
    its own, which is what stops one patient's two background cohorts from
    overwriting each other.
    """
    name = str(analysis_name or DEFAULT_ANALYSIS).strip() or DEFAULT_ANALYSIS
    if name == DEFAULT_ANALYSIS:
        return Path(root)
    # '.' and '..' are checked by name: Path('..').name is '..', so they pass
    # the plain-name test below and would otherwise resolve outside the work
    # directory. A leading underscore is reserved for _cohort and friends.
    if name in (".", "..") or name.startswith("_") or name != Path(name).name:
        raise ValueError(
            f"analysis_name must be a plain folder name, not {analysis_name!r}")
    return Path(root) / name


def row_analysis(row):
    """
    The analysis a samplesheet row belongs to, defaulting when the column is
    absent or blank.

    Read off the row rather than taken as an argument: once the analyses are
    merged into one samplesheet a patient can occupy several rows, and each has
    to resolve its own paths within a single pass over the sheet.
    """
    name = getattr(row, ANALYSIS_COLUMN, None)
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return DEFAULT_ANALYSIS
    return str(name).strip() or DEFAULT_ANALYSIS