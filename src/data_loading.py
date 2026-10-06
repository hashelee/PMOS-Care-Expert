"""Load the Kottarathil PCOS dataset and apply minimal, logged type fixes.

What this module does:
  1. finds the raw file in data/raw/
  2. renames columns to canonical snake_case names (see feature_groups.py)
  3. drops columns that are completely empty
  4. converts numeric columns stored as text into numbers, and logs every cell it
     had to change

What it deliberately does NOT do: remove rows, impute missing values, or fix
outliers. Those are modelling decisions, made later inside the cross-validation
pipeline so that nothing leaks from the test folds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .feature_groups import FEATURE_TABLE, match_key

RAW_FILE_PATTERNS = [
    "*without*infertility*.xlsx",
    "*without*infertility*.xls",
    "*without*infertility*.csv",
]


def find_raw_file(raw_dir: Path = config.DATA_RAW) -> Path:
    """Return the main Kottarathil file in data/raw/, or explain how to get it."""
    for pattern in RAW_FILE_PATTERNS:
        matches = sorted(Path(raw_dir).glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find the dataset in {raw_dir}.\n"
        f"Download it from {config.KOTTARATHIL_URL}, unzip it, and put "
        "'PCOS_data_without_infertility.xlsx' in data/raw/."
    )


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw file exactly as it is, with no changes."""
    path = Path(path) if path else find_raw_file()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    sheets = pd.ExcelFile(path).sheet_names
    sheet = config.KOTTARATHIL_SHEET if config.KOTTARATHIL_SHEET in sheets else sheets[0]
    return pd.read_excel(path, sheet_name=sheet)


def rename_columns(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rename source columns to canonical names.

    Returns the renamed frame and a mapping table (source -> canonical), where
    unknown columns get canonical = None so they can be reviewed.
    """
    lookup = dict(zip(FEATURE_TABLE["match_key"], FEATURE_TABLE["canonical"]))
    rows, rename = [], {}
    for col in raw.columns:
        canonical = lookup.get(match_key(col))
        rows.append({"source_column": repr(col), "canonical": canonical})
        if canonical:
            rename[col] = canonical
    mapping = pd.DataFrame(rows)
    return raw.rename(columns=rename), mapping


def _parse_number(value):
    """Parse one cell to float. Returns (number_or_nan, action)."""
    if pd.isna(value):
        return np.nan, None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), None
    text = str(value).strip()
    try:
        return float(text), "text converted to number"
    except ValueError:
        pass
    # Common typo in this dataset: a trailing full stop, e.g. "1.99."
    cleaned = text.rstrip(".").strip()
    try:
        return float(cleaned), f"fixed typo '{text}' -> {cleaned}"
    except ValueError:
        return np.nan, f"unparseable '{text}' -> set to missing"


def coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make every known feature numeric. Logs each cell that needed a fix.

    Text-to-number conversions where the value is unchanged (e.g. '2.5' -> 2.5)
    are counted but not logged row by row.
    """
    df = df.copy()
    log = []
    known = FEATURE_TABLE.set_index("canonical")
    for col in df.columns:
        if col not in known.index or known.loc[col, "type"] == "id":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        parsed = df[col].map(_parse_number)
        values = parsed.map(lambda t: t[0]).astype(float)
        actions = parsed.map(lambda t: t[1])
        silent = int((actions == "text converted to number").sum())
        for idx in actions.index[actions.notna() & (actions != "text converted to number")]:
            log.append({
                "row_index": idx,
                config.PATIENT_ID: df.at[idx, config.PATIENT_ID] if config.PATIENT_ID in df else None,
                "column": col,
                "original_value": repr(df.at[idx, col]),
                "action": actions[idx],
            })
        if silent:
            log.append({
                "row_index": None, config.PATIENT_ID: None, "column": col,
                "original_value": f"{silent} cells",
                "action": "column stored as text; values converted to numbers unchanged",
            })
        df[col] = values
    cols = ["row_index", config.PATIENT_ID, "column", "original_value", "action"]
    log = pd.DataFrame(log, columns=cols)
    for c in ("row_index", config.PATIENT_ID):
        log[c] = pd.to_numeric(log[c], errors="coerce").astype("Int64")
    return df, log


def load_dataset(path: Path | None = None) -> dict:
    """Load, rename and type-fix the dataset.

    Returns a dict with:
      raw            - the file exactly as read
      data           - renamed, empty columns dropped, numeric types fixed
      column_mapping - source column -> canonical name
      dropped_empty  - columns removed because every value was missing
      cleaning_log   - every cell that was changed during type conversion
      source_path    - the file that was read
    """
    path = Path(path) if path else find_raw_file()
    raw = load_raw(path)
    renamed, mapping = rename_columns(raw)

    empty_cols = [c for c in renamed.columns if renamed[c].isna().all()]
    renamed = renamed.drop(columns=empty_cols)

    data, log = coerce_types(renamed)
    if config.TARGET in data:
        data[config.TARGET] = data[config.TARGET].astype(int)

    return {
        "raw": raw,
        "data": data,
        "column_mapping": mapping,
        "dropped_empty": [repr(c) for c in empty_cols],
        "cleaning_log": log,
        "source_path": path,
    }
