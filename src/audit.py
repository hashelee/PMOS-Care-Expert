"""Dataset audit for the Kottarathil PCOS dataset.

Covers tracker tasks:
  IMP-02  dataset profile (records, features, class balance)
  IMP-03  missing values per feature
  IMP-04  duplicates and data-quality problems (implausible values, extreme outliers)
  IMP-05  feature-group dictionary (written to docs/)

Run from the repository root:
    python -m src.audit
Outputs go to results/audit/, docs/ and data/processed/.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from matplotlib.figure import Figure  # draws to file without needing a display
from matplotlib.ticker import MaxNLocator

from . import config
from .data_loading import load_dataset
from .feature_groups import FEATURE_TABLE, MODEL_GROUPS

# Chart styling: one hue, recessive axes, direct labels.
BAR_COLOR = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"


# ---------------------------------------------------------------- IMP-02
def profile(result: dict) -> dict:
    raw, data = result["raw"], result["data"]
    features = [c for c in data.columns if c in set(FEATURE_TABLE.loc[FEATURE_TABLE["group"].isin(MODEL_GROUPS), "canonical"])]
    counts = data[config.TARGET].value_counts().sort_index()
    return {
        "source_file": result["source_path"].name,
        "n_records": len(data),
        "n_columns_raw": raw.shape[1],
        "n_columns_after_dropping_empty": data.shape[1],
        "n_model_features": len(features),
        "n_pcos": int(counts.get(1, 0)),
        "n_not_pcos": int(counts.get(0, 0)),
        "pct_pcos": round(100 * counts.get(1, 0) / len(data), 1),
        "unmapped_columns": [c for c in result["column_mapping"].loc[result["column_mapping"]["canonical"].isna(), "source_column"]
                             if c not in result["dropped_empty"]],
        "dropped_empty_columns": result["dropped_empty"],
    }


def features_per_group(data: pd.DataFrame) -> pd.DataFrame:
    present = FEATURE_TABLE[FEATURE_TABLE["canonical"].isin(data.columns)]
    out = present.groupby("group")["canonical"].agg(n_features="count", features=lambda s: ", ".join(s))
    order = ["identifier", "target"] + MODEL_GROUPS
    return out.reindex([g for g in order if g in out.index]).reset_index()


# ---------------------------------------------------------------- IMP-03
def missing_values(data: pd.DataFrame) -> pd.DataFrame:
    groups = FEATURE_TABLE.set_index("canonical")["group"]
    n_missing = data.isna().sum()
    by_class = data.groupby(config.TARGET).apply(lambda g: g.isna().sum(), include_groups=False).T
    out = pd.DataFrame({
        "column": n_missing.index,
        "group": [groups.get(c, "UNMAPPED") for c in n_missing.index],
        "n_missing": n_missing.values,
        "pct_missing": (100 * n_missing / len(data)).round(2).values,
        "missing_in_not_pcos": by_class.get(0, 0).reindex(n_missing.index).values,
        "missing_in_pcos": by_class.get(1, 0).reindex(n_missing.index).values,
    })
    return out.sort_values(["n_missing", "column"], ascending=[False, True]).reset_index(drop=True)


def rows_with_missing(data: pd.DataFrame) -> pd.DataFrame:
    mask = data.isna().any(axis=1)
    rows = data.loc[mask]
    return pd.DataFrame({
        "row_index": rows.index,
        config.PATIENT_ID: rows.get(config.PATIENT_ID),
        "missing_columns": rows.apply(lambda r: ", ".join(r.index[r.isna()]), axis=1),
    }).reset_index(drop=True)


# ---------------------------------------------------------------- IMP-04
def duplicates(data: pd.DataFrame) -> pd.DataFrame:
    ids = [c for c in ("sl_no", config.PATIENT_ID) if c in data]
    feature_cols = [c for c in data.columns if c not in ids]
    checks = []
    for col in ids:
        dup = data[data[col].duplicated(keep=False)]
        checks.append({"check": f"duplicate {col}", "n_rows_involved": len(dup),
                       "row_indices": ", ".join(map(str, dup.index.tolist()))})
    dup = data[data[feature_cols].duplicated(keep=False)]
    checks.append({"check": "identical rows (all features + label, ignoring IDs)",
                   "n_rows_involved": len(dup), "row_indices": ", ".join(map(str, dup.index.tolist()))})
    without_label = [c for c in feature_cols if c != config.TARGET]
    dup = data[data[without_label].duplicated(keep=False)]
    conflict = dup.groupby(without_label, dropna=False)[config.TARGET].nunique()
    checks.append({"check": "identical features but different label",
                   "n_rows_involved": int((conflict > 1).sum()), "row_indices": ""})
    return pd.DataFrame(checks)


def implausible_values(data: pd.DataFrame) -> pd.DataFrame:
    """Values outside the wide screening bounds in feature_groups.py."""
    rows = []
    for _, f in FEATURE_TABLE.iterrows():
        col = f["canonical"]
        if col not in data or f["type"] != "numeric":
            continue
        lo, hi = f["plausible_low"], f["plausible_high"]
        s = data[col]
        bad = pd.Series(False, index=s.index)
        if pd.notna(lo):
            bad |= s < lo
        if pd.notna(hi):
            bad |= s > hi
        for idx in s.index[bad]:
            rows.append({"row_index": idx, config.PATIENT_ID: data.at[idx, config.PATIENT_ID] if config.PATIENT_ID in data else None,
                         "column": col, "value": s[idx], "allowed_range": f"{lo} to {hi}",
                         "pcos": data.at[idx, config.TARGET]})
    return pd.DataFrame(rows, columns=["row_index", config.PATIENT_ID, "column", "value", "allowed_range", "pcos"])


def derived_consistency(data: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    """Check derived columns against the columns they are computed from."""
    checks = []

    def compare(name, observed, expected):
        rel = ((observed - expected).abs() / expected.abs()).where(expected != 0)
        bad = rel > tol
        checks.append({"check": name, "n_checked": int(rel.notna().sum()),
                       "n_mismatch_over_5pct": int(bad.sum()),
                       "row_indices": ", ".join(map(str, rel.index[bad].tolist()[:50]))})

    if {"bmi", "weight", "height"} <= set(data):
        compare("BMI = weight / (height in m)^2", data["bmi"], data["weight"] / (data["height"] / 100) ** 2)
    if {"waist_hip_ratio", "waist", "hip"} <= set(data):
        compare("waist_hip_ratio = waist / hip", data["waist_hip_ratio"], data["waist"] / data["hip"])
    if {"fsh_lh_ratio", "fsh", "lh"} <= set(data):
        compare("fsh_lh_ratio = FSH / LH", data["fsh_lh_ratio"], data["fsh"] / data["lh"])
    return pd.DataFrame(checks)


def extreme_outliers(data: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Count values beyond Q1 - k*IQR or Q3 + k*IQR (k=3 flags only extreme values)."""
    numeric = FEATURE_TABLE.loc[FEATURE_TABLE["type"] == "numeric", "canonical"]
    rows = []
    for col in [c for c in numeric if c in data]:
        s = data[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        out = s[(s < lo) | (s > hi)]
        rows.append({"column": col, "q1": q1, "median": s.median(), "q3": q3,
                     "min": s.min(), "max": s.max(), "n_extreme": len(out),
                     "extreme_values": ", ".join(f"{v:g}" for v in sorted(out.unique())[:15])})
    return pd.DataFrame(rows).sort_values("n_extreme", ascending=False).reset_index(drop=True)


def categorical_levels(data: pd.DataFrame) -> pd.DataFrame:
    cats = FEATURE_TABLE.loc[FEATURE_TABLE["type"].isin(["binary", "categorical"]), "canonical"]
    rows = []
    for col in [c for c in cats if c in data]:
        for level, n in data[col].value_counts(dropna=False).sort_index().items():
            rows.append({"column": col, "level": level, "count": n})
    return pd.DataFrame(rows)


def class_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Median (numeric) or % = 1 (binary) per class. Descriptive only, not a test."""
    rows = []
    for _, f in FEATURE_TABLE[FEATURE_TABLE["group"].isin(MODEL_GROUPS)].iterrows():
        col = f["canonical"]
        if col not in data:
            continue
        g = data.groupby(config.TARGET)[col]
        if f["type"] == "binary":
            stat, vals = "% yes", (100 * g.mean()).round(1)
        elif f["type"] == "numeric":
            stat, vals = "median", g.median().round(2)
        else:
            continue
        rows.append({"column": col, "group": f["group"], "statistic": stat,
                     "not_pcos": vals.get(0), "pcos": vals.get(1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- IMP-05
def data_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    dd = FEATURE_TABLE.drop(columns=["match_key"]).copy()
    dd["present_in_file"] = dd["canonical"].isin(data.columns)
    dd["n_missing"] = dd["canonical"].map(lambda c: int(data[c].isna().sum()) if c in data else None)
    dd["n_unique"] = dd["canonical"].map(lambda c: int(data[c].nunique()) if c in data else None)
    dd["observed_min"] = dd["canonical"].map(lambda c: data[c].min() if c in data and pd.api.types.is_numeric_dtype(data[c]) else None)
    dd["observed_max"] = dd["canonical"].map(lambda c: data[c].max() if c in data and pd.api.types.is_numeric_dtype(data[c]) else None)
    return dd


# ---------------------------------------------------------------- figures
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d5d0")
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_balance(prof: dict, path):
    labels = ["Not PCOS (0)", "PCOS (1)"]
    values = [prof["n_not_pcos"], prof["n_pcos"]]
    total = sum(values)
    fig = Figure(figsize=(5, 3.2))
    ax = fig.subplots()
    bars = ax.bar(labels, values, color=BAR_COLOR, width=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.01, f"{v} ({100 * v / total:.1f}%)",
                ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_ylabel("Records", color=TEXT_MUTED, fontsize=9)
    ax.set_title(f"Class balance (n = {total})", color=TEXT, fontsize=11, loc="left")
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_missing(missing: pd.DataFrame, path):
    m = missing[missing["n_missing"] > 0].sort_values("n_missing")
    fig = Figure(figsize=(6, max(1.6, 0.4 * len(m) + 1)))
    ax = fig.subplots()
    if m.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", color=TEXT)
        ax.axis("off")
    else:
        bars = ax.barh(m["column"], m["n_missing"], color=BAR_COLOR, height=0.55)
        for b, n, p in zip(bars, m["n_missing"], m["pct_missing"]):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {n} ({p:.1f}%)",
                    va="center", color=TEXT, fontsize=9)
        ax.set_xlabel("Missing values", color=TEXT_MUTED, fontsize=9)
        ax.set_xlim(0, m["n_missing"].max() * 1.35)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _style(ax)
        ax.grid(axis="y", visible=False)
    ax.set_title("Columns with missing values", color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- run everything
def run_audit(path=None, write: bool = True) -> dict:
    result = load_dataset(path)
    data = result["data"]
    out = {
        "result": result,
        "profile": profile(result),
        "features_per_group": features_per_group(data),
        "missing": missing_values(data),
        "rows_with_missing": rows_with_missing(data),
        "duplicates": duplicates(data),
        "implausible": implausible_values(data),
        "derived": derived_consistency(data),
        "outliers": extreme_outliers(data),
        "categorical": categorical_levels(data),
        "class_summary": class_summary(data),
        "dictionary": data_dictionary(data),
    }
    if write:
        write_outputs(out)
    return out


def write_outputs(out: dict) -> None:
    for d in (config.AUDIT_DIR, config.FIGURES_DIR, config.DOCS, config.DATA_PROCESSED):
        d.mkdir(parents=True, exist_ok=True)
    tables = {
        "missing_values.csv": out["missing"],
        "rows_with_missing.csv": out["rows_with_missing"],
        "duplicates.csv": out["duplicates"],
        "implausible_values.csv": out["implausible"],
        "derived_column_checks.csv": out["derived"],
        "extreme_outliers.csv": out["outliers"],
        "categorical_levels.csv": out["categorical"],
        "class_summary.csv": out["class_summary"],
        "cleaning_log.csv": out["result"]["cleaning_log"],
        "column_mapping.csv": out["result"]["column_mapping"],
        "features_per_group.csv": out["features_per_group"],
    }
    for name, df in tables.items():
        df.to_csv(config.AUDIT_DIR / name, index=False)
    out["dictionary"].to_csv(config.DOCS / "data_dictionary.csv", index=False)
    out["dictionary"][["canonical", "source_name", "group", "acquisition", "autopcos_group", "type", "unit", "notes"]] \
        .to_csv(config.DOCS / "feature_groups.csv", index=False)
    out["result"]["data"].to_csv(config.DATA_PROCESSED / "kottarathil_typed.csv", index=False)
    plot_class_balance(out["profile"], config.FIGURES_DIR / "class_balance.png")
    plot_missing(out["missing"], config.FIGURES_DIR / "missing_values.png")
    (config.AUDIT_DIR / "audit_summary.md").write_text(summary_markdown(out), encoding="utf-8")


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None found._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def summary_markdown(out: dict) -> str:
    p = out["profile"]
    miss = out["missing"]
    miss_nonzero = miss[miss["n_missing"] > 0]
    log = out["result"]["cleaning_log"]
    imp = out["implausible"]
    dup = out["duplicates"]
    derived = out["derived"]
    s = [
        "# Dataset audit: Kottarathil PCOS dataset",
        f"Generated {date.today().isoformat()} by `python -m src.audit` (seed {config.SEED}). "
        "Covers tracker tasks IMP-02 to IMP-05.\n",
        "## 1. Profile (IMP-02)",
        f"- Source file: `{p['source_file']}`",
        f"- Records: **{p['n_records']}**",
        f"- Columns in file: {p['n_columns_raw']} (after dropping empty columns: {p['n_columns_after_dropping_empty']})",
        f"- Model features: **{p['n_model_features']}** (identifiers and label excluded)",
        f"- Class balance: **{p['n_pcos']} PCOS ({p['pct_pcos']}%)** vs {p['n_not_pcos']} not PCOS",
        f"- Empty columns dropped: {', '.join(p['dropped_empty_columns']) or 'none'}",
        f"- Columns not recognised by the feature dictionary: {', '.join(p['unmapped_columns']) or 'none'}\n",
        "![Class balance](figures/class_balance.png)\n",
        "### Features per group (IMP-05)",
        _md_table(out["features_per_group"][["group", "n_features"]]),
        "## 2. Missing values (IMP-03)",
        f"{len(miss_nonzero)} columns have missing values; "
        f"{len(out['rows_with_missing'])} of {p['n_records']} records have at least one missing value.\n",
        _md_table(miss_nonzero[["column", "group", "n_missing", "pct_missing", "missing_in_not_pcos", "missing_in_pcos"]]),
        "![Missing values](figures/missing_values.png)\n",
        "## 3. Type problems fixed while loading",
        _md_table(log),
        "## 4. Duplicates (IMP-04)",
        _md_table(dup[["check", "n_rows_involved"]]),
        "## 5. Values outside plausible ranges (IMP-04)",
        "Screening bounds are set in `src/feature_groups.py`. They are wide on purpose and only flag values to check, "
        "not values to delete.\n",
        _md_table(imp.groupby("column").agg(n_values=("value", "size"),
                                            values=("value", lambda v: ", ".join(f"{x:g}" for x in v)),
                                            allowed_range=("allowed_range", "first")).reset_index()
                  if not imp.empty else imp),
        "## 6. Derived-column consistency",
        _md_table(derived[["check", "n_checked", "n_mismatch_over_5pct"]] if not derived.empty else derived),
        "## 7. Extreme outliers (beyond 3 x IQR)",
        _md_table(out["outliers"].loc[out["outliers"]["n_extreme"] > 0, ["column", "median", "min", "max", "n_extreme", "extreme_values"]]),
        "## 8. Decisions to record (fill in after reviewing)",
        "- [ ] How to handle each implausible value (correct, set to missing, or keep)",
        "- [ ] Whether to drop pregnancy-related beta-HCG columns and blood group",
        "- [ ] Whether derived columns (BMI, ratios) are kept, recomputed, or dropped to avoid redundancy",
        "- [ ] Confirm the feature groups in `docs/feature_groups.csv` with your supervisor",
        "- [ ] Note any limitation for the report (single centre, label definition, sample size)",
    ]
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    o = run_audit()
    p = o["profile"]
    print(f"Records: {p['n_records']} | features: {p['n_model_features']} | "
          f"PCOS: {p['n_pcos']} ({p['pct_pcos']}%) | not PCOS: {p['n_not_pcos']}")
    print(f"Columns with missing values: {(o['missing']['n_missing'] > 0).sum()}")
    print(f"Cells fixed while loading: {len(o['result']['cleaning_log'])}")
    print(f"Implausible values flagged: {len(o['implausible'])}")
    print(f"Outputs written to {config.AUDIT_DIR}, {config.DOCS} and {config.DATA_PROCESSED}")
