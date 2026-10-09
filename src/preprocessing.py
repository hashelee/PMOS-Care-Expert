"""Reproducible preprocessing for the Kottarathil PCOS dataset (tracker task IMP-06).

Two stages, kept strictly apart to avoid data leakage:

  Stage A  clean()               Rule-based fixes taken from docs/cleaning_decisions.xlsx and
                                 docs/feature_groups.xlsx. Each rule looks at one record only and
                                 learns nothing from the data, so it is applied once, before any split.

  Stage B  build_preprocessor()  Learned steps (median imputation, missing indicators, log transform,
                                 scaling). Returned as a scikit-learn Pipeline so that cross-validation
                                 refits it on the training folds only.

Every later notebook imports this file, so the preprocessing is written once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DATA_FILE = Path("data") / "raw" / "PCOS_data_without_infertility.xlsx"
GROUPS_FILE = Path("docs") / "feature_groups.xlsx"
DECISIONS_FILE = Path("docs") / "cleaning_decisions.xlsx"

LABEL = "PCOS (Y/N)"
ID = "Patient File No."
APPLIED_STATUSES = ("approved", "proposed")   # 'rejected' decisions are never applied


# ----------------------------------------------------------------------------------- loading
def load_raw(root: Path) -> pd.DataFrame:
    """The Full_new sheet exactly as stored, with hidden spaces removed from column names."""
    df = pd.read_excel(Path(root) / DATA_FILE, sheet_name="Full_new")
    df.columns = df.columns.str.strip()
    return df


def load_feature_groups(root: Path) -> pd.DataFrame:
    """docs/feature_groups.xlsx: one row per feature with group, acquisition and variable_type."""
    groups = pd.read_excel(Path(root) / GROUPS_FILE, sheet_name="feature_groups", header=3)
    groups = groups[["variable", "group", "acquisition", "variable_type"]].dropna(how="all")
    return groups.apply(lambda s: s.astype("string").str.strip())


def load_decisions(root: Path, statuses=APPLIED_STATUSES) -> pd.DataFrame:
    """docs/cleaning_decisions.xlsx, keeping only decisions whose status should be applied."""
    d = pd.read_excel(Path(root) / DECISIONS_FILE, sheet_name="decisions", header=3, dtype={"new_value": "string"})
    d = d.dropna(how="all")
    for col in ["scope", "column", "action", "status"]:
        d[col] = d[col].astype("string").str.strip()
    d["patient_file_no"] = pd.to_numeric(d["patient_file_no"], errors="coerce").astype("Int64")
    return d[d["status"].isin(statuses)].reset_index(drop=True)


# ----------------------------------------------------------------------------------- stage A
def clean(df: pd.DataFrame, groups: pd.DataFrame, decisions: pd.DataFrame):
    """Apply the rule-based decisions. Returns (X, y, ids, log).

    X    model features only (excluded and non-feature columns removed), all numeric
    y    the label (1 = PCOS, 0 = not PCOS)
    ids  Patient File No., kept separately to trace records
    log  one row per value that was changed, for the report
    """
    df = df.copy()
    log = []

    def record(pid, column, old, new, decision_id, rule):
        log.append({"decision": decision_id, "patient_file_no": pid, "column": column,
                    "before": old, "after": new, "rule": rule})

    unknown = set(decisions["column"]) - set(df.columns) - {"(all lab columns)"}
    if unknown:
        raise ValueError(f"Decisions refer to columns not in the dataset: {sorted(unknown)}")

    # 1. Cell-level decisions, applied to the values exactly as stored (before any conversion)
    row_of = dict(zip(df[ID], df.index))
    recompute_bmi = []
    for _, d in decisions[decisions["scope"] == "cell"].iterrows():
        idx = row_of.get(int(d["patient_file_no"]))
        if idx is None:
            raise ValueError(f"{d['id']}: patient {d['patient_file_no']} not found")
        old = df.at[idx, d["column"]]
        if d["action"] == "set_missing":
            df.at[idx, d["column"]] = np.nan
            record(d["patient_file_no"], d["column"], old, np.nan, d["id"], "set to missing")
        elif d["action"] == "correct":
            df.at[idx, d["column"]] = float(d["new_value"])
            record(d["patient_file_no"], d["column"], old, float(d["new_value"]), d["id"], "corrected")
        elif d["action"] == "recompute" and d["column"] == "BMI":
            recompute_bmi.append((idx, d["id"]))
        # 'keep' needs no change

    # 2. Every feature to a number; any text not covered by a decision becomes missing and is logged
    features = groups["variable"].tolist()
    for col in features:
        converted = pd.to_numeric(df[col], errors="coerce")
        for idx in df.index[converted.isna() & df[col].notna()]:
            record(df.at[idx, ID], col, df.at[idx, col], np.nan, "-", "text not covered by a decision: set to missing")
        df[col] = converted

    # 3. Cell-level recomputations that need numeric inputs
    for idx, decision_id in recompute_bmi:
        old = df.at[idx, "BMI"]
        new = round(df.at[idx, "Weight (Kg)"] / (df.at[idx, "Height(Cm)"] / 100) ** 2, 2)
        df.at[idx, "BMI"] = new
        record(df.at[idx, ID], "BMI", old, new, decision_id, "recomputed from weight and height")

    # 4. Column-level decisions
    for _, d in decisions[decisions["scope"] == "column"].iterrows():
        if d["action"] == "recompute" and d["column"] == "FSH/LH":
            new = df["FSH(mIU/mL)"] / df["LH(mIU/mL)"]
            # log only real changes (> 1%); hand-typed ratios that were merely rounded are replaced silently
            changed = ~np.isclose(new, df["FSH/LH"], rtol=0.01, equal_nan=True)
            for idx in df.index[changed]:
                record(df.at[idx, ID], "FSH/LH", df.at[idx, "FSH/LH"], new[idx], d["id"], "recomputed from FSH and LH")
            df["FSH/LH"] = new
        elif d["action"] == "recode" and d["column"] == "Cycle(R/I)":
            df["Cycle(R/I)"] = df["Cycle(R/I)"].map({2: 0, 4: 1})   # anything else becomes missing
            log.append({"decision": d["id"], "patient_file_no": None, "column": "Cycle(R/I)",
                        "before": "2 / 4", "after": "0 / 1", "rule": "recoded (whole column)"})

    # 5. Keep only model features; separate label and IDs
    model_features = groups.loc[groups["group"] != "excluded", "variable"].tolist()
    X = df[model_features].astype(float)
    y = df[LABEL].astype(int)
    ids = df[ID]
    log = pd.DataFrame(log)
    log["patient_file_no"] = pd.to_numeric(log["patient_file_no"], errors="coerce").astype("Int64")
    return X, y, ids, log


# ----------------------------------------------------------------------------------- stage B
def build_preprocessor(groups: pd.DataFrame, model_family: str) -> Pipeline:
    """Learned preprocessing for one model family, as an unfitted scikit-learn Pipeline.

    model_family:
      "linear"  Logistic Regression, SVM: impute + missing indicators, log-transform skewed lab
                values, scale continuous and count variables
      "tree"    Random Forest: impute + missing indicators only (trees ignore scale)
      "native"  XGBoost, CatBoost: no learned steps; they handle missing values themselves
    """
    g = groups[groups["group"] != "excluded"]
    binary = g.loc[g["variable_type"].isin(["binary", "nominal"]), "variable"].tolist()
    numeric = g.loc[g["variable_type"].isin(["continuous", "count"]), "variable"].tolist()
    # Skewed laboratory values: continuous blood-test results
    log_cols = g.loc[(g["acquisition"] == "blood_test") & (g["variable_type"] == "continuous"), "variable"].tolist()
    other_numeric = [c for c in numeric if c not in log_cols]

    if model_family == "native":
        return Pipeline([("passthrough", FunctionTransformer(feature_names_out="one-to-one"))]).set_output(transform="pandas")

    impute_median = lambda: SimpleImputer(strategy="median", add_indicator=True)
    impute_mode = lambda: SimpleImputer(strategy="most_frequent", add_indicator=True)

    if model_family == "tree":
        transformers = [
            ("numeric", impute_median(), numeric),
            ("binary", impute_mode(), binary),
        ]
    elif model_family == "linear":
        log_then_scale = Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("impute", impute_median()),
            ("scale", StandardScaler()),
        ])
        scale = Pipeline([("impute", impute_median()), ("scale", StandardScaler())])
        transformers = [
            ("lab_log", log_then_scale, log_cols),
            ("numeric", scale, other_numeric),
            ("binary", impute_mode(), binary),
        ]
    else:
        raise ValueError("model_family must be 'linear', 'tree' or 'native'")

    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("preprocess", preprocessor)]).set_output(transform="pandas")
