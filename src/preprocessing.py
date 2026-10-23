"""Reproducible preprocessing for the Kottarathil PCOS dataset (tracker task IMP-06).

Two stages, kept strictly apart to avoid data leakage:

  Stage A  clean()               Rule-based fixes taken from docs/cleaning_decisions.xlsx and
                                 docs/feature_groups.xlsx. Each rule looks at one record only and
                                 learns nothing from the data, so it is applied once, before any split.

  Stage B  build_preprocessor()  Learned steps (median imputation, missing indicators, log transform,
                                 scaling). Returned as a scikit-learn Pipeline so that cross-validation
                                 refits it on the training folds only.

Cleaning profiles (W1-06):
  "approved"     default; applies only decisions with status 'approved'. Use for all reported results.
  "exploratory"  also applies 'disputed' decisions, to see what they would change. Never report these
                 results as the main analysis; the profile name is written into the change log.
  'unresolved' and 'rejected' decisions are never applied. Unresolved follicle records (D17-D29) are
  instead found by rule in quality_flags() and handled through the follicle_variant switch of clean().

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
APPLIED_STATUSES = ("approved",)   # default: only decisions the evidence settles are applied
PROFILES = {
    "approved": ("approved",),
    "exploratory": ("approved", "disputed"),
}

PREGNANT = "Pregnant(Y/N)"
HCG = "I   beta-HCG(mIU/mL)"
# Warning threshold only (quality_flags); it never changes data. Check against a cited source before
# using it in any write-up.
HCG_WARNING_LEVEL = 25.0
LOD_VALUE = 1.99   # value treated as a possible laboratory detection limit (decision C04)
FOLLICLES = {  # ovary: (count column, average-size column)
    "L": ("Follicle No. (L)", "Avg. F size (L) (mm)"),
    "R": ("Follicle No. (R)", "Avg. F size (R) (mm)"),
}
FOLLICLE_VARIANTS = ("raw", "count_missing", "count_size_missing")
LOD_VARIANTS = ("keep", "missing")


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


def load_decisions(root: Path, statuses=None, profile: str = "approved") -> pd.DataFrame:
    """docs/cleaning_decisions.xlsx, keeping only decisions whose status should be applied.

    By default the 'approved' profile is used. Pass profile="exploratory" to add 'disputed' decisions,
    or an explicit tuple of statuses (e.g. to list every decision); the profile then reads "custom".
    """
    if statuses is None:
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {sorted(PROFILES)}")
        statuses = PROFILES[profile]
    else:
        profile = "custom"
    d = pd.read_excel(Path(root) / DECISIONS_FILE, sheet_name="decisions", header=3, dtype={"new_value": "string"})
    d = d.dropna(how="all")
    for col in ["scope", "column", "action", "status"]:
        d[col] = d[col].astype("string").str.strip()
    d["patient_file_no"] = pd.to_numeric(d["patient_file_no"], errors="coerce").astype("Int64")
    d = d[d["status"].isin(statuses)].reset_index(drop=True)
    d.attrs["profile"] = profile
    return d


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Data-quality flags found by rule, one row per record. Never used as model inputs.

    follicle_inconsistent_L / _R   follicle count is 0 but an average follicle size above 0 is recorded
                                   (the contradiction behind decisions D17-D29)
    follicle_inconsistent          either ovary inconsistent
    pregnant                       Pregnant(Y/N) = 1
    hcg_mismatch                   not marked pregnant, but beta-HCG I >= HCG_WARNING_LEVEL (warning only)
    lod_value_count                number of laboratory cells equal to LOD_VALUE in the record
    """
    num = lambda col: pd.to_numeric(df[col], errors="coerce")
    flags = pd.DataFrame({ID: df[ID]})
    for side, (count_col, size_col) in FOLLICLES.items():
        flags[f"follicle_inconsistent_{side}"] = (num(count_col) == 0) & (num(size_col) > 0)
    flags["follicle_inconsistent"] = flags["follicle_inconsistent_L"] | flags["follicle_inconsistent_R"]
    flags["pregnant"] = num(PREGNANT) == 1
    flags["hcg_mismatch"] = ~flags["pregnant"] & (num(HCG) >= HCG_WARNING_LEVEL)
    lab = [c for c in df.columns if "mIU" in c or "ng/mL" in c or "mg/dl" in c or "g/dl" in c]
    flags["lod_value_count"] = (df[lab].apply(pd.to_numeric, errors="coerce") == LOD_VALUE).sum(axis=1)
    return flags


# ----------------------------------------------------------------------------------- stage A
def clean(df: pd.DataFrame, groups: pd.DataFrame, decisions: pd.DataFrame,
          follicle_variant: str = "raw", lod_variant: str = "keep", exclude_pregnant: bool = False):
    """Apply the rule-based decisions. Returns (X, y, ids, log).

    X    model features only (excluded and non-feature columns removed), all numeric
    y    the label (1 = PCOS, 0 = not PCOS)
    ids  Patient File No., kept separately to trace records
    log  one row per value that was changed (or record excluded), for the report

    Sensitivity switches (defaults change nothing):
    follicle_variant  "raw"                 keep contradictory follicle records as recorded (default)
                      "count_missing"       set the count to missing where count = 0 but size > 0
                      "count_size_missing"  set both count and size to missing in those ovaries
    lod_variant       "keep"                keep 1.99 as a measured value (default, decision C04)
                      "missing"             set laboratory values of 1.99 to missing
    exclude_pregnant  False                 keep all records (default)
                      True                  drop records with Pregnant(Y/N) = 1
    """
    if follicle_variant not in FOLLICLE_VARIANTS:
        raise ValueError(f"follicle_variant must be one of {FOLLICLE_VARIANTS}")
    if lod_variant not in LOD_VARIANTS:
        raise ValueError(f"lod_variant must be one of {LOD_VARIANTS}")
    df = df.copy()
    log = []
    profile = decisions.attrs.get("profile", "custom")

    def record(pid, column, old, new, decision_id, rule):
        log.append({"decision": decision_id, "patient_file_no": pid, "column": column,
                    "before": old, "after": new, "rule": rule})

    # 0. Optional eligibility filter (sensitivity analysis), applied before anything else
    if exclude_pregnant:
        pregnant = pd.to_numeric(df[PREGNANT], errors="coerce") == 1
        for pid in df.loc[pregnant, ID]:
            record(pid, PREGNANT, 1, None, "S-PREG", "record excluded: pregnant (sensitivity analysis)")
        excluded_ids = set(df.loc[pregnant, ID])
        df = df.loc[~pregnant].reset_index(drop=True)
    else:
        excluded_ids = set()

    unknown = set(decisions["column"]) - set(df.columns) - {"(all lab columns)"}
    if unknown:
        raise ValueError(f"Decisions refer to columns not in the dataset: {sorted(unknown)}")

    # 1. Cell-level decisions, applied to the values exactly as stored (before any conversion)
    row_of = dict(zip(df[ID], df.index))
    recompute_bmi = []
    for _, d in decisions[decisions["scope"] == "cell"].iterrows():
        idx = row_of.get(int(d["patient_file_no"]))
        if idx is None and int(d["patient_file_no"]) in excluded_ids:
            continue   # record removed by the pregnancy filter
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

    # 2b. Sensitivity switches that work on numeric values
    if follicle_variant != "raw":
        for side, (count_col, size_col) in FOLLICLES.items():
            bad = (df[count_col] == 0) & (df[size_col] > 0)
            cols = [count_col] if follicle_variant == "count_missing" else [count_col, size_col]
            for idx in df.index[bad]:
                for col in cols:
                    record(df.at[idx, ID], col, df.at[idx, col], np.nan, "S-FOLL",
                           f"follicle record inconsistent ({follicle_variant})")
                    df.at[idx, col] = np.nan
    if lod_variant == "missing":
        lab_cols = groups.loc[groups["acquisition"] == "blood_test", "variable"].tolist()
        for col in lab_cols:
            for idx in df.index[df[col] == LOD_VALUE]:
                record(df.at[idx, ID], col, LOD_VALUE, np.nan, "S-LOD", "possible detection limit set to missing")
                df.at[idx, col] = np.nan

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
            # log every overwritten value, including hand-typed ratios that were only rounded;
            # differences at floating-point precision (rtol 1e-9) are not real changes
            changed = ~np.isclose(new, df["FSH/LH"], rtol=1e-9, atol=0, equal_nan=True)
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
    log = pd.DataFrame(log, columns=["decision", "patient_file_no", "column", "before", "after", "rule"])
    log["patient_file_no"] = pd.to_numeric(log["patient_file_no"], errors="coerce").astype("Int64")
    log["profile"] = profile
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
