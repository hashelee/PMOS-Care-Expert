"""Feature dictionary for the Kottarathil PCOS dataset (tracker task IMP-05).

This is the single source of truth for every column:
  - the canonical snake_case name used in code
  - the original column name in the Kaggle file
  - its feature group (clinical / hormonal_biochemical / metabolic / ultrasound_morphology)
  - how it is acquired in practice (self-report, physical exam, blood test, ultrasound)
  - an approximate AutoPCOS-style group (clinical / laboratory / ultrasound) for the
    B1-B4 information scenarios
  - its type and a wide plausibility range used only to FLAG suspicious values

Grouping decisions that you should confirm with your supervisor are marked
"REVIEW" in the notes. The plausibility ranges are deliberately wide screening
bounds, not clinical reference ranges.
"""
from __future__ import annotations

import re

import pandas as pd


def match_key(name: str) -> str:
    """Normalise a column name so small spelling/spacing differences still match.

    ' Age (yrs)' -> 'ageyrs', 'II    beta-HCG(mIU/mL)' -> 'iibetahcgmiuml'
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# fmt: off
# (canonical, original name in Kaggle file, group, acquisition, autopcos_group, type, unit, low, high, notes)
_FEATURES = [
    # identifiers and label
    ("sl_no",             "Sl. No",                  "identifier", "id", None, "id", "", None, None, "Row number. Never used as a feature."),
    ("patient_file_no",   "Patient File No.",        "identifier", "id", None, "id", "", None, None, "Patient identifier. Never used as a feature; used for duplicate checks."),
    ("pcos",              "PCOS (Y/N)",              "target", "label", None, "binary", "", None, None, "Label: 1 = PCOS, 0 = not PCOS."),

    # clinical: demographics, history, symptoms, lifestyle, vital signs
    ("age",               "Age (yrs)",               "clinical", "self_report", "clinical", "numeric", "years", 10, 60, ""),
    ("blood_group",       "Blood Group",             "clinical", "blood_test", "clinical", "categorical", "code", None, None, "Numeric codes (11-18) for ABO/Rh groups. REVIEW: low clinical relevance to PCOS."),
    ("pulse_rate",        "Pulse rate(bpm)",         "clinical", "physical_exam", "clinical", "numeric", "bpm", 40, 150, ""),
    ("resp_rate",         "RR (breaths/min)",        "clinical", "physical_exam", "clinical", "numeric", "breaths/min", 8, 40, ""),
    ("cycle_regularity",  "Cycle(R/I)",              "clinical", "self_report", "clinical", "categorical", "code", None, None, "Regular vs irregular cycle. Check the codes in the value-count table (expected 2 = regular, 4 = irregular)."),
    ("cycle_length",      "Cycle length(days)",      "clinical", "self_report", "clinical", "numeric", "days", 0, 60, "REVIEW: values look like duration of bleeding, not cycle length. Check the distribution."),
    ("marriage_years",    "Marraige Status (Yrs)",   "clinical", "self_report", "clinical", "numeric", "years", 0, 45, "Years married (spelling as in source)."),
    ("pregnant",          "Pregnant(Y/N)",           "clinical", "self_report", "clinical", "binary", "", None, None, ""),
    ("n_abortions",       "No. of abortions",        "clinical", "self_report", "clinical", "numeric", "count", 0, 15, ""),
    ("weight_gain",       "Weight gain(Y/N)",        "clinical", "self_report", "clinical", "binary", "", None, None, "Symptom."),
    ("hair_growth",       "hair growth(Y/N)",        "clinical", "self_report", "clinical", "binary", "", None, None, "Hirsutism symptom (clinical hyperandrogenism)."),
    ("skin_darkening",    "Skin darkening (Y/N)",    "clinical", "self_report", "clinical", "binary", "", None, None, "Acanthosis-type symptom."),
    ("hair_loss",         "Hair loss(Y/N)",          "clinical", "self_report", "clinical", "binary", "", None, None, "Symptom."),
    ("pimples",           "Pimples(Y/N)",            "clinical", "self_report", "clinical", "binary", "", None, None, "Acne symptom (clinical hyperandrogenism)."),
    ("fast_food",         "Fast food (Y/N)",         "clinical", "self_report", "clinical", "binary", "", None, None, "Lifestyle."),
    ("regular_exercise",  "Reg.Exercise(Y/N)",       "clinical", "self_report", "clinical", "binary", "", None, None, "Lifestyle."),

    # metabolic: anthropometry, blood pressure, glucose
    ("weight",            "Weight (Kg)",             "metabolic", "physical_exam", "clinical", "numeric", "kg", 25, 200, ""),
    ("height",            "Height(Cm)",              "metabolic", "physical_exam", "clinical", "numeric", "cm", 120, 200, "REVIEW: height is anthropometric; kept with metabolic because BMI is derived from it."),
    ("bmi",               "BMI",                     "metabolic", "physical_exam", "clinical", "numeric", "kg/m2", 10, 70, "Derived from weight and height. Check it matches."),
    ("hip",               "Hip(inch)",               "metabolic", "physical_exam", "clinical", "numeric", "inch", 20, 70, ""),
    ("waist",             "Waist(inch)",             "metabolic", "physical_exam", "clinical", "numeric", "inch", 20, 70, ""),
    ("waist_hip_ratio",   "Waist:Hip Ratio",         "metabolic", "physical_exam", "clinical", "numeric", "ratio", 0.5, 1.5, "Derived from waist and hip."),
    ("bp_systolic",       "BP _Systolic (mmHg)",     "metabolic", "physical_exam", "clinical", "numeric", "mmHg", 70, 200, ""),
    ("bp_diastolic",      "BP _Diastolic (mmHg)",    "metabolic", "physical_exam", "clinical", "numeric", "mmHg", 40, 130, ""),
    ("rbs",               "RBS(mg/dl)",              "metabolic", "blood_test", "laboratory", "numeric", "mg/dl", 40, 600, "Random blood sugar."),

    # hormonal / biochemical: laboratory assays
    ("hb",                "Hb(g/dl)",                "hormonal_biochemical", "blood_test", "laboratory", "numeric", "g/dl", 5, 20, "Haemoglobin."),
    ("beta_hcg_1",        "I beta-HCG(mIU/mL)",      "hormonal_biochemical", "blood_test", "laboratory", "numeric", "mIU/mL", 0, None, "Pregnancy hormone. REVIEW: likely low relevance to PCOS."),
    ("beta_hcg_2",        "II beta-HCG(mIU/mL)",     "hormonal_biochemical", "blood_test", "laboratory", "numeric", "mIU/mL", 0, None, "Second beta-HCG measurement. Often stored as text in the source."),
    ("fsh",               "FSH(mIU/mL)",             "hormonal_biochemical", "blood_test", "laboratory", "numeric", "mIU/mL", 0, 200, ""),
    ("lh",                "LH(mIU/mL)",              "hormonal_biochemical", "blood_test", "laboratory", "numeric", "mIU/mL", 0, 200, ""),
    ("fsh_lh_ratio",      "FSH/LH",                  "hormonal_biochemical", "blood_test", "laboratory", "numeric", "ratio", 0, None, "Derived from FSH and LH."),
    ("tsh",               "TSH (mIU/L)",             "hormonal_biochemical", "blood_test", "laboratory", "numeric", "mIU/L", 0, 100, ""),
    ("amh",               "AMH(ng/mL)",              "hormonal_biochemical", "blood_test", "laboratory", "numeric", "ng/mL", 0, 60, "Often stored as text in the source. AMH can stand in for ultrasound morphology in adults (2023 guideline)."),
    ("prl",               "PRL(ng/mL)",              "hormonal_biochemical", "blood_test", "laboratory", "numeric", "ng/mL", 0, 200, "Prolactin."),
    ("vit_d3",            "Vit D3 (ng/mL)",          "hormonal_biochemical", "blood_test", "laboratory", "numeric", "ng/mL", 0, 150, ""),
    ("prg",               "PRG(ng/mL)",              "hormonal_biochemical", "blood_test", "laboratory", "numeric", "ng/mL", 0, 100, "Progesterone."),

    # ultrasound morphology: technician-reported measurements (not raw images)
    ("follicle_no_left",  "Follicle No. (L)",        "ultrasound_morphology", "ultrasound", "ultrasound", "numeric", "count", 0, 60, "Follicle count, left ovary. Your follicle-quantification module would produce this automatically."),
    ("follicle_no_right", "Follicle No. (R)",        "ultrasound_morphology", "ultrasound", "ultrasound", "numeric", "count", 0, 60, "Follicle count, right ovary."),
    ("avg_f_size_left",   "Avg. F size (L) (mm)",    "ultrasound_morphology", "ultrasound", "ultrasound", "numeric", "mm", 0, 40, "Average follicle size, left ovary."),
    ("avg_f_size_right",  "Avg. F size (R) (mm)",    "ultrasound_morphology", "ultrasound", "ultrasound", "numeric", "mm", 0, 40, "Average follicle size, right ovary."),
    ("endometrium",       "Endometrium (mm)",        "ultrasound_morphology", "ultrasound", "ultrasound", "numeric", "mm", 0, 30, "Endometrial thickness."),
]
# fmt: on

_COLUMNS = [
    "canonical", "source_name", "group", "acquisition", "autopcos_group",
    "type", "unit", "plausible_low", "plausible_high", "notes",
]

FEATURE_TABLE = pd.DataFrame(_FEATURES, columns=_COLUMNS)
FEATURE_TABLE["match_key"] = FEATURE_TABLE["source_name"].map(match_key)

MODEL_GROUPS = ["clinical", "metabolic", "hormonal_biochemical", "ultrasound_morphology"]


def features_in_group(group: str) -> list[str]:
    """Canonical names of all model features in one group."""
    return FEATURE_TABLE.loc[FEATURE_TABLE["group"] == group, "canonical"].tolist()


def model_features() -> list[str]:
    """All columns that may be used as model inputs (excludes identifiers and label)."""
    return FEATURE_TABLE.loc[FEATURE_TABLE["group"].isin(MODEL_GROUPS), "canonical"].tolist()


def feature_info(canonical: str) -> pd.Series:
    return FEATURE_TABLE.set_index("canonical").loc[canonical]
