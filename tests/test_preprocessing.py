"""Tests for src/preprocessing.py (W1-06).

The synthetic tests run anywhere. The real-data test runs only when
data/raw/PCOS_data_without_infertility.xlsx is present (it is never committed).

Run from the repository root:  python -m pytest -q
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import preprocessing as pp  # noqa: E402


# ----------------------------------------------------------------------------- synthetic data
GROUPS = pd.DataFrame(
    [
        ("FSH(mIU/mL)", "hormonal_biochemical", "blood_test", "continuous"),
        ("LH(mIU/mL)", "hormonal_biochemical", "blood_test", "continuous"),
        ("FSH/LH", "excluded", "blood_test", "continuous"),
        ("AMH(ng/mL)", "morphology", "blood_test", "continuous"),
        ("Follicle No. (L)", "morphology", "ultrasound", "count"),
        ("Follicle No. (R)", "morphology", "ultrasound", "count"),
        ("Avg. F size (L) (mm)", "morphology", "ultrasound", "continuous"),
        ("Avg. F size (R) (mm)", "morphology", "ultrasound", "continuous"),
        ("Pregnant(Y/N)", "excluded", "self_report", "binary"),
        ("I   beta-HCG(mIU/mL)", "excluded", "blood_test", "continuous"),
    ],
    columns=["variable", "group", "acquisition", "variable_type"],
).astype("string")


def make_df():
    """Four records: 101 normal, 102 contradictory left ovary, 103 pregnant, 104 has a 1.99 value."""
    return pd.DataFrame({
        pp.ID: [101, 102, 103, 104],
        pp.LABEL: [0, 1, 0, 1],
        "FSH(mIU/mL)": [6.0, 5.0, 4.0, 7.0],
        "LH(mIU/mL)": [3.0, 2.0, 4.0, 3.5],
        "FSH/LH": [2.0, 2.501, 1.0, 2.0],          # 102 is a rounded hand-typed ratio (2.5 exactly)
        "AMH(ng/mL)": [3.0, 9.0, 1.99, 1.99],
        "Follicle No. (L)": [6, 0, 5, 12],
        "Follicle No. (R)": [7, 9, 4, 14],
        "Avg. F size (L) (mm)": [12.0, 15.0, 10.0, 8.0],
        "Avg. F size (R) (mm)": [13.0, 14.0, 11.0, 9.0],
        "Pregnant(Y/N)": [0, 0, 1, 0],
        "I   beta-HCG(mIU/mL)": [1.99, 30.0, 300.0, 1.99],
    })


def decisions(rows):
    cols = ["id", "scope", "patient_file_no", "column", "value_in_file", "action", "new_value", "status"]
    d = pd.DataFrame(rows, columns=cols)
    d["patient_file_no"] = pd.to_numeric(d["patient_file_no"]).astype("Int64")
    return d


def write_register(root: Path, rows):
    (root / "docs").mkdir(parents=True, exist_ok=True)
    reg = decisions(rows)
    with pd.ExcelWriter(root / pp.DECISIONS_FILE) as xl:
        reg.to_excel(xl, sheet_name="decisions", startrow=3, index=False)


# ----------------------------------------------------------------------------- tests
def test_default_profile_applies_only_approved(tmp_path):
    write_register(tmp_path, [
        ("D01", "cell", 101, "AMH(ng/mL)", "3", "set_missing", None, "approved"),
        ("D02", "cell", 102, "AMH(ng/mL)", "9", "set_missing", None, "disputed"),
        ("D03", "cell", 102, "Follicle No. (L)", "0", "set_missing", None, "unresolved"),
        ("D04", "cell", 104, "AMH(ng/mL)", "1.99", "set_missing", None, "rejected"),
    ])
    assert pp.load_decisions(tmp_path)["id"].tolist() == ["D01"]
    assert pp.load_decisions(tmp_path).attrs["profile"] == "approved"
    exploratory = pp.load_decisions(tmp_path, profile="exploratory")
    assert exploratory["id"].tolist() == ["D01", "D02"]
    assert exploratory.attrs["profile"] == "exploratory"
    with pytest.raises(ValueError):
        pp.load_decisions(tmp_path, profile="everything")


def test_every_fsh_lh_overwrite_is_logged():
    d = decisions([("C01", "column", None, "FSH/LH", None, "recompute", "FSH / LH", "approved")])
    X, y, ids, log = pp.clean(make_df(), GROUPS, d)
    changed = log[log["decision"] == "C01"]
    # only record 102 differs (2.501 typed vs 2.5 recomputed), and it must be logged although < 1%
    assert changed["patient_file_no"].tolist() == [102]
    assert changed["after"].iloc[0] == pytest.approx(2.5)


def test_follicle_flags_and_variants():
    flags = pp.quality_flags(make_df())
    assert flags.loc[flags[pp.ID] == 102, "follicle_inconsistent_L"].item()
    assert flags["follicle_inconsistent"].sum() == 1

    d = decisions([])
    raw, *_ = pp.clean(make_df(), GROUPS, d)
    assert raw.loc[1, "Follicle No. (L)"] == 0            # default keeps the record as recorded

    count_only, *_ = pp.clean(make_df(), GROUPS, d, follicle_variant="count_missing")
    assert np.isnan(count_only.loc[1, "Follicle No. (L)"])
    assert count_only.loc[1, "Avg. F size (L) (mm)"] == 15.0

    both, *_ = pp.clean(make_df(), GROUPS, d, follicle_variant="count_size_missing")
    assert np.isnan(both.loc[1, "Follicle No. (L)"]) and np.isnan(both.loc[1, "Avg. F size (L) (mm)"])

    with pytest.raises(ValueError):
        pp.clean(make_df(), GROUPS, d, follicle_variant="guess")


def test_pregnancy_and_detection_limit_switches():
    d = decisions([("D01", "cell", 103, "AMH(ng/mL)", "1.99", "set_missing", None, "approved")])
    X, y, ids, log = pp.clean(make_df(), GROUPS, d, exclude_pregnant=True)
    assert 103 not in ids.tolist() and len(X) == 3       # decision for an excluded record is skipped
    assert (log["decision"] == "S-PREG").sum() == 1

    X, y, ids, log = pp.clean(make_df(), GROUPS, decisions([]), lod_variant="missing")
    assert X["AMH(ng/mL)"].isna().sum() == 2              # both 1.99 AMH values set to missing

    flags = pp.quality_flags(make_df())
    assert flags.loc[flags[pp.ID] == 102, "hcg_mismatch"].item()   # not pregnant, beta-HCG 30


def test_feature_dictionary_inputs():
    groups = pp.load_feature_groups(ROOT)
    model = groups.loc[groups["group"] != "excluded", "variable"].tolist()
    assert len(groups) == 41 and len(model) == 34
    for col in ["Height(Cm)", "Marraige Status (Yrs)", "Pregnant(Y/N)", "FSH/LH", "Blood Group"]:
        assert col not in model


@pytest.mark.skipif(not (ROOT / pp.DATA_FILE).exists(), reason="raw dataset not present (never committed)")
def test_real_data_counts():
    df = pp.load_raw(ROOT)
    groups = pp.load_feature_groups(ROOT)
    X, y, ids, log = pp.clean(df, groups, pp.load_decisions(ROOT))
    assert X.shape == (541, 34)
    assert y.sum() == 177
    assert set(log["profile"]) == {"approved"}
    assert (log["decision"] == "C01").sum() == 9          # every rounded FSH/LH ratio is logged
    flags = pp.quality_flags(df)
    assert flags["follicle_inconsistent"].sum() == 13
    assert flags["pregnant"].sum() == 206
