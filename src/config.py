"""Project-wide paths and constants.

Every script and notebook imports from here, so paths and the random seed are
defined in exactly one place.
"""
from pathlib import Path

# Fixed random seed used for every split, model and resampling step.
SEED = 42

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
AUDIT_DIR = RESULTS / "audit"
FIGURES_DIR = AUDIT_DIR / "figures"
DOCS = ROOT / "docs"

# Kaggle: "Polycystic ovary syndrome (PCOS)" by Prasoon Kottarathil.
# The main file is PCOS_data_without_infertility.xlsx, sheet "Full_new".
KOTTARATHIL_URL = (
    "https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos"
)
KOTTARATHIL_SHEET = "Full_new"

TARGET = "pcos"          # canonical name of the label column (1 = PCOS, 0 = not PCOS)
PATIENT_ID = "patient_file_no"
