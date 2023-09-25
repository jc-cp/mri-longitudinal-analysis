"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

# Directories
SEG_DIR = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/"
    "curated_no_ops_29_surgery_cohort_reviewed/output/seg_predictions"
)
OUTPUT_DIR = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/"
    "curated_no_ops_29_surgery_cohort_reviewed/output"
)
PLOTS_DIR = OUTPUT_DIR / "volume_plots"
CSV_DIR = OUTPUT_DIR / "time_series_csv"

# Individual files
REDCAP_FILE = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/redcap/redcap_full_89_cohort.csv"
)
FEW_SCANS_FILE = OUTPUT_DIR / "patients_with_few_scans.txt"

# Options for creating the plots
LIMIT_LOADING = None
KERNEL_SMOOTHING = False
POLY_SMOOTHING = False
NORMALIZATION = False

# If test data is being used or not
TEST_DATA = False
