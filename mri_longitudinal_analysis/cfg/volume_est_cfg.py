"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

# Directories
SEG_DIR = Path("/mnt/kannlab_rfa/JuanCarlos/fuckupcheck/juancarloseg")
OUTPUT_DIR = Path("/mnt/kannlab_rfa/JuanCarlos/fuckupcheck/output")
PLOTS_DIR = OUTPUT_DIR / "volume_plots"
CSV_DIR = OUTPUT_DIR / "time_series_csv"

# Individual files
REDCAP_FILE = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/redcap/redcap_full_89_cohort.csv"
)
FEW_SCANS_FILE = OUTPUT_DIR / "patients_with_few_scans.txt"

NUMBER_TOTAL_PATIENTS = 89

# Options for creating the plots
LIMIT_LOADING = None

FILTERED = True
POLY_SMOOTHING = True
POLY_SMOOTHING_DEGREE = 5
KERNEL_SMOOTHING = True
BANDWIDTH = 200
PLOT_COMPARISON = True

# If test data is being used or not
TEST_DATA = False
