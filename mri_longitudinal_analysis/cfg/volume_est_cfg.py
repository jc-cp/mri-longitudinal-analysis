"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

# Directories
SEG_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_longitudinal_dataset_new/accepted"
)
OUTPUT_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_longitudinal_dataset_new/output"
)
PLOTS_DIR = OUTPUT_DIR / "volume_plots"
CSV_DIR = OUTPUT_DIR / "time_series_csv_kernel_smoothed"

# Individual files
REDCAP_FILE = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/redcap/redcap_full_108_cohort.csv"
)
FEW_SCANS_FILE = OUTPUT_DIR / "patients_with_few_scans.txt"
ZERO_VOLUME_FILE = OUTPUT_DIR / "zero_volume_segmentations.txt"
NUMBER_TOTAL_PATIENTS = 108

# Options for creating the plots
LIMIT_LOADING = 108

RAW = True
FILTERED = True
POLY_SMOOTHING = True
KERNEL_SMOOTHING = True
WINDOW_SMOOTHING = True
BANDWIDTH = 200
PLOT_COMPARISON = True

# If test data is being used or not
TEST_DATA = False

# Other options
NORMALIZE = False
CONFIDENCE_INTERVAL = True
