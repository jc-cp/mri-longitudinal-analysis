"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

# Directories and files
SEG_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/qa"
)
OUTPUT_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/output"
)
PLOTS_DIR = OUTPUT_DIR / "volume_plots"
CSV_DIR = OUTPUT_DIR / "time_series_csv_kernel_smoothed"
ZERO_VOLUME_FILE = OUTPUT_DIR / "zero_volume_segmentations.txt"
FEW_SCANS_FILE = OUTPUT_DIR / "few_scans_patients.txt"

# Clinical data files
CLINICAL_DATA_FILE = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/cbtn_filtered_and_pruned_513.csv"
)
# If test data is being used or not
TEST_DATA = False
# Current dataset being used
BCH_DATA = False
CBTN_DATA = True
NUMBER_TOTAL_CBTN_PATIENTS = 115
NUMBER_TOTAL_BCH_PATIENTS = 85

# Filtering options
RAW = True
FILTERED = True
POLY_SMOOTHING = True
KERNEL_SMOOTHING = True
WINDOW_SMOOTHING = True
BANDWIDTH = 200
PLOT_COMPARISON = True

# Other options
CONFIDENCE_INTERVAL = True

# Growth pattern output
RAPID_GROWTH = 20
MODERATE_GROWTH = 10

# Growth type 
R2_THRESHOLD = 0.5
HIGH_VAR = 10