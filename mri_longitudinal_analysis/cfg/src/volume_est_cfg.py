"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

# Directories and files
BCH = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/bch_longitudinal_dataset/final")
CBTN = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/accepted/pre_treatment")
JOINT = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/final_dataset")

# Outputs
SEG_DIR = JOINT # change this line
OUTPUT_DIR = SEG_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "volume_plots"
CSV_DIR = OUTPUT_DIR / "time_series"
ZERO_VOLUME_FILE = OUTPUT_DIR / "zero_volume_segmentations.txt"
FEW_SCANS_FILE = OUTPUT_DIR / "few_scans_patients.txt"
HIGH_VOLUME_FILE = OUTPUT_DIR / "high_volume_segmentations.txt"
VOLUME_THRESHOLD = 14000 

# Clinical data files
CLINICAL_DATA_FILE_CBTN = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/cbtn_filtered_pruned_treatment_513.csv"
)
CLINICAL_DATA_FILE_BCH = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/input/clinical/bch_filtering_68_.csv")

# If test data is being used or not
TEST_DATA = False
# Current dataset being used
BCH_DATA = False
CBTN_DATA = False
JOINT_DATA = True
NUMBER_TOTAL_CBTN_PATIENTS = 45
NUMBER_TOTAL_BCH_PATIENTS = 66
NUMBER_TOTAL_JOINT_PATIENTS = 111

# Filtering options
RAW = True
FILTERED = True
POLY_SMOOTHING = True
KERNEL_SMOOTHING = True
WINDOW_SMOOTHING = True
MOVING_AVERAGE = True
PLOT_COMPARISON = True

# Other options
CONFIDENCE_INTERVAL = True