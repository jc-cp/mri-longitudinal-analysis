"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

# Input
CSV_FILE = Path("/home/jc053/GIT/mri-longitudinal-analysis/data/redcap/redcap_full_89_cohort.csv")

# Output directory
OUTPUT_DIR = Path("/home/jc053/GIT/mri-longitudinal-analysis/data/output/clinical_data")

# Plots of clinical data
VISUALIZE_DATA = True
OUTPUT_TREATMENT = OUTPUT_DIR / "treatment_clinical_data.png"
TREATMENT_PLOT = True
OUTPUT_MUTATION = OUTPUT_DIR / "mutational_clinical_data.png"
MUTATION_PLOT = True
OUTPUT_DIAGNOSIS = OUTPUT_DIR / "diagnosis_clinical_data.png"
DIAGNOSIS_PLOT = True

# Output file
OUTPUT_FILE_NAME = OUTPUT_DIR / "output.txt"
OUTPUT_FILE = True

# Deleting after surgery data points
DELETING_SURGERY = False
DATA_DIR = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/"
    "curated_no_ops_29_surgery_cohort_reviewed/1_no_comments"
)
