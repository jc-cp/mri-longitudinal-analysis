from pathlib import Path

# Input
CSV_FILE = Path(
    "/mnt/c/Users/jccli/Documents/GIT/mri-longitudinal-segmentation/data/redcap/redcap_no_ops_cohort_full_89.csv"
)

# Outputs
OUTPUT_DIR = Path(
    "/mnt/c/Users/jccli/Documents/GIT/mri-longitudinal-segmentation/output"
)
OUTPUT_TREATMENT = OUTPUT_DIR / "treatment_clinical_data.png"
TREATMENT_PLOT = True
OUTPUT_MUTATION = OUTPUT_DIR / "mutational_clinical_data.png"
MUTATION_PLOT = True
OUTPUT_DIAGNOSIS = OUTPUT_DIR / "diagnosis_clinical_data.png"
DIAGNOSIS_PLOT = True
OUTPUT_FILE_NAME = OUTPUT_DIR / "output.txt"
OUTPUT_FILE = True

# Deleting after surgery data points
DELETING_SURGERY = True
# DATA_DIR = Path()
