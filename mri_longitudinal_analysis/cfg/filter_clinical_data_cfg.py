"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

# Input
CSV_FILE_BCH = Path(
    "/home/juanqui55/git/mri-longitudinal-analysis/data/redcap/redcap_full_108_cohort_final.csv"
)
CSV_FILE_CBTN = Path("/home/juanqui55/git/mri-longitudinal-analysis/data/redcap")

# Output directory
OUTPUT_DIR = Path("/home/juanqui55/git/mri-longitudinal-analysis/data/output/clinical_data")

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
REMOVING_SURGERY = False
DATA_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_45_surgery_cohort_reviewed/1_no_comments"
)


BCH_mapping = {
    "patient_id": "BCH MRN",
    "dob": "Date of Birth",
    "clinical_status": "Clinical status at last follow-up",
    "pathologic_diagnosis": "Pathologic diagnosis",
    "sex": "Sex",
    "BRAF_V600E": "BRAF V600E mutation",
    "BRAF fusion": "BRAF fusion",
    "Diagnosis": "Pathologic diagnosis",
    "Surgery": "Surgical Resection",
    "Surgery Date": "Date of first surgery",
    "Chemotherapy": "Systemic therapy before radiation",
    "Chemotherapy Date": "Date of Systemic Therapy Start",
    "Radiation": "Radiation as part of initial treatment",
    "Radiation Date": "Start Date of Radiation",
}
