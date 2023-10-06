"""Config file for the script evaluation_t2w.py"""

from pathlib import Path

DATA_FOLDER_60 = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/60_no_ops_cohort/T2/")
CSV_FILE_60 = DATA_FOLDER_60 / "annotations.csv"
DATA_FOLDER_29 = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/29_no_ops_surgery_cohort/T2/")
CSV_FILE_29 = DATA_FOLDER_29 / "annotations.csv"
DATA_FOLDER_19 = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/19_new_mixed_cohort/T2/")
CSV_FILE_19 = DATA_FOLDER_19 / "annotations.csv"

# output images
OUTPUT_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/plots/")
OUT_FILE_PREFIX = OUTPUT_DIR / "output_evaluation_t2w"
