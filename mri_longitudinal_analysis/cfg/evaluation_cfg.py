from pathlib import Path

# overall path for reviewed no_ops T2 data
DATA_FOLDER_60 = Path(
    "/home/jc053/GIT/mri-longitudinal-analysis/data/60_no_ops_cohort/T2/"
)
CSV_FILE_60 = DATA_FOLDER_60 / "annotations.csv"
DATA_FOLDER_29 = Path(
    "/home/jc053/GIT/mri-longitudinal-analysis/data/29_no_ops_surgery_cohort/T2/"
)
CSV_FILE_29 = DATA_FOLDER_29 / "annotations.csv"

# output images
OUTPUT_DIR = Path("/home/jc053/GIT/mri-longitudinal-analysis/data/output")
OUT_FILE_PREFIX = OUTPUT_DIR / "output_evaluation_t2w"
