from pathlib import Path

# overall path for reviewed no_ops T2 data
DATA_FOLDER_60 = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_60_cohort_filtered/T2"
)
CSV_FILE_60 = DATA_FOLDER_60 / "annotations.csv"
DATA_FOLDER_29 = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_29_surgery_cohort_filtered/T2"
)
CSV_FILE_29 = DATA_FOLDER_29 / "annotations.csv"

# output images
OUTPUT_DIR = Path("/home/jc053/GIT/mri-longitudinal-segmentation/data/output")
OUT_FILE_PREFIX = OUTPUT_DIR / "output_evaluation_t2w"
