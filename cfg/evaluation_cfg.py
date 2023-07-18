from pathlib import Path

# overall path for reviewed no_ops T2 data
DATA_FOLDER = Path("/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_60_cohort_filtered/T2")
CSV_FILE = DATA_FOLDER / "annotations.csv"

# once reviewed move classified files here
REVIEWED_FOLDER = Path("/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_60_cohort_reviewed/")
DIR1_NO_COMMENTS = REVIEWED_FOLDER / "1_no_comments"
DIR1_WITH_COMMENTS = REVIEWED_FOLDER / "1_with_comments"
DIR5 = REVIEWED_FOLDER / "5"

# output images
OUTPUT_DIR = Path(
    "/home/jc053/GIT/mri-longitudinal-segmentation/data/output"
)
OUT_FILE = OUTPUT_DIR / "output_evaluation_t2w.png"

# condition flag to copy files once evaluated
MOVING = True
