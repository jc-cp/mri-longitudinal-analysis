from pathlib import Path

# overall path for reviewed no_ops T2 data
DATA_FOLDER = Path(
    "/home/jc053/GIT/mri-longitudinal-segmentation/data/60_no_ops_cohort/T2"
)

CSV_FILE = DATA_FOLDER / "annotations.csv"
DIR1_NO_COMMENTS = DATA_FOLDER / "1_no_comments"
DIR1_WITH_COMMENTS = DATA_FOLDER / "1_with_comments"
DIR5 = DATA_FOLDER / "5"

OUTPUT_DIR = Path(
    "/home/jc053/GIT/mri-longitudinal-segmentation/data/output"
)
OUT_FILE = OUTPUT_DIR / "output_evaluation_t2w.png"

MOVING = False
