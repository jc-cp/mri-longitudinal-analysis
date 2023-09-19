from pathlib import Path

# General path of the repo
PROJ_DIR = Path("/home/jc053/GIT/mri-longitudinal-segmentation")

# For remote data on the drive
DATA_DIR = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_60_cohort_reviewed"
)
INPUT_DIR = DATA_DIR / "1_no_comments"
OUPUT_DIR = DATA_DIR / "output"

# For local data on the repo
# INPUT_DIR = PROJ_DIR / "data/test_data"
# INPUT_DIR = PROJ_DIR / "data/T2"
# OUPUT_DIR = PROJ_DIR / "data/output"

REG_DIR = OUPUT_DIR / "T2W_registration"
BF_CORRECTION_DIR = OUPUT_DIR / "T2W_bf_correction"
BRAIN_EXTRACTION_DIR = OUPUT_DIR / "T2W_brain_extraction"
SEG_PRED_DIR = OUPUT_DIR / "seg_predictions"

TEMP_DIR = PROJ_DIR / "temp_dir"
TEMP_IMG = TEMP_DIR / "temp_head.nii.gz"

REGISTRATION = True
EXTRACTION = True
BF_CORRECTION = (
    False  # beware,there is a predefined BF in the registration process already
)


LIMIT_LOADING = 1000
