"""Config file for the script mri_preprocess_3d.py"""

from pathlib import Path

# General path of the repo
PROJ_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/mri_longitudinal_analysis")

# For remote data on the drive
DATA_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/bch_longitudinal_dataset/new_review"
)

INPUT_DIR = DATA_DIR / "pp_retry"
OUPUT_DIR = INPUT_DIR / "output"

REG_DIR = OUPUT_DIR / "T2W_registration"
BF_CORRECTION_DIR = OUPUT_DIR / "T2W_bf_correction"
BRAIN_EXTRACTION_DIR = OUPUT_DIR / "T2W_brain_extraction"
SEG_PRED_DIR = OUPUT_DIR / "seg_predictions"

TEMP_DIR = PROJ_DIR / "templates"
TEMP_IMG = TEMP_DIR / "temp_head.nii.gz"

REGISTRATION = True
EXTRACTION = True
BF_CORRECTION = False  # beware,there is a predefined BF in the registration process already


LIMIT_LOADING = 2000
