from pathlib import Path

PROJ_DIR = Path("/home/jc053/GIT/mri-longitudinal-segmentation")
# INPUT_DIR = PROJ_DIR / "data/test_data"
INPUT_DIR = PROJ_DIR / "data/T2"
OUPUT_DIR = PROJ_DIR / "data/output"

REG_DIR = OUPUT_DIR / "T2W_reg"
BF_CORRECTION_DIR = OUPUT_DIR / "T2W_bf_correction"
BRAIN_EXTRACTION_DIR = OUPUT_DIR / "T2W_brain_extraction"
SEG_PRED_DIR = OUPUT_DIR / "seg_predictions"

TEMP_DIR = PROJ_DIR / "temp_dir"
TEMP_IMG = TEMP_DIR / "temp_head.nii.gz"

REGISTRATION = True
EXTRACTION = False
BF_CORRECTION = False
