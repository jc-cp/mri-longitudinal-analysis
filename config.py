from pathlib import Path

PROJ_DIR = Path("/home/jc053/GIT/mri-longitudinal-segmentation")
INPUT_DIR = PROJ_DIR / "data/test_data"
OUPUT_DIR = PROJ_DIR / "data/output"

REG_DIR = OUPUT_DIR / "T2W_reg"
BRAIN_DIR = OUPUT_DIR / "nnunet/imagesTs"
CORRECTION_DIR = OUPUT_DIR / "T2W_correction"
PRO_DATA_DIR = OUPUT_DIR / "pro_data"
NNUNET_OUTPUT_DIR = OUPUT_DIR / "nnunet"

TEMP_DIR = PROJ_DIR / "temp_dir"
TEMP_IMG = TEMP_DIR / "temp_head.nii.gz"

REGISTRATION = True
EXTRACTION = False
BF_CORRECTION = False