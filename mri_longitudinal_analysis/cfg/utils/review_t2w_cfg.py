"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

####################################################################################################

# overall path(s) for reviewed T2 data
DATA_FOLDER = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event"
)
CSV_FILE = DATA_FOLDER / "annotations.csv"
####################################################################################################

# condition flag to move files once reviewed
MOVING_AFTER_REVIEW = True
DIR1 = DATA_FOLDER / "1"
DIR2 = DATA_FOLDER / "2"
DIR5 = DATA_FOLDER / "5"

####################################################################################################

# scanning of other folders for images containing a t2 in their name
DETECTING = False
OUTPUT_DETECTTION = DATA_FOLDER / "Detected"

####################################################################################################

# condition flag to copy files once completely reviewed by radiologist
MOVING_4_DATASET = False

####################################################################################################

# condition flag to remove patients with massive artifacts
DELETE_ARTIFACTS = False
PATIENTS_WITH_ARTIFACTS = [
    # "4073188",
    # "4303399",
    # "4394032",
    # "1194890",
    # "2126809",
    # "5127658",
    # "4864792",
]

####################################################################################################

# condition flag to compare IDs and select images with the best resolution
COMPARE_IDS = False

####################################################################################################
