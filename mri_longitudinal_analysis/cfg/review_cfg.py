"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

####################################################################################################

# overall path(s) for filtered surgery T2 data
DATA_FOLDER = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_2_ops")
DATA_FOLDER_T2 = DATA_FOLDER / "T2"
CSV_FILE = DATA_FOLDER_T2 / "annotations.csv"
####################################################################################################

# condition flag to copy files once reviewed
MOVING_2_REVIEW = True

# once initial review is donde, move classified files here
REVIEWED_FOLDER = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_2_ops_reviewed"
)
DIR1_NO_COMMENTS = REVIEWED_FOLDER / "1_no_comments"
DIR1_WITH_COMMENTS = REVIEWED_FOLDER / "1_with_comments"
DIR5 = REVIEWED_FOLDER / "5"

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
    "4073188",
    "4303399",
    "4394032",
    "1194890",
    # "2126809",
    # "5127658",
    # "4864792",
]

####################################################################################################

# condition flag to compare IDs and select images with the best resolution
COMPARE_IDS = False

####################################################################################################
