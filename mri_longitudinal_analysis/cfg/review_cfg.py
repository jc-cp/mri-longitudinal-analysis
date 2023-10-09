"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

####################################################################################################

# overall path(s) for filtered no_ops T2 data
# DATA_FOLDER_60 = Path(
#     "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_60_cohort_filtered"
# )
# DATA_FOLDER_60_T2 = DATA_FOLDER_60 / "T2"
# CSV_FILE_60 = DATA_FOLDER_60_T2 / "annotations.csv"

# # overall path(s) for filtered surgery T2 data
# DATA_FOLDER_29 = Path(
#     "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/"
#     "curated_no_ops_29_surgery_cohort_filtered"
# )
# DATA_FOLDER_29_T2 = DATA_FOLDER_29 / "T2"
# CSV_FILE_29 = DATA_FOLDER_29_T2 / "annotations.csv"

# overall path(s) for filtered surgery T2 data
DATA_FOLDER_18 = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_19_surgery_cohort_filtered"
)
DATA_FOLDER_18_T2 = DATA_FOLDER_18 / "T2"
CSV_FILE_18 = DATA_FOLDER_18_T2 / "annotations.csv"
####################################################################################################

# condition flag to copy files once reviewed
MOVING_2_REVIEW = False

# once initial review is donde, move classified files here
# REVIEWED_FOLDER_60 = Path(
#     "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_60_cohort_reviewed/"
# )
# DIR1_NO_COMMENTS_60 = REVIEWED_FOLDER_60 / "1_no_comments"
# DIR1_WITH_COMMENTS_60 = REVIEWED_FOLDER_60 / "1_with_comments"
# DIR5_60 = REVIEWED_FOLDER_60 / "5"

# # once initial review is donde, move classified files here
# REVIEWED_FOLDER_29 = Path(
#     "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/"
#     "curated_no_ops_29_surgery_cohort_reviewed/"
# )
# DIR1_NO_COMMENTS_29 = REVIEWED_FOLDER_29 / "1_no_comments"
# DIR1_WITH_COMMENTS_29 = REVIEWED_FOLDER_29 / "1_with_comments"
# DIR5_29 = REVIEWED_FOLDER_29 / "5"

# once initial review is donde, move classified files here
REVIEWED_FOLDER_18 = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/"
    "curated_no_ops_18_surgery_cohort_filtered/T2"
)
DIR1_NO_COMMENTS_18 = REVIEWED_FOLDER_18 / "1_no_comments"
DIR1_WITH_COMMENTS_18 = REVIEWED_FOLDER_18 / "1_with_comments"
DIR5_18 = REVIEWED_FOLDER_18 / "5"

####################################################################################################

# scanning of other folders for images containing a t2 in their name
DETECTING = True
# OUTPUT_DETECTTION_60 = DATA_FOLDER_60 / "Detected"
# OUTPUT_DETECTTION_29 = DATA_FOLDER_29 / "Detected"
OUTPUT_DETECTTION_19 = DATA_FOLDER_18 / "Detected"

####################################################################################################

# condition flag to copy files once completely reviewed by radiologist
MOVING_4_DATASET = False

####################################################################################################

# condition flag to remove patients with massive artifacts
DELETE_ARTIFACTS = False
PATIENTS_WITH_ARTIFACTS = ["4073188", "4303399", "4394032", "1194890"]

####################################################################################################

# condition flag to compare IDs and select images with the best resolution
COMPARE_IDS = False

####################################################################################################
