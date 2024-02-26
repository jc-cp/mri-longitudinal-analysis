"""
Config file for the qa process.
"""
from pathlib import Path


##########
# PART 1 #
##########
# matches images and masks and moves them to the target folder for a joint review / qa from the clinician
PART_1 = False
IMAGES_FOLDER = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/bch_longitudinal_dataset/new_review/after_review_before_pp/output/T2W_brain_extraction"
)

MASKS_FOLDER = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/bch_longitudinal_dataset/new_review/after_review_before_pp/output/seg_predictions"
)

TARGET_FOLDER = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/qa_second_time/qa"
)

SEG_REVIEWS_FOLDER = Path("/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/second_review")

##########
# PART 2 #
##########
# checks reviewed files according to the annotations.csv file and moves them to corresponding folders
PART_2 = False
ANNOTATIONS_CSV = TARGET_FOLDER / "annotations_jc.csv"
MOVING_FILES = False

REJECTED_FOLDER = TARGET_FOLDER / "rejected"
REJECTED_MASKS_FOLDER = REJECTED_FOLDER / "old_masks"
ACCEPTED_FOLDER = TARGET_FOLDER / "accepted"
REVIEWS_FOLDER = TARGET_FOLDER / "reviews"
REVIEW_IMAGES_FOLDER = REVIEWS_FOLDER / "images"
REVIEW_OLD_MASKS_FOLDER = REVIEWS_FOLDER / "old_masks"
REVIEW_NEW_MASKS_FOLDER = REVIEWS_FOLDER / "new_masks"
REVIEW_MASKS_EDITED_FOLDER = REVIEWS_FOLDER / "masks_edited"

##########
# PART 3 #
##########
# renaming files for second segmentation run by appending a suffix to the file name
# after PART_3 run segmentation algorithm for new masks
PART_3 = False

##########
# PART 4 #
##########
# analzing the new generated masks and marking them as acceptable or not
PART_4 = False
ANNOTATIONS_CSV_MASKS = REVIEW_NEW_MASKS_FOLDER / "annotations.csv"

##########
# PART 5 #
##########
# renaming files with a hash to original file names
PART_5 = False

##########
# PART 6 #
##########
# moving files to the final folder structure,, images should have the _0000 suffix removed
PART_6 = False
SECOND_REVIEW = TARGET_FOLDER / "second_review"
VOXEL_COUNT = 10


##########
# PART 7 #
##########
# moving files to the final folder structure after segemntation qa
PART_7 = False
ANNOTATIONS_FINAL = SEG_REVIEWS_FOLDER / "annotations_jc.csv"
FINAL_MOVE = False
