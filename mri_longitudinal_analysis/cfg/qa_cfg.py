from pathlib import Path

IMAGES_FOLDER_1 = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_45_surgery_cohort_reviewed/output/T2W_brain_extraction"
)
IMAGES_FOLDER_2 = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_63_cohort_reviewed/output/T2W_brain_extraction"
)
MASKS_FOLDER_1 = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_45_surgery_cohort_reviewed/output/seg_predictions"
)
MASKS_FOLDER_2 = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_no_ops_63_cohort_reviewed/output/seg_predictions"
)

# Target directory for the consolidated files
TARGET_FOLDER = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/curated_longitudinal_dataset_new"
)
REJECTED_FOLDER = TARGET_FOLDER / "rejected"
REJECTED_MASKS_FOLDER = REJECTED_FOLDER / "old_masks"
ACCEPTED_FOLDER = TARGET_FOLDER / "accepted"
REVIEWS_FOLDER = TARGET_FOLDER / "reviews"
REVIEW_IMAGES_FOLDER = REVIEWS_FOLDER / "images"
REVIEW_IMAGES_EDITED_FOLDER = REVIEWS_FOLDER / "images_without_0000"
REVIEW_OLD_MASKS_FOLDER = REVIEWS_FOLDER / "old_masks"
REVIEW_NEW_MASKS_FOLDER = REVIEWS_FOLDER / "new_masks"
ANNOTATIONS_CSV_MASKS = REVIEW_NEW_MASKS_FOLDER / "annotations.csv"
ANNOTATIONS_CSV = TARGET_FOLDER / "annotations.csv"

# matches images and masks and moves them to the target folder for a joint review
PART_1 = False

# checks reviewed files according to the annotations.csv file and moves them to corresponding folders
PART_2 = False
MOVING_FILES = False

# renaming files for second segmentation run by appending a suffix to the file name
# after ART_3 run segmentation algorithm for new masks
PART_3 = False

# analzing the new generated masks and marking them as acceptable or not
PART_4 = False

# renaming files with a hash to original file names
PART_5 = False

# moving files to the final folder structure
PART_6 = False
VOXEL_COUNT = 10
