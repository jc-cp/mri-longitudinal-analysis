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
REVIEW_OLD_MASKS_FOLDER = REVIEWS_FOLDER / "old_masks"
REVIEW_NEW_MASKS_FOLDER = REVIEWS_FOLDER / "new_masks"
ANNOTATIONS_CSV = TARGET_FOLDER / "annotations.csv"

PART_1 = False  # matches images and masks and moves them to the target folder for a joint review
PART_2 = False  # checks reviewed files according to the annotations.csv file and moves them to corresponding folders
MOVING_FILES = False
PART_3 = False  # renaming files for second segmentation run by appending a suffix to the file name
# run segmentation again for new masks
PART_4 = False  # analzing the new generated masks and marking them as acceptable or not
PART_5 = True  # renaming files with a hash to original file names
