"""
Configuration file for parsing the pLGG CBTN dataset.
"""
from pathlib import Path


CSV_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/redcap")
PATH_CLINICAL_CSV = CSV_DIR / "cbtn_filtered_and_pruned_513.csv"
PATH_METADATA_CSV = CSV_DIR / "flywheel_file_metadata.csv"


PATH_IMAGES = Path("/mnt/an225b/Anna/fw/fw/data")
NEW_PATH_IMAGES = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset"
)
NEW_PATH_IMAGES_PRE = NEW_PATH_IMAGES / "pre_event" / "1_no_comments"
NEW_PATH_IMAGES_POST = NEW_PATH_IMAGES / "post_event" / "1_no_comments"

ASSERTATIONS = True
LENGTH_DATA = 513  # do not count header


OUTPUT_PATH = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/clinical_data")
OUTPUT_CSV_PRE_EVENT = OUTPUT_PATH / "cbtn_pre_event.csv"
OUTPUT_CSV_POST_EVENT = OUTPUT_PATH / "cbtn_post_event.csv"

PARSING = True
MOVING = False
