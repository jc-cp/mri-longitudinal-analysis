from pathlib import Path


PATH_CSV = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/redcap/cbtn_536_filtered.csv")

PATH_METADATA = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/redcap/flywheel_file_metadata.csv"
)

PATH_IMAGES = Path("/mnt/an225b/Anna/fw/fw/data")

NEW_PATH_IMAGES = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset"
)

ASSERTATIONS = True
LENGTH_DATA = 536  # do not count header


OUTPUT_CSV_PRE_EVENT = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/output/clinical_data/cbtn_pre_event.csv"
)

OUTPUT_CSV_POST_EVENT = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/output/clinical_data/cbtn_post_event.csv"
)
