from pathlib import Path

SEG_DIR = Path(
    "/home/jc053/GIT/mri-longitudinal-segmentation/data/test_data/output/segmentation_predictions"
)


OUTPUT_DIR = Path("/home/jc053/GIT/mri-longitudinal-segmentation/data/test_data/output")

PLOTS_DIR = OUTPUT_DIR / "volume_plots"
CSV_DIR = OUTPUT_DIR / "time_series_csv"

REDCAP_FILE = Path(
    "/home/jc053/GIT/mri-longitudinal-segmentation/data/redcap/redcap_full_89_cohort.csv"
)
LIMIT_LOADING = 4

POLY_SMOOTHING = False

TEST_DATA = True