"""Config file for the script filter_clinical_data.py"""

from pathlib import Path

SEG_DIR = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/"
    "curated_no_ops_60_cohort_reviewed/output/seg_predictions"
)


OUTPUT_DIR = Path(
    "/mnt/kannlab_rfa/JuanCarlos/mri-classification-sequences/"
    "curated_no_ops_60_cohort_reviewed/output"
)

PLOTS_DIR = OUTPUT_DIR / "volume_plots"
CSV_DIR = OUTPUT_DIR / "time_series_csv"

REDCAP_FILE = Path(
    "/home/jc053/GIT/mri-longitudinal-analysis/data/redcap/redcap_full_89_cohort.csv"
)
LIMIT_LOADING = None

POLY_SMOOTHING = False

TEST_DATA = False
