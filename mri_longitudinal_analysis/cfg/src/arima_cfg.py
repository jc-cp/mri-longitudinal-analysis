"""Config file for the script arima.py"""

from pathlib import Path

COHORT = "BCH"  # "BCH" or "CBTN"
TIME_SERIES_DIR_CBTN = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/accepted/pre_treatment/output/time_series/moving_average")
TIME_SERIES_DIR_BCH = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/bch_longitudinal_dataset/final/pre_treatment/output/time_series/moving_average")
TIME_SERIES_DIR_COHORT = TIME_SERIES_DIR_CBTN if COHORT == "CBTN" else TIME_SERIES_DIR_BCH

OUTPUT_DIR = Path(
    f"/home/jc053/GIT/mri_longitudinal_analysis/data/output/arima_plots_{COHORT.lower()}"
)


PLOTTING = True  # For AC and PAC plots at the begging of each patient
DIAGNOSTICS = False
LOADING_LIMIT = 56 # BCH: 56, CBTN: 43
INTERPOLATION_FREQ = 7
