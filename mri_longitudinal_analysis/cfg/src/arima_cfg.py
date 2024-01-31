"""Config file for the script arima.py"""

from pathlib import Path

TIME_SERIES_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/output/time_series_csv_kernel_smoothed"
)
COHORT = "CBTN"

OUTPUT_DIR = Path(
    f"/home/jc053/GIT/mri_longitudinal_analysis/data/output/arima_plots_{COHORT.lower()}"
)


PLOTTING = True  # For AC and PAC plots at the begging of each patient
DIAGNOSTICS = False
LOADING_LIMIT = 1
