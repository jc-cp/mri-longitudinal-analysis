"""Config file for the script arima.py"""

from pathlib import Path

TIME_SERIES_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/output/time_series_csv_kernel_smoothed"
)

OUTPUT_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/arima_plots")

COHORT = "CBTN"

PLOTTING = True  # For AC and PAC plots at the begging of each patient
DIAGNOSTICS = False
LOADING_LIMIT = 1
