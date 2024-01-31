"""Config file for the script arima.py"""

from pathlib import Path

TIME_SERIES_DIR = Path("/home/juanqui55/git/mri-longitudinal-analysis/data/input/time_series")
COHORT = "CBTN"

OUTPUT_DIR = Path(
    f"/home/juanqui55/git/mri-longitudinal-analysis/data/output/arima_plots_{COHORT.lower()}"
)


PLOTTING = True  # For AC and PAC plots at the begging of each patient
DIAGNOSTICS = False
LOADING_LIMIT = 1
