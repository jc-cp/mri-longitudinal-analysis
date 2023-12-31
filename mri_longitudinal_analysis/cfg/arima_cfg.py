"""Config file for the script arima.py"""

from pathlib import Path

PLOTS_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/test_data/output/volume_plots")
TIME_SERIES_DIR = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/test_data/output/time_series_csv"
)

OUTPUT_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/test_data/output/arima_plots")

LOADING_LIMIT = 5

FROM_IMAGES = False
FROM_DATA = True
