"""Config file for the script arima.py"""

from pathlib import Path

COHORT = "JOINT"  #  or "BCH" or "CBTN"
TIME_SERIES_DIR_CBTN = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/accepted/pre_treatment/output/time_series/moving_average")
TIME_SERIES_DIR_BCH = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/bch_longitudinal_dataset/final/pre_treatment/output/time_series/moving_average")
TIME_SERIES_DIR_JOINT = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/final_dataset/output/time_series/moving_average")
TIME_SERIES_DIR_COHORT = TIME_SERIES_DIR_JOINT if COHORT == "JOINT" else TIME_SERIES_DIR_CBTN if COHORT == "CBTN" else TIME_SERIES_DIR_BCH

OUTPUT_DIR = Path(
    f"/home/juanqui55/git/mri-longitudinal-analysis/data/output/05_volumetric_forecasting/{COHORT.lower()}"
)


PLOTTING = True  # For AC and PAC plots at the begging of each patient
DIAGNOSTICS = False
LOADING_LIMIT = 99 if COHORT =='JOINT' else 56 if COHORT =='BCH' else 43
INTERPOLATION_FREQ = 7
