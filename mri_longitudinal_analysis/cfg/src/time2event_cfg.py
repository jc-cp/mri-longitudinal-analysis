"""
Time to event configuration file.
"""
from pathlib import Path
from cfg.src.lr_and_correlations_cfg import LR_VARS

COHORT = "JOINT"    # DF_BCH or CBTN or JOINT
COHORT_DATAFRAME = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/02_trajectories/JOINT_cohort_data_features_traj.csv_cohort_data_features.csv")
OUTPUT_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/04_time_to_event")


EVENT_COL = "Event_Occurred"
DURATION_COL = "Duration"

STRATIFICATION_VARS = [
                "Location",
                "Sex",
                "BRAF Status",
                "Age Group at Diagnosis",
                "Time Period Since Diagnosis",
                "Symptoms",
                "Received Treatment",
                None,
            ]


BASELINE_VARS = LR_VARS