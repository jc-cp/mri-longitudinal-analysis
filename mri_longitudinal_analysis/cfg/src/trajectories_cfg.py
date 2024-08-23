"""Config file for the trajectories analysis script."""
from pathlib import Path
COHORT = "JOINT"    # DF_BCH or CBTN or JOINT
SAMPLE_SIZE = 99    # DF_BCH: 56, CBTN: 43, JOINT: 99
COHORT_DATAFRAME = Path(f"/home/jc053/GIT/mri_longitudinal_analysis/data/output/01_cohort_data/{COHORT.lower()}_cohort_data_features.csv")

OUTPUT_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/02_trajectories")
CURVE_VARS = [
            "Sex",
            "BRAF Status",
            "Received Treatment",
            "Location",
            "Treatment Type",
            "Symptoms",
            #"Histology",
        ]

PROGRESSION_THRESHOLD = 1.25    # +25% volume change threshold that defines progression on normalized volume
REGRESSION_THRESHOLD = 0.75     # -25% volume change threshold that defines regression on normalized volume
CHANGE_THRESHOLD = 1.10         # +10% volume change threshold for stability index / time gap

TIME_PERIOD_MAPPING  = {
        "0-1 years": 1,
        "1-3 years": 3,
        "3-5 years": 5,
        "5+ years": 10,
        #"10+ years": 20
    }

AGE_GROUPS = ["Infant", "Preschool", "School Age", "Adolescent"]