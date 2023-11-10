"""Config file for the correlation analysis script."""
from pathlib import Path

CLINICAL_CSV = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/redcap/redcap_full_108_cohort.csv"
)

VOLUMES_CSVs_45 = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_45_surgery_cohort_reviewed/output/time_series_csv"
)
VOLUMES_CSVs_63 = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_63_cohort_reviewed/output/time_series_csv"
)

OUTPUT_DIR = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation")

CORRELATION_PRE_TREATMENT = "spearman"
CORRELATION_POST_TREATMENT = "spearman"


# Step 1
SENSITIVITY = True
SENSITIVITY_THRESHOLD = 1.5

# Step 2
PROPENSITY = True

# Step 3
ANLYSIS = True

SAMPLE_SIZE = 10  # for plotting growth trajectories
UNCHANGING_THRESHOLD = 0.05  # for trakcing stable volumes

# Step 4
CORRECTION = True
CORRECTION_ALPHA = 0.05

# Step 5
TRENDS = False

# Step 6
FEATURE_ENG = False
