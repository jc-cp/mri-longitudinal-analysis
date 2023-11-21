"""Config file for the correlation analysis script."""
from pathlib import Path

CLINICAL_CSV = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/redcap/redcap_full_108_cohort.csv"
)

VOLUMES_CSVs_45 = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_45_surgery_cohort_reviewed/output/time_series_csv_kernel_smoothed"
)
VOLUMES_CSVs_63 = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_no_ops_63_cohort_reviewed/output/time_series_csv_kernel_smoothed"
)

OUTPUT_DIR_CORRELATIONS = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_plots"
)
OUTPUT_DIR_STATS = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_stats")

CORRELATION_PRE_TREATMENT = "spearman"
CORRELATION_POST_TREATMENT = "spearman"

# Step 0
SEPARATION = True

# Step 1
SENSITIVITY = True
SENSITIVITY_THRESHOLD = 2

# Step 2
PROPENSITY = True

# Step 3
ANLYSIS = True
PROGRESSION_THRESHOLD = 5  # angle value that defines progression
HIGH_RISK_THRESHOLD = 25  # angle value that defines high increase
STABILITY_THRESHOLD = 0.5  # angle value that defines stability
SAMPLE_SIZE = 300  # for plotting growth trajectories
UNCHANGING_THRESHOLD = 0.1  # for trakcing stable volumes

# Step 4
CORRECTION = True
CORRECTION_ALPHA = 0.05

# Step 5
FEATURE_ENG = False
