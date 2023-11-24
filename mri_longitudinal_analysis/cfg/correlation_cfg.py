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
SENSITIVITY = False
SENSITIVITY_THRESHOLD = 2

# Step 2
PROPENSITY = False
CALIPER = 0.1  # usually between 0.05 and 0.2

# Step 3
ANALYSIS_PRE_TREATMENT = True

PROGRESSION_THRESHOLD = 10  # angle value that defines progression
HIGH_RISK_THRESHOLD = 25  # angle value that defines high increase
STABILITY_THRESHOLD = 2  # angle value that defines stability

SAMPLE_SIZE = 300  # for plotting growth trajectories

END_POINTS = False

VOLUME_WEIGHT = 0.25
GROWTH_WEIGHT = 0.75
CHANGE_THRESHOLD = 25  # % volume change threshold for stability index

ANALYSIS_POST_TREATMENT = False


# Step 4
CORRECTION = False
CORRECTION_ALPHA = 0.05

# Step 5
FEATURE_ENG = False
