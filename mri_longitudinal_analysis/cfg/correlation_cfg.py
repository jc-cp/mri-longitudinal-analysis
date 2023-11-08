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


SENSITIVITY = True

PROPENSITY = True

ANLYSIS = True

CORRECTION = True
CORRECTION_ALPHA = 0.05

FEATURE_ENG = False
