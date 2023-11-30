"""Config file for the script clustering.py"""

from pathlib import Path

INPUT_PATH = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_stats/pre-treatment_dl_features.csv"
)

OUTPUT_PATH = Path("/home/jc053/GIT/mri-longitudinal-analysis/data/output/")
PLOT_OUTPUT_PATH = OUTPUT_PATH / "clustering_plots"
METRICS_OUTPUT_PATH = OUTPUT_PATH / "metrics.txt"
