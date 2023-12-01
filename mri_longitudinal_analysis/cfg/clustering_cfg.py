"""Config file for the script clustering.py"""

from pathlib import Path

INPUT_PATH = Path(
    "/home/jc053/GIT/mri_longitudinal_analysis/data/output/correlation_stats/pre-treatment_dl_features.csv"
)

OUTPUT_PATH = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/")
PLOTS_OUTPUT_PATH = OUTPUT_PATH / "clustering_plots"
METRICS_OUTPUT_PATH = OUTPUT_PATH / "metrics.txt"

# Variables for the dimensionality reduction
USE_UMAP = False
USE_TSNE = False

# Attributes to be used for clustering
N_CLUSTERS = 3
KMEANS_VERBOSE = True
KMEANS_METRIC = "dtw"


# Plotting
PLOT_PATIENT_DATA = True
PLOT_HEATMAP = True
