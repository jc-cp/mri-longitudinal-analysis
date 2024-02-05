"""Config file for the script clustering.py"""

from pathlib import Path

# Input and output paths
INPUT_PATH = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/cbtn_longitudinal_dataset/pre_event/output/time_series_csv_kernel_smoothed"
)


OUTPUT_PATH = Path("/home/jc053/GIT/mri_longitudinal_analysis/data/output/")
PLOTS_OUTPUT_PATH = OUTPUT_PATH / "clustering_plots"
METRICS_OUTPUT_PATH = OUTPUT_PATH / "metrics.txt"

# Variables
SELECTED_FEATURES = ["Age", "Normalized Volume", "Volume Growth[%]", 
                     "Volume Growth[%] Rate", "Volume Growth[%] Avg",
                     "Volume Growth[%] Std"]
COHORT = "CBTN"
LIMIT_LOADING = 5

# Variables for the dimensionality reduction
USE_UMAP = True
USE_TSNE = True

# Attributes to be used for clustering
N_CLUSTERS = 3
KMEANS_VERBOSE = True
KMEANS_METRIC = "dtw"

# Plotting
PLOT_PATIENT_DATA_EDA = True
PLOT_PATIENT_DATA_SCALING = True
PLOT_HEATMAP = True
