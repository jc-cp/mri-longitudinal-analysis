"""
Cluster Analysis Script
========================

This script is dedicated to performing cluster analysis on a given dataset.
The primary class, ClusterAnalysis, supports various clustering methods including K-means,
DBSCAN, Faiss clustering, and a custom deep clustering method, and provides utility functions
to visualize the results through UMAP and t-SNE projections, heatmaps, and silhouette scores.

Usage:
    To use this script, initialize the ClusterAnalysis class with appropriate parameters
    and call the desired methods to perform clustering and generate visualizations and metrics.

Functions and Methods:
    - standardize_data: Standardizes the numerical data.
    - apply_umap: Applies UMAP to reduce dimensions of the data.
    - apply_tsne: Applies t-SNE to reduce dimensions of the data.
    - perform_kmeans_clustering: Applies K-means clustering on the data.
    - perform_faiss_clustering: Applies Faiss clustering on the data.
    - deep_clustering: Applies deep clustering using a simple neural network.
    - evaluate_silhouette_score: Evaluates and stores the silhouette score for a clustering method.
    - save_metrics: Saves the silhouette scores to a file.
    - plot_dimensionality_reduction: Generates UMAP and t-SNE scatter plots of the clusters.
    - perform_dbscan_clustering: Applies DBSCAN clustering on the data.
    - add_cluster_ellipses: Adds ellipses around clusters in scatter plots.
    - plot_clusters_with_boundaries: Generates UMAP and t-SNE scatter plots with ellipses.
    - plot_heatmap: Generates a heatmap of the data sorted by cluster labels.

Attributes:
    - The script assumes the input DataFrame contains a patient_id_column and feature columns.
    - The paths for saving plots and metrics are specified through 
    output_path and metrics_output_path respectively.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib import colormaps
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from cfg.src import clustering_cfg


class ClusterAnalysis:
    """
    A class used to perform and visualize different clustering methods on the given data.

    Attributes
    ----------
    data : pd.DataFrame
        The input data.
    patient_id_column : str
        The name of the column in the dataframe representing patient IDs.
    output_path : str
        The path where the output plots will be saved.
    metrics_output_path : str
        The path where the silhouette scores will be saved.
    data : np.array
        Numerical data to be used for clustering.
    umap_embedding : np.array
        2D UMAP embeddings of the data.
    tsne_embedding : np.array
        2D t-SNE embeddings of the data.
    cluster_labels : np.array
        Labels assigned to each data point by clustering algorithms.
    silhouette_scores : dict
        Dictionary to store silhouette scores for different clustering methods.
    """

    def __init__(self):
        # Define and initialize attributes
        # DR
        self.umap_embedding = None
        self.tsne_embedding = None
        self.cluster_labels = None
        # Clustering
        self.n_clusters = clustering_cfg.N_CLUSTERS
        self.kmeans_metric = clustering_cfg.KMEANS_METRIC
        self.kmeans_verbose = clustering_cfg.KMEANS_VERBOSE
        self.ts_kmeans_model = None
        self.ts_dbscan_model = None
        self.silhouette_scores = {}
        # Output
        self.output_path = clustering_cfg.PLOTS_OUTPUT_PATH
        os.makedirs(self.output_path, exist_ok=True)
        self.metrics_output_path = clustering_cfg.METRICS_OUTPUT_PATH

    #####################
    # UTILITY FUNCTIONS #
    #####################
    def load_data(self, path, selected_features, limit):
        """
        Loads the data from a directory with CSV files and returns a list of DataFrames.
        """
        dfs = []
        files_processed = 0
        for file in os.listdir(path):
            if file.endswith(".csv"):
                if limit is not None and files_processed>=limit:
                    break
                patient_id = file.split(".")[0]
                file_path = os.path.join(path, file)
                df = pd.read_csv(file_path)
                selected_features = [feature for feature in selected_features if feature in df.columns]
                df_selected = df[selected_features]
                df_selected = df_selected.fillna(0.0)
                df_selected = df_selected.copy()
                df_selected.loc[:, "Patient_ID"] = patient_id
                dfs.append(df_selected)
                files_processed += 1
        return dfs

    def standardize_data(self, dfs):
        """
        Standardizes numerical data in the dataframe while preserving non-numeric columns.
        """
        scaled_dfs = []
        for df in dfs:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame.")
           
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()        
            if numeric_columns:
                numeric_df  = df[numeric_columns]
                numeric_data_3d = numeric_df.to_numpy().reshape(len(numeric_df), -1, 1)
                scaler = TimeSeriesScalerMeanVariance()
                scaled_data_3d = scaler.fit_transform(numeric_data_3d)
                scaled_data_2d = scaled_data_3d.reshape(scaled_data_3d.shape[0], scaled_data_3d.shape[1])
                scaled_numeric_df = pd.DataFrame(scaled_data_2d, columns=numeric_columns, index=numeric_df.index)
                for column in numeric_columns:
                    df.loc[:, column] = scaled_numeric_df[column]
        
            scaled_dfs.append(df)

        return dfs

    def align_and_interpolate_time_series(self, dfs, time_var="Age", features=None):
        """
        Aligns, interpolates, and ensures uniform length for the list of DataFrames, focusing on selected time-dependent features.
        
        Parameters:
        - dfs: List of pandas DataFrames, each representing a patient's time series.
        - time_var: The column name representing the time variable.
        - features: List of feature columns to be included in the clustering.
        
        Returns:
        - A 3D numpy array suitable for TimeSeriesKMeans with DTW metric.
        """
        # Determine the maximum length across all series
        max_length = max(df.shape[0] for df in dfs)
        aligned_dfs = []

        for df in dfs:
            df = df.sort_values(by=time_var)
            # Interpolate missing values if necessary
            # Assuming interpolation is still relevant after standardization
            df[features] = df[features].interpolate(method="linear", limit_direction='both')
            # If after interpolation there are still NaNs (e.g., at the start or end), fill them
            df[features] = df[features].fillna(method='bfill').fillna(method='ffill')
            
            # Handle uniform length: truncate or pad with NaNs (which should be minimal after interpolation)
            if len(df) < max_length:
                # Padding: Create a padding DataFrame and concatenate
                padding = pd.DataFrame(np.NaN, index=range(max_length - len(df)), columns=features)
                df = pd.concat([df, padding], ignore_index=True)
            elif len(df) > max_length:
                # Truncate
                df = df.iloc[:max_length]

            aligned_dfs.append(df[features].values)
        
        # Form the 3D array from the list of aligned, possibly padded arrays
        data_array = np.stack(aligned_dfs, axis=0)
        
        return data_array

    #############################
    # DIMENSIONALITY REDUCTION  #
    #############################
    def aggregate_data(self, dfs):
        """Aggregate a list of DataFrames into a single DataFrame with numeric columns only."""
        combined_df = pd.concat(dfs, ignore_index=True)
        numeric_df = combined_df.select_dtypes(include=[np.number])
        return numeric_df

    def apply_umap(self, data, n_neighbors=15, min_dist=0.1, n_components=3):
        """
        Applies UMAP (Uniform Manifold Approximation and Projection) to the data.

        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors to consider for UMAP, by default 15.
        min_dist : float, optional
            Minimum distance between points in the low-dimensional representation, by default 0.1.
        n_components : int, optional
            Number of dimensions to reduce to, by default 2.
        """
        print("\tApplying UMAP...")
        self.umap_embedding = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components
        ).fit_transform(data)

    def apply_tsne(self, data, n_components=2):
        """
        Applies t-SNE (t-Distributed Stochastic Neighbor Embedding) to the data.

        Parameters
        ----------
        n_components : int, optional
            Number of dimensions to reduce to, by default 2.
        """
        print("Applying t-SNE...")
        self.tsne_embedding = TSNE(n_components=n_components).fit_transform(data)

    ######################
    # CLUSTERING METHODS #
    ######################
    def perform_time_series_kmeans(self, data):
        """
        Performs K-means clustering on the data.
        """        
        ts_kmeans = TimeSeriesKMeans(
            n_clusters=self.n_clusters, metric=self.kmeans_metric, verbose=self.kmeans_verbose
        )
        self.cluster_labels = ts_kmeans.fit_predict(data)
        self.ts_kmeans_model = ts_kmeans

    def perform_dbscan_clustering(self, data, eps=0.5, min_samples=5):
        """
        Performs DBSCAN clustering on the data.

        Parameters
        ----------
        eps : float, optional
            The maximum distance between two samples for one to be considered as
            in the neighborhood of the other, by default 0.5.
        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point to
            be considered as a core point, by default 5.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(data)

    ######################
    # ANALYSIS FUNCTIONS #
    ######################
    def cluster_analysis(self, n_clusters=3):
        """
        Performs further cluster analysis on the data.
        """
        for cluster in range(n_clusters):
            members = cluster == n_clusters
            print(f"Cluster {cluster} has {len(members)} members.")

        # TODO: Further analysis can be done here, for example:
        # - Average 'Normalized Volume' and 'Volume Change' for each cluster
        # - Plotting representative time series from each cluster
        # - Examining other clinical variables in relation to the clusters
        # - Temporal patterns in the clusters

    def evaluate_silhouette_score(self, data, method_name):
        """
        Evaluates the silhouette score for the current cluster labels
        and saves it in the silhouette_scores dictionary.

        Parameters
        ----------
        method_name : str
            The name of the clustering method for which the silhouette score is being calculated.

        Returns
        -------
        float
            The calculated silhouette score.
        """
        method_score = silhouette_score(data, self.cluster_labels)
        self.silhouette_scores[method_name] = method_score  # Save the score in the dictionary
        return method_score

    def save_metrics(self, method_name, method_score):
        """
        Appends the silhouette score for a given method to the metrics_output_path file.

        Parameters
        ----------
        method : str
            The name of the clustering method for which the silhouette score is being saved.
        score : float
            The silhouette score to be saved.
        """
        # Check if file exists; if not, create it
        if not os.path.exists(self.metrics_output_path):
            with open(self.metrics_output_path, "w", encoding="utf-8") as file:
                file.write("Method, Score\n")

        # Append the new metric
        with open(self.metrics_output_path, "a", encoding="utf-8") as file:
            file.write(f"{method_name}, {float(method_score)}\n")

    def validate_clusters(self):
        """
        Validates the clusters using the following metrics:
        - Average silhouette score
        - Davies-Bouldin index
        - Calinski-Harabasz index
        - Bootstraping
        """
        print("Validating clusters...")
        # TODO: Implement validation metrics like bootstraping, silhouette score, etc.

    ######################
    # PLOTTING FUNCTIONS #
    ######################
    def add_cluster_ellipses(self, a_x, embedding, labels, cmap):
        """
        Adds ellipses around the clusters in the scatter plot of embeddings.

        Parameters
        ----------
        a_x : matplotlib Axes
            The axes on which to draw the ellipses.
        embedding : np.array
            The 2D embeddings of the data.
        labels : np.array
            The cluster labels for each data point.
        cmap : matplotlib Colormap
            The colormap used to color the ellipses and data points.
        """
        for label in np.unique(labels):
            if label == -1:  # Exclude noise points represented by -1 in DBSCAN
                continue

            cluster_data = embedding[labels == label]
            cluster_color = cmap(label)
            lighter_color = [min(1, x + 0.3) for x in cluster_color[:3]] + [
                0.5
            ]  # Lighter and more transparent

            covar = np.cov(cluster_data, rowvar=False)
            eigenvals, eigenvecs = np.linalg.eigh(covar)
            order = eigenvals.argsort()[::-1]
            eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
            theta = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvals)

            ellip = Ellipse(
                xy=np.mean(cluster_data, axis=0),
                width=width,
                height=height,
                angle=theta,
                edgecolor=cluster_color,
                facecolor=lighter_color,
            )

            a_x.add_patch(ellip)

    def plot_dr_clusters_with_boundaries(self, suffix_name):
        """
        Plots the UMAP and t-SNE embeddings of the data with ellipses around the clusters.

        Parameters
        ----------
        suffix_name : str
            A string to be appended to the output file name.
        """
        plt.figure(figsize=(12, 6))

        # Get the colormap
        cmap = colormaps["tab10"]

        # Plot UMAP clusters with boundaries
        plt.subplot(1, 2, 1)
        a_x1 = plt.gca()
        a_x1.scatter(
            self.umap_embedding[:, 0],
            self.umap_embedding[:, 1],
            c=self.cluster_labels,
            cmap=cmap,
            s=5,
        )
        self.add_cluster_ellipses(a_x1, self.umap_embedding, self.cluster_labels, cmap)
        plt.title("UMAP Projection with Boundaries")

        # Plot t-SNE clusters with boundaries
        plt.subplot(1, 2, 2)
        a_x2 = plt.gca()
        a_x2.scatter(
            self.tsne_embedding[:, 0],
            self.tsne_embedding[:, 1],
            c=self.cluster_labels,
            cmap=cmap,
            s=5,
        )
        self.add_cluster_ellipses(a_x2, self.tsne_embedding, self.cluster_labels, cmap)
        plt.title("t-SNE Projection with Boundaries")

        path = os.path.join(self.output_path, suffix_name)
        plt.savefig(path)

    def plot_heatmap(self, data, suffix_name):
        """
        Plots a heatmap of the data, sorted by cluster labels.

        Parameters
        ----------
        suffix_name : str
            A string to be appended to the output file name.
        """
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        # Sort data according to cluster labels if applicable
        sorted_indices = np.argsort(self.cluster_labels)
        sorted_data = data[sorted_indices]

        plt.figure(figsize=(10, 8))

        # Create a heatmap
        a_x = sns.heatmap(sorted_data, cmap="coolwarm", annot=True, linewidths=0.5)

        # Draw lines to separate clusters
        boundaries = np.cumsum(
            [np.sum(self.cluster_labels == i) for i in np.unique(self.cluster_labels) if i != -1]
        )[:-1]
        for boundary in boundaries:
            a_x.axhline(boundary, color="black", linewidth=0.8)
            a_x.axvline(boundary, color="black", linewidth=0.8)

        # Add a color scale legend
        cbar = a_x.collections[0].colorbar
        cbar.set_label("Your Data Metric")

        plt.title(f"Cluster Heatmap ({suffix_name})")
        plt.xlabel("Your X-axis Label")
        plt.ylabel("Your Y-axis Label")

        path = os.path.join(self.output_path, f"heatmap_{suffix_name}")
        plt.savefig(path)

    def plot_embeddings(self, dr_method, prefix):
        """
        Plots the 2D embeddings from UMAP or t-SNE.

        Parameters
        ----------
        dr_method : str
            The dimensionality reduction method used ('umap' or 'tsne').
        """
        embedding = None
        if dr_method.lower() == "umap" and hasattr(self, 'umap_embedding') and self.umap_embedding is not None:
            embedding = self.umap_embedding
            title = "UMAP Embedding"
        elif dr_method.lower() == "tsne" and hasattr(self, 'tsne_embedding') and self.umap_embedding is not None:
            embedding = self.tsne_embedding
            title = "t-SNE Embedding"
        else:
            print(f"No embedding found for {dr_method}.")
            return

        centroid = np.mean(embedding, axis=0)
        distances = np.sqrt(np.sum((embedding - centroid)**2, axis=1))
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=distances, cmap="Spectral", s=5)
        plt.colorbar(scatter, label="Cluster Label")
    
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        plt.tight_layout()
        file_name = os.path.join(self.output_path, f"{prefix}_{dr_method}_embedding.png")
        plt.savefig(file_name)
        plt.close()


    ##########################
    # PLOTTING FUNCTIONS EDA #
    ##########################
    def plot_patient_data(self, dfs, prefix, n_samples=clustering_cfg.LIMIT_LOADING):
        """
        Plots curves for a sample of patients based on the list of DataFrames.
        Each DataFrame in the list is assumed to represent data for one patient.
        """
        # Sample a subset of data frames if there are more than n_samples patients
        sampled_dfs = dfs[:n_samples]

        for df in sampled_dfs:
            df.describe()
            patient_id = df["Patient_ID"].iloc[0]
            plt.figure(figsize=(12, 6))
            
            # Normalized volume data
            plt.subplot(1, 3, 1)
            plt.plot(
                df["Age"], df["Normalized Volume"], marker="o", linestyle="-"
            )
            plt.title(f"Patient ID: {patient_id} - Normalized Volume")
            plt.xlabel("Age")
            plt.ylabel("Normalized Volume")

            # Volume Growth
            plt.subplot(1, 3, 2)
            plt.plot(df["Age"], df["Volume Growth[%]"], marker="o", linestyle="-")
            plt.title(f"Patient ID: {patient_id} - Volume Growth")
            plt.xlabel("Age")
            plt.ylabel("Volume Growth [%]")

            # Volume Growth Rate data
            plt.subplot(1, 3, 3)
            plt.plot(df["Age"], df["Volume Growth[%] Rate"], marker="o", linestyle="-")
            plt.title(f"Patient ID: {patient_id} - Volume Growth Rate")
            plt.xlabel("Age")
            plt.ylabel("Volume Growth Rate [%]")

            plt.tight_layout()
            file_name = os.path.join(self.output_path, f"{prefix}_{patient_id}.png")
            plt.savefig(file_name)
            plt.close()

    def plot_avg_std_across_patients(self, data, prefix, feature_names):
        """
        Plots the average and standard deviation of 'Normalized Volume', "Volume Growth" 
        and 'Volume Growth Rate' across all patients.
        """
        if isinstance(data, list):  # Handling list of pandas DataFrames
            # Aggregate data from each DataFrame
            aggregated_data = {name: [df[name].mean() for df in data] for name in feature_names}
        elif isinstance(data, np.ndarray):  # Handling NumPy ndarray
            # Assuming ndarray shape is (n_patients, n_samples, n_features)
            # and feature order matches `feature_names`
            aggregated_data = {name: data[:, :, i].mean(axis=1) for i, name in enumerate(feature_names)}
        else:
            raise ValueError("Data must be either a list of pandas DataFrames or a NumPy ndarray.")
        
        plt.figure(figsize=(14, 6))
        for i, (name, values) in enumerate(aggregated_data.items(), start=1):
            avg = np.mean(values)
            std = np.std(values)
            plt.subplot(1, len(aggregated_data), i)
            plt.bar([name], [avg], yerr=[std], capsize=5)
            plt.title(f"Average and Std Dev of {name}")
            plt.ylabel(name)
        
        plt.tight_layout()
        file_name = os.path.join(self.output_path, f"{prefix}_avg_std_across_patients.png")
        plt.savefig(file_name)
        plt.close()

    def plot_pairwise_relationships(self, data, prefix, feature_names):
        """
        Pairplot of all features across all patients.        
        """
        if isinstance(data, list):  # Handling list of pandas DataFrames
            all_patient_data = pd.concat(data, ignore_index=True)
        elif isinstance(data, np.ndarray):  # Handling NumPy ndarray
            # Convert ndarray to DataFrame
            all_patient_data = pd.DataFrame(data.reshape(-1, data.shape[-1]), columns=feature_names)
        else:
            raise ValueError("Data must be either a list of pandas DataFrames or a NumPy ndarray.")
        sns.pairplot(all_patient_data, vars=feature_names)
        plt.savefig(os.path.join(self.output_path, f"{prefix}_pairplot_all_features.png"))
        plt.close()


if __name__ == "__main__":
    STEP = 0
    # Data loading and initialization
    print(f"Step {STEP}: Initializing clustering script, reading data from {clustering_cfg.COHORT}...")
    cluster_analysis = ClusterAnalysis()
    data_frames = cluster_analysis.load_data(clustering_cfg.INPUT_PATH, clustering_cfg.SELECTED_FEATURES, clustering_cfg.LIMIT_LOADING)
    print("\tCluster analysis class initialized. Data loaded.")
    STEP += 1
    time_series_features = ["Age", "Normalized Volume", "Volume Growth[%]", "Volume Growth[%] Rate"]

    # Data analysis, embedding and plotting
    # FIXME - This is printing wrongly the data
    if clustering_cfg.PLOT_PATIENT_DATA_EDA:
        print(f"Step {STEP}: Plotting patient data for EDA...")
        pre="pre_scaling"
        #cluster_analysis.plot_patient_data(data_frames, pre)
        cluster_analysis.plot_avg_std_across_patients(data_frames, pre, time_series_features)
        cluster_analysis.plot_pairwise_relationships(data_frames, pre, time_series_features)
        print("\tPatient data plotted.")
        STEP += 1
        if clustering_cfg.USE_UMAP:
            numeric_data = cluster_analysis.aggregate_data(data_frames)
            cluster_analysis.apply_umap(numeric_data)
            cluster_analysis.plot_embeddings("umap", pre)
            print("\tPatient data plotted.")
        if clustering_cfg.USE_TSNE:
            numeric_data = cluster_analysis.aggregate_data(data_frames)
            cluster_analysis.apply_tsne(numeric_data)
            cluster_analysis.plot_embeddings("tsne", pre)
            print("\tPatient data plotted.")
    
    # Data stadarization and plotting
    print(f"Step {STEP}: Standardizing data...")
    data_frames = cluster_analysis.standardize_data(data_frames)
    print("\tData standardized.")
    STEP += 1
    prepared_data = cluster_analysis.align_and_interpolate_time_series(data_frames, time_var="Age", features=time_series_features)
    
    # FIXME - This is printing wrongly the data
    if clustering_cfg.PLOT_PATIENT_DATA_SCALING:
        print(f"Step {STEP}: Plotting patient data adter scaling...")
        post="post_scaling"
        #cluster_analysis.plot_patient_data(data_frames, post)
        cluster_analysis.plot_avg_std_across_patients(data_frames, post, time_series_features)
        cluster_analysis.plot_pairwise_relationships(data_frames, post, time_series_features)
        print("\tPatient data plotted.")
        STEP += 1
        if clustering_cfg.USE_UMAP:
            numeric_data = cluster_analysis.aggregate_data(data_frames)
            cluster_analysis.apply_umap(numeric_data)
            cluster_analysis.plot_embeddings("umap", post)
            print("\tPatient data plotted.")
        if clustering_cfg.USE_TSNE:
            numeric_data = cluster_analysis.aggregate_data(data_frames)
            cluster_analysis.apply_tsne(numeric_data)
            cluster_analysis.plot_embeddings("tsne", post)
            print("\tPatient data plotted.")
        
    # Definition of methods to be used in the clustering, multiple possible
    methods_and_suffixes = {
        "K-means": ("perform_time_series_kmeans", "clustering_ts_kmeans.png"),
        "DBSCAN": ("perform_dbscan_clustering", "clustering_dbscan.png"),
        # "DeepClustering": ("deep_clustering", "clustering_deepcl.png"),
        # "Hierarchical": ("perform_hierarchical_clustering, "clustering_hierarchical.png"),
    }

    for method, (func_name, suffix) in tqdm(methods_and_suffixes.items(), desc="Clustering methods"):
        print(f"Step {STEP}: Performing clustering method: {method}.")
        func = getattr(cluster_analysis, func_name)
        if func_name == "perform_dbscan_clustering":
            n_samples, n_timesteps, n_features = prepared_data.shape
            # FIXME - DBSCAN fails due to NaN in data
            flattened_data = prepared_data.reshape(n_samples, n_timesteps * n_features)
            func(flattened_data)
        else:
            func(prepared_data)
        if len(np.unique(cluster_analysis.cluster_labels)) > 1 and prepared_data.shape[0] == 2:
            score = cluster_analysis.evaluate_silhouette_score(prepared_data.reshape(-1, prepared_data.shape[-1]), method)
            print(f"Silhouette score for {method}: {score}")
            cluster_analysis.save_metrics(method, score)
        else:
            print(f"{method} found only one cluster. Silhouette score is not applicable.")

        print(f"\tClustering method {method} performed.")

        # Plotting
        if clustering_cfg.USE_UMAP and clustering_cfg.USE_TSNE:
            print(f"Step {STEP}: Plotting dimensionality reduction with boundaries...")
            cluster_analysis.plot_dr_clusters_with_boundaries(suffix)
        if clustering_cfg.PLOT_HEATMAP:
            cluster_analysis.plot_heatmap(prepared_data, suffix)

        STEP += 1
