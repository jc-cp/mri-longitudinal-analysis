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

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as torch
import torch.nn as nn
import umap
from matplotlib import colormaps
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from cfg import clustering_cfg


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

    def __init__(self, dataframe, patient_id_column, output_plots, output_metrics):
        self.data = dataframe
        self.patient_id_column = patient_id_column
        self.unique_patients = self.data[self.patient_id_column].unique()
        self.time_series = []

        for patient in self.unique_patients:
            patient_data = self.data[self.data["Patient_ID"] == patient]
            series = patient_data[["Normalized Volume", "Volume Change"]].to_numpy()
            self.time_series.append(series)
        self.time_series = np.array(self.time_series)
        if self.time_series.isnan():
            print(
                "Warning: NaN values found in the data. Changing them to 0. Please remove them"
                " before proceeding."
            )
            self.time_series = np.nan_to_num(self.time_series)

        self.umap_embedding = None
        self.tsne_embedding = None
        self.cluster_labels = None

        self.output_path = output_plots
        self.metrics_output_path = output_metrics

        self.silhouette_scores = {}

        self.time_series_scaled = self.standardize_data()

    #####################
    # UTILITY FUNCTIONS #
    #####################

    def standardize_data(self):
        """
        Standardizes and scales the numerical data.
        """
        scaler = TimeSeriesScalerMeanVariance()
        time_series_scaled = scaler.fit_transform(self.time_series)
        return time_series_scaled

    def align_time_series(self):
        """
        Aligns the time series by interpolating shorter series to match the length of the longest series.
        """
        # TODO: Implement this function

    #############################
    # DIMENSIONALITY REDUCTION  #
    #############################

    def apply_umap(self, n_neighbors=15, min_dist=0.1, n_components=2):
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
        print("Applying UMAP...")
        self.umap_embedding = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components
        ).fit_transform(self.data)

    def apply_tsne(self, n_components=2):
        """
        Applies t-SNE (t-Distributed Stochastic Neighbor Embedding) to the data.

        Parameters
        ----------
        n_components : int, optional
            Number of dimensions to reduce to, by default 2.
        """
        print("Applying t-SNE...")
        self.tsne_embedding = TSNE(n_components=n_components).fit_transform(self.data)

    ######################
    # CLUSTERING METHODS #
    ######################

    def perform_time_series_kmeans(self, n_clusters, metric, verbose):
        """
        Performs K-means clustering on the data.

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters to form, by default 3.
        """
        ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, verbose=verbose)
        cluster_labels = ts_kmeans.fit_predict(self.time_series_scaled)

    def perform_faiss_clustering(self, n_clusters=3):
        """
        Performs clustering using the Faiss library on the data.

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters to form, by default 3.
        """
        index = faiss.IndexFlatL2(self.data.shape[1])
        index.add(self.data)
        _, self.cluster_labels = index.search(self.data, n_clusters)
        self.cluster_labels = np.argmin(self.cluster_labels, axis=1)

    def deep_clustering(self, n_clusters=3, epochs=100):
        """
        Performs deep clustering using a simple neural network model on the data.

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters to form, by default 3.
        epochs : int, optional
            The number of training epochs, by default 100.
        """
        model = nn.Sequential(
            nn.Linear(self.data.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_clusters),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        tensor_data = torch.Tensor(self.data)
        tensor_labels = torch.Tensor(self.cluster_labels)
        dataset = TensorDataset(tensor_data, tensor_labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in tqdm(range(epochs), desc="Training"):
            for batch_data, batch_labels in loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                predicted_labels = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, batch_labels.long())
                loss.backward()
                optimizer.step()

        self.cluster_labels = predicted_labels.detach().numpy()

    def perform_dbscan_clustering(self, eps=0.5, min_samples=5):
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
        self.cluster_labels = dbscan.fit_predict(self.data)

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

    def evaluate_silhouette_score(self, method_name):
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
        method_score = silhouette_score(self.data, self.cluster_labels)
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

    def plot_dimensionality_reduction(self, suffix_name):
        """
        Plots the UMAP and t-SNE embeddings of the data, colored by cluster labels.

        Parameters
        ----------
        suffix_name : str
            A string to be appended to the output file name.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(
            self.umap_embedding[:, 0],
            self.umap_embedding[:, 1],
            c=self.cluster_labels,
            cmap="tab10",
            s=5,
        )
        plt.colorbar(
            boundaries=np.arange(min(self.cluster_labels) - 0.5, max(self.cluster_labels) + 1.5)
        ).set_ticks(np.unique(self.cluster_labels))
        plt.title("UMAP Projection")

        plt.subplot(1, 2, 2)
        plt.scatter(
            self.tsne_embedding[:, 0],
            self.tsne_embedding[:, 1],
            c=self.cluster_labels,
            cmap="tab10",
            s=5,
        )
        plt.colorbar(
            boundaries=np.arange(min(self.cluster_labels) - 0.5, max(self.cluster_labels) + 1.5)
        ).set_ticks(np.unique(self.cluster_labels))
        plt.title("t-SNE Projection")

        path = os.path.join(self.output_path, suffix_name)
        plt.savefig(path)

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

    def plot_clusters_with_boundaries(self, suffix_name):
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

    def plot_heatmap(self, suffix_name):
        """
        Plots a heatmap of the data, sorted by cluster labels.

        Parameters
        ----------
        suffix_name : str
            A string to be appended to the output file name.
        """
        # Sort data according to cluster labels if applicable
        sorted_indices = np.argsort(self.cluster_labels)
        sorted_data = self.data[sorted_indices]

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


if __name__ == "__main__":
    print("Running clustering script, reading data...")
    pre_treatment_data = pd.read_csv(clustering_cfg.INPUT_PATH)
    pre_treatment_data["Date"] = pd.to_datetime(pre_treatment_data["Date"])
    pre_treatment_data.sort_values(by=["Patient_ID", "Date"], inplace=True)

    output_path = clustering_cfg.PLOT_OUTPUT_PATH
    metrics_output_path = clustering_cfg.METRICS_OUTPUT_PATH

    cluster_analysis = ClusterAnalysis(
        pre_treatment_data, "Patient_ID", output_path, metrics_output_path
    )
    print("Cluster analysis class initialized.")

    # cluster_analysis.apply_umap()
    # cluster_analysis.apply_tsne()

    methods_and_suffixes = {
        # "DBSCAN": ("perform_dbscan_clustering", "clustering_dbscan.png"),
        "K-means": ("perform_time_series_kmeans", "clustering_ts_kmeans.png"),
        # "Faiss": ("perform_faiss_clustering", "clustering_faiss.png"),
        # "DeepClustering": ("deep_clustering", "clustering_deepcl.png"),
    }

    for method, (func, suffix) in tqdm(methods_and_suffixes.items(), desc="Clustering methods"):
        print(f"Performing clustering method: {method}.")
        getattr(cluster_analysis, func)()
        if len(np.unique(cluster_analysis.cluster_labels)) > 1:
            score = cluster_analysis.evaluate_silhouette_score(method)
            print(f"Silhouette score for {method}: {score}")
            cluster_analysis.save_metrics(method, score)
        else:
            print(f"{method} found only one cluster. Silhouette score is not applicable.")
        cluster_analysis.plot_clusters_with_boundaries(suffix)
        cluster_analysis.plot_heatmap(suffix)
