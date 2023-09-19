import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse
from matplotlib import colormaps
import matplotlib.pyplot as plt
import umap
import faiss
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class ClusterAnalysis:
    def __init__(self, dataframe, patient_id_column, output_path, metrics_output_path):
        self.data = dataframe
        self.patient_id_column = patient_id_column
        self.data_to_cluster = self.data.drop(columns=[self.patient_id_column]).values.astype('float32')
        self.umap_embedding = None
        self.tsne_embedding = None
        self.cluster_labels = None
        self.output_path = output_path
        self.silhouette_scores = {}  # Initialize an empty dictionary to store silhouette scores
        self.metrics_output_path = metrics_output_path

    def standardize_data(self):
        scaler = StandardScaler()
        self.data_to_cluster = scaler.fit_transform(self.data_to_cluster)
        
    def apply_umap(self, n_neighbors=15, min_dist=0.1, n_components=2):
        self.umap_embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components).fit_transform(self.data_to_cluster)

    def apply_tsne(self, n_components=2):
        self.tsne_embedding = TSNE(n_components=n_components).fit_transform(self.data_to_cluster)

    def perform_kmeans_clustering(self, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters)
        self.cluster_labels = kmeans.fit_predict(self.data_to_cluster)
        
    def perform_faiss_clustering(self, n_clusters=3):
        index = faiss.IndexFlatL2(self.data_to_cluster.shape[1])
        index.add(self.data_to_cluster)
        _, self.cluster_labels = index.search(self.data_to_cluster, n_clusters)
        self.cluster_labels = np.argmin(self.cluster_labels, axis=1)

    def deep_clustering(self, n_clusters=3, epochs=100):
        model = nn.Sequential(
            nn.Linear(self.data_to_cluster.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_clusters)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        tensor_data = torch.tensor(self.data_to_cluster)
        tensor_labels = torch.tensor(self.cluster_labels)
        dataset = TensorDataset(tensor_data, tensor_labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in tqdm(range(epochs), desc="Training"):
            for batch_data, _ in loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, torch.argmax(outputs, dim=1))
                loss.backward()
                optimizer.step()

        self.cluster_labels = torch.argmax(model(tensor_data).detach(), dim=1).numpy()
    
    def evaluate_silhouette_score(self, method_name):
        score = silhouette_score(self.data_to_cluster, self.cluster_labels)
        self.silhouette_scores[method_name] = score  # Save the score in the dictionary
        return score
    
    def save_metrics(self, method, score):
        # Check if file exists; if not, create it
        if not os.path.exists(self.metrics_output_path):
            with open(self.metrics_output_path, 'w') as f:
                f.write("Method, Score\n")

        # Append the new metric
        with open(self.metrics_output_path, 'a') as f:
            f.write(f"{method}, {float(score)}\n")

    def plot_clusters(self, suffix):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(self.umap_embedding[:, 0], self.umap_embedding[:, 1], c=self.cluster_labels, cmap='tab10', s=5)
        plt.colorbar(boundaries=np.arange(min(self.cluster_labels) - 0.5, max(self.cluster_labels) + 1.5)).set_ticks(np.unique(self.cluster_labels))
        plt.title('UMAP Projection')
        
        plt.subplot(1, 2, 2)
        plt.scatter(self.tsne_embedding[:, 0], self.tsne_embedding[:, 1], c=self.cluster_labels, cmap='tab10', s=5)
        plt.colorbar(boundaries=np.arange(min(self.cluster_labels) - 0.5, max(self.cluster_labels) + 1.5)).set_ticks(np.unique(self.cluster_labels))
        plt.title('t-SNE Projection')
        
        path = os.path.join(self.output_path, suffix)
        plt.savefig(path)

    def perform_dbscan_clustering(self, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(self.data_to_cluster)

    def add_cluster_ellipses(self, ax, embedding, labels, cmap):
        for label in np.unique(labels):
            if label == -1:  # Exclude noise points represented by -1 in DBSCAN
                continue

            cluster_data = embedding[labels == label]
            cluster_color = cmap(label)
            lighter_color = [min(1, x + 0.3) for x in cluster_color[:3]] + [0.5]  # Lighter and more transparent

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
                facecolor=lighter_color
            )
            
            ax.add_patch(ellip)

    def plot_clusters_with_boundaries(self, suffix):
        plt.figure(figsize=(12, 6))
        
        # Get the colormap
        cmap = colormaps['tab10']

        # Plot UMAP clusters with boundaries
        plt.subplot(1, 2, 1)
        ax1 = plt.gca()
        ax1.scatter(self.umap_embedding[:, 0], self.umap_embedding[:, 1], c=self.cluster_labels, cmap=cmap, s=5)
        self.add_cluster_ellipses(ax1, self.umap_embedding, self.cluster_labels, cmap)
        plt.title('UMAP Projection with Boundaries')

        # Plot t-SNE clusters with boundaries
        plt.subplot(1, 2, 2)
        ax2 = plt.gca()
        ax2.scatter(self.tsne_embedding[:, 0], self.tsne_embedding[:, 1], c=self.cluster_labels, cmap=cmap, s=5)
        self.add_cluster_ellipses(ax2, self.tsne_embedding, self.cluster_labels, cmap)
        plt.title('t-SNE Projection with Boundaries')

        path = os.path.join(self.output_path, suffix)
        plt.savefig(path)

    def plot_heatmap(self, suffix):
        # Sort data according to cluster labels if applicable
        sorted_indices = np.argsort(self.cluster_labels)
        sorted_data = self.data_to_cluster[sorted_indices]
        
        plt.figure(figsize=(10, 8))

        # Create a heatmap
        ax = sns.heatmap(sorted_data, cmap='coolwarm', annot=True, linewidths=.5)
        
        # Draw lines to separate clusters
        boundaries = np.cumsum([np.sum(self.cluster_labels == i) for i in np.unique(self.cluster_labels) if i != -1])[:-1]
        for boundary in boundaries:
            ax.axhline(boundary, color='black', linewidth=0.8)
            ax.axvline(boundary, color='black', linewidth=0.8)

        # Add a color scale legend
        cbar = ax.collections[0].colorbar
        cbar.set_label('Your Data Metric')
        
        plt.title(f'Cluster Heatmap ({suffix})')
        plt.xlabel('Your X-axis Label')
        plt.ylabel('Your Y-axis Label')
        
        path = os.path.join(self.output_path, f"heatmap_{suffix}")
        plt.savefig(path)


if __name__ == "__main__":
    all_data = pd.DataFrame({'patient_id': range(100), 'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})
    
    output_path = "/home/jc053/GIT/mri-longitudinal-analysis/data/output/plots/"
    metrics_output_path = "/home/jc053/GIT/mri-longitudinal-analysis/data/output/metrics.txt"

    cluster_analysis = ClusterAnalysis(all_data, 'patient_id', output_path, metrics_output_path)
    
    cluster_analysis.standardize_data()
    cluster_analysis.apply_umap()
    cluster_analysis.apply_tsne()
    
    methods_and_suffixes = {
        'DBSCAN': ('perform_dbscan_clustering', 'clustering_dbscan.png'),
        'K-means': ('perform_kmeans_clustering', 'clustering_kmeans.png'),
        'Faiss': ('perform_faiss_clustering', 'clustering_faiss.png'),
        'DeepClustering': ('deep_clustering', 'clustering_deepcl.png')
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
