# -*- coding: utf-8 -*-
# - Synthetic CLD Data Analysis Script
# - Creation: 2024-01-30
# - Update: 2024-01-31
# - author: Flower Yang
#
"""
This script performs synthetic CLD data analysis, including summary statistics generation,
visualization of variable distributions, and clustering analyses.
"""

import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from nltk import Text
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from tabulate import tabulate
import squarify

warnings.filterwarnings("ignore", category=FutureWarning)

# Load synthetic dataset
def load_data(file_path):
    """
    Load and preprocess the synthetic dataset.
    Extract nodes, edges, and cycles from the CLDs in DOT format.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    df = pd.read_csv(file_path, encoding="utf-8")

    # Extract nodes and edges from DOT format
    df["node_list"] = df["generated_cld_DOT"].apply(lambda x: re.findall(r'"(.*?)"', x))
    df["num_nodes"] = df["node_list"].apply(len)

    def extract_edges(dot_string):
        return re.findall(r'"(.*?)"\s*->\s*"(.*?)"', dot_string)

    df["edge_list"] = df["generated_cld_DOT"].apply(extract_edges)
    df["num_links"] = df["edge_list"].apply(len)

    # Count cycles using networkx
    def count_cycles(dot_string):
        G = nx.DiGraph()
        edges = extract_edges(dot_string)
        G.add_edges_from(edges)
        return len(list(nx.simple_cycles(G)))

    df["num_loops"] = df["generated_cld_DOT"].apply(count_cycles)

    return df

# Generate summary statistics table
def generate_summary_table(df):
    """
    Generate a summary table of the dataset characteristics.

    Args:
        df (pd.DataFrame): Dataset with computed node, edge, and loop counts.

    Returns:
        pd.DataFrame: Transposed summary table for horizontal display.
    """
    summary_stats = df[["num_nodes", "num_links", "num_loops"]].describe().T
    summary_stats["Reference CLD"] = [4, 5, 2]  # Replace with actual reference values
    summary_stats = summary_stats[["mean", "50%", "min", "max", "std", "Reference CLD"]]
    summary_stats.rename(columns={"50%": "median"}, inplace=True)
    summary_stats = summary_stats.T
    summary_stats.columns = ["Nodes", "Edges", "Cycles"]
    print(tabulate(summary_stats, headers="keys", tablefmt="grid"))
    return summary_stats

# Dispersion plot of top variable names
def plot_dispersion(df):
    """
    Generate a dispersion plot for the top 15 variable names.

    Args:
        df (pd.DataFrame): Dataset with variable lists.
    """
    word_sequence = [node for nodes in df["node_list"] for node in nodes]
    text_object = Text(word_sequence)
    top_keywords = df["node_list"].explode().value_counts().head(15).index.tolist()

    plt.figure(figsize=(12, 6))
    text_object.dispersion_plot(top_keywords)
    plt.title("Dispersion Plot of Top 15 Variable Names")
    plt.xlabel("CLD Index")
    plt.ylabel("Variable Names")
    plt.tight_layout()
    plt.show()

# Tree map visualization of variable names
def plot_treemap(df):
    """
    Generate a tree map for the top 20 variable names.

    Args:
        df (pd.DataFrame): Dataset with variable lists.
    """
    top_variables = df["node_list"].explode().value_counts().head(20).reset_index()
    top_variables.columns = ["Variable", "Count"]

    plt.figure(figsize=(10, 6))
    squarify.plot(sizes=top_variables["Count"], label=top_variables["Variable"], alpha=0.8,
                  color=sns.color_palette("viridis", len(top_variables)))
    plt.title("Tree Map of Top 20 Variable Names")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Hierarchical clustering heatmap
def plot_variable_clusters(df, top_n=15):
    """
    Generate a heatmap based on hierarchical clustering of the top variable names.

    Args:
        df (pd.DataFrame): Dataset with variable lists.
        top_n (int): Number of top variables to include in the clustering.
    """
    top_variables = df["node_list"].explode().value_counts().head(top_n).index.tolist()
    unique_variables = list(set(top_variables))
    adjacency_matrix = np.zeros((len(unique_variables), len(unique_variables)))

    for nodes in df["node_list"]:
        filtered_nodes = [node for node in nodes if node in top_variables]
        for i in range(len(filtered_nodes)):
            for j in range(len(filtered_nodes)):
                if i != j:
                    adjacency_matrix[unique_variables.index(filtered_nodes[i]),
                                     unique_variables.index(filtered_nodes[j])] += 1

    adjacency_matrix += 1e-6  # Avoid division by zero
    distance_matrix = 1 / adjacency_matrix
    np.fill_diagonal(distance_matrix, 0)
    linkage_matrix = linkage(squareform(distance_matrix), method="ward")

    dendro = dendrogram(linkage_matrix, no_plot=True)
    ordered_indices = dendro["leaves"]
    ordered_variables = [unique_variables[i] for i in ordered_indices]

    plt.figure(figsize=(12, 10))
    sns.heatmap(adjacency_matrix[np.ix_(ordered_indices, ordered_indices)],
                xticklabels=ordered_variables, yticklabels=ordered_variables,
                cmap="coolwarm", linewidths=0.5)
    plt.title(f"Cluster Heatmap of Top {top_n} Variable Names")
    plt.xlabel("Variables")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()

# K-Means clustering visualization
def plot_variable_clusters_kmeans(df, top_n=15, cluster_numbers=[4, 5, 6, 7, 10]):
    """
    Perform and visualize K-Means clustering for top variable names.

    Args:
        df (pd.DataFrame): Dataset with variable lists.
        top_n (int): Number of top variables to include in the clustering.
        cluster_numbers (list): List of cluster counts to visualize.
    """
    top_variables = df["node_list"].explode().value_counts().head(top_n).index.tolist()
    co_occurrence = np.zeros((top_n, top_n))

    for nodes in df["node_list"]:
        indices = [top_variables.index(node) for node in nodes if node in top_variables]
        for i in indices:
            for j in indices:
                if i != j:
                    co_occurrence[i, j] += 1

    for n_clusters in cluster_numbers:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(co_occurrence)

        plt.figure(figsize=(10, 8))
        for i, label in enumerate(np.unique(labels)):
            cluster_items = [top_variables[idx] for idx, cluster in enumerate(labels) if cluster == label]
            plt.scatter([label] * len(cluster_items), cluster_items, label=f"Cluster {i+1}")

        plt.xticks(range(n_clusters), [f"Cluster {i+1}" for i in range(n_clusters)])
        plt.yticks(range(len(top_variables)), top_variables)
        plt.title(f"K-Means Clustering of Top {top_n} Variables ({n_clusters} Clusters)")
        plt.xlabel("Clusters")
        plt.ylabel("Variable Names")
        plt.legend()
        plt.tight_layout()
        plt.show()

# Main script execution
if __name__ == "__main__":
    file_path = "synthetic_cld_data.csv"  # Update with actual file path
    df = load_data(file_path)

    print("Summary Table:")
    summary_table = generate_summary_table(df)

    print("\nNumber of unique variable names:", len(set(df["node_list"].explode())))

    plot_dispersion(df)
    plot_treemap(df)
    plot_variable_clusters(df, top_n=15)
    plot_variable_clusters_kmeans(df, top_n=15, cluster_numbers=[4, 5, 6, 7, 10])
