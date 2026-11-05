"""
Evaluation Module for Customer Segmentation.
Calculates cluster distribution metrics, intra-cluster statistical summaries,
and validation scores (Silhouette, Davies-Bouldin, Calinski-Harabasz).
"""

import pandas as pd
import numpy as np


def compute_cluster_statistics(df_full: pd.DataFrame, cluster_labels: list[int], feature_names: list[str]) -> list[dict]:
    """
    Computes in-depth statistical distribution for each cluster.
    Includes count, share, feature mean, std, min, max, and demographic breakdowns.
    """
    df = df_full.copy()
    df['Cluster'] = cluster_labels
    total_samples = len(df)
    
    cluster_stats = []
    unique_clusters = sorted(df['Cluster'].unique())

    for c in unique_clusters:
        c_df = df[df['Cluster'] == c]
        count = len(c_df)
        share = round((count / total_samples) * 100, 2)

        feat_metrics = {}
        for feat in feature_names:
            if feat in c_df.columns:
                feat_metrics[feat] = {
                    'mean': round(float(c_df[feat].mean()), 2),
                    'std': round(float(c_df[feat].std()), 2) if count > 1 else 0.0,
                    'min': round(float(c_df[feat].min()), 2),
                    'median': round(float(c_df[feat].median()), 2),
                    'max': round(float(c_df[feat].max()), 2)
                }

        # Demographic breakdown if Gender is available
        gender_split = {}
        if 'Gender' in c_df.columns:
            gender_split = c_df['Gender'].value_counts().to_dict()

        cluster_stats.append({
            'cluster_id': int(c),
            'count': int(count),
            'share_percentage': share,
            'features': feat_metrics,
            'gender_distribution': gender_split
        })

    return cluster_stats


def get_cluster_comparison_matrix(df_full: pd.DataFrame, cluster_labels: list[int], feature_names: list[str]) -> dict:
    """
    Return a structured matrix of cluster averages suitable for Chart.js grouped bar charts.
    """
    df = df_full.copy()
    df['Cluster'] = cluster_labels
    unique_clusters = sorted(df['Cluster'].unique())

    matrix = {
        'clusters': [f"Cluster {c}" for c in unique_clusters],
        'features': feature_names,
        'data': {}
    }

    for feat in feature_names:
        matrix['data'][feat] = [
            round(float(df[df['Cluster'] == c][feat].mean()), 2)
            for c in unique_clusters
        ]

    return matrix
