"""
Preprocessing Module for Customer Segmentation.
Handles data cleaning, missing value handling, feature scaling with StandardScaler,
and PCA dimensionality reduction for multi-feature visual projection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DEFAULT_FEATURES = ['Annual Income (k$)', 'Spending Score (1-100)']
ALL_NUMERICAL_FEATURES = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']


def prepare_features(df: pd.DataFrame, feature_names: list[str] = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Select and validate requested features from dataframe.
    Cleans missing values by imputing with column median.
    """
    df = df.copy()
    if not feature_names:
        # Default to Age, Annual Income, Spending Score if present, else Income & Spending
        if all(f in df.columns for f in ALL_NUMERICAL_FEATURES):
            feature_names = ALL_NUMERICAL_FEATURES
        else:
            feature_names = [f for f in DEFAULT_FEATURES if f in df.columns]
            
    # Verify selected features exist
    available_features = [f for f in feature_names if f in df.columns]
    if len(available_features) < 2:
        raise ValueError("At least 2 numerical features are required for clustering.")

    # Missing value handling (median imputation)
    X = df[available_features].copy()
    for col in available_features:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    return X, available_features


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler to achieve zero mean and unit variance.
    Essential for distance-based clustering algorithms like K-Means.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def apply_pca_reduction(X_scaled: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA, list[float]]:
    """
    Apply Principal Component Analysis (PCA) to project multi-dimensional feature space
    into a 2D plane for visualization.
    Returns: (X_pca, pca_model, explained_variance_ratios)
    """
    n_comp = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    return X_pca, pca, explained_variance
