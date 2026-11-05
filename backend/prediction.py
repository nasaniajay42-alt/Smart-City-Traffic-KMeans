"""
Prediction Module for Customer Segmentation.
Uses the persisted fitted scaler and K-Means model to predict customer clusters
from raw input values, returning dynamic segment names and tailored marketing recommendations.
"""

import numpy as np
import pandas as pd
from .clustering import load_model_bundle


def validate_customer_input(input_data: dict, feature_names: list[str]) -> tuple[bool, str, dict]:
    """
    Validates input numerical fields for inference.
    Returns (is_valid, error_message, parsed_values).
    """
    parsed = {}
    
    # Mapping alternative key names
    key_aliases = {
        'Age': ['age', 'Age'],
        'Annual Income (k$)': ['annual_income', 'annual_income_k', 'income', 'Annual Income (k$)'],
        'Spending Score (1-100)': ['spending_score', 'spending', 'score', 'Spending Score (1-100)']
    }

    for feat in feature_names:
        found_val = None
        for alias in key_aliases.get(feat, [feat]):
            if alias in input_data and input_data[alias] is not None and str(input_data[alias]).strip() != "":
                found_val = input_data[alias]
                break

        if found_val is None:
            return False, f"Missing required input feature: '{feat}'", {}

        try:
            val = float(found_val)
        except (ValueError, TypeError):
            return False, f"Invalid value for '{feat}': must be a valid number.", {}

        # Domain boundary checks
        if 'Age' in feat and not (10 <= val <= 110):
            return False, f"Age ({val}) must be between 10 and 110.", {}
        if 'Income' in feat and not (1 <= val <= 1000):
            return False, f"Annual Income ({val}) must be between 1 and 1000 k$.", {}
        if 'Spending' in feat and not (1 <= val <= 100):
            return False, f"Spending Score ({val}) must be between 1 and 100.", {}

        parsed[feat] = val

    return True, "", parsed


def predict_customer_segment(input_data: dict) -> dict:
    """
    Predict cluster for a new customer based on input values.
    Uses the exact fitted StandardScaler and KMeans model saved from training.
    """
    bundle = load_model_bundle()
    if not bundle:
        raise ValueError("No trained model found. Please train a K-Means model first.")

    model = bundle['model']
    scaler = bundle['scaler']
    feature_names = bundle['feature_names']
    profiles = {p['cluster_id']: p for p in bundle['cluster_profiles']}
    pca_model = bundle.get('pca_model')

    # Validate input
    is_valid, err_msg, parsed_features = validate_customer_input(input_data, feature_names)
    if not is_valid:
        raise ValueError(err_msg)

    # Prepare vector in identical feature order with proper column names
    feature_df = pd.DataFrame([[parsed_features[feat] for feat in feature_names]], columns=feature_names)
    
    # Scale using the saved scaler
    scaled_vector = scaler.transform(feature_df)

    # Predict cluster ID
    predicted_cluster_id = int(model.predict(scaled_vector)[0])
    
    # Calculate Euclidean distance to each cluster centroid
    centroids = model.cluster_centers_
    distances = np.linalg.norm(centroids - scaled_vector, axis=1)
    distance_to_cluster = round(float(distances[predicted_cluster_id]), 4)

    # Calculate coordinates for visualization
    # 1. Raw coordinates (Income vs Spending)
    scatter_x = parsed_features.get('Annual Income (k$)', None)
    scatter_y = parsed_features.get('Spending Score (1-100)', None)

    # 2. PCA coordinates if PCA was used
    pca_coords = None
    if pca_model is not None and feature_df.shape[1] >= 2:
        try:
            coords = pca_model.transform(scaled_vector)[0]
            pca_coords = {
                'pc1': round(float(coords[0]), 3),
                'pc2': round(float(coords[1]), 3)
            }
        except Exception:
            pass

    # Retrieve profile
    cluster_profile = profiles.get(predicted_cluster_id, {
        'cluster_id': predicted_cluster_id,
        'segment_name': f"Cluster {predicted_cluster_id}",
        'archetype': "Customer Segment",
        'characteristics': "Segment characteristics based on purchasing habits.",
        'recommendation': "Deploy standard retention campaigns."
    })

    return {
        'predicted_cluster': predicted_cluster_id,
        'segment_name': cluster_profile.get('segment_name'),
        'badge_color': cluster_profile.get('badge_color', '#3b82f6'),
        'archetype': cluster_profile.get('archetype'),
        'characteristics': cluster_profile.get('characteristics'),
        'recommendation': cluster_profile.get('recommendation'),
        'input_values': parsed_features,
        'distance_to_centroid': distance_to_cluster,
        'coordinates': {
            'income': scatter_x,
            'spending': scatter_y,
            'pca': pca_coords
        }
    }
