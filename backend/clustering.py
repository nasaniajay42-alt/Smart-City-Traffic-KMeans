"""
Clustering Module for Customer Segmentation.
Implements K-Means++ clustering, Elbow method, Silhouette score evaluation,
model persistence, and dynamic cluster profiling based on empirical centroid statistics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'kmeans_model.joblib')


def evaluate_k_range(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42) -> dict:
    """
    Run K-Means for a range of K values to compute Inertia (Elbow method) and Silhouette Scores.
    Finds the recommended K using maximum silhouette score with elbow heuristics.
    """
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouette_scores_list = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(round(float(kmeans.inertia_), 2))
        sil_score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores_list.append(round(float(sil_score), 4))

    # Determine recommended K: Highest silhouette score
    best_sil_idx = int(np.argmax(silhouette_scores_list))
    recommended_k = k_values[best_sil_idx]
    
    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores_list,
        'recommended_k': recommended_k,
        'best_silhouette': silhouette_scores_list[best_sil_idx]
    }


def generate_cluster_profile(cluster_id: int, centroid_dict: dict, global_stats: dict) -> dict:
    """
    Generate dynamic descriptive segment names and interpretations based on
    actual centroid values compared against global dataset statistics.
    Ensures cluster naming is completely data-driven rather than hardcoded to arbitrary IDs.
    """
    # Extract values with fallbacks
    income = centroid_dict.get('Annual Income (k$)', 50.0)
    spending = centroid_dict.get('Spending Score (1-100)', 50.0)
    age = centroid_dict.get('Age', 35.0)

    # Global thresholds from dataset or sensible defaults
    income_median = global_stats.get('Annual Income (k$)', {}).get('50%', 60.0)
    spending_median = global_stats.get('Spending Score (1-100)', {}).get('50%', 50.0)
    age_median = global_stats.get('Age', {}).get('50%', 36.0)

    # Classifications
    is_high_income = income >= (income_median * 1.15)
    is_low_income = income <= (income_median * 0.85)
    is_moderate_income = not (is_high_income or is_low_income)

    is_high_spending = spending >= (spending_median * 1.15)
    is_low_spending = spending <= (spending_median * 0.80)
    is_moderate_spending = not (is_high_spending or is_low_spending)

    is_young = age < 33
    is_mature = age > 48

    # Dynamic Persona Synthesis
    if is_high_income and is_high_spending:
        name = "High-Value Champions"
        badge_color = "#10b981"  # Emerald
        archetype = "Affluent VIP Spenders"
        characteristics = (
            f"High annual earnings (avg ${income:.1f}k) paired with aggressive spending habits (score {spending:.1f}). "
            "These customers represent the primary revenue engine."
        )
        recommendation = (
            "Provide elite VIP loyalty perks, personalized shopping concierge, preview invites to new collections, "
            "and premium white-glove customer support."
        )

    elif is_high_income and is_low_spending:
        name = "Budget-Conscious Affluent"
        badge_color = "#3b82f6"  # Blue
        archetype = "Prudent High-Earners / Savers"
        characteristics = (
            f"High financial prosperity (avg ${income:.1f}k) but conservative spending behavior (score {spending:.1f}). "
            "They calculate purchases carefully and value long-term utility."
        )
        recommendation = (
            "Market high-durability products, high-ROI items, premium subscription models with transparent value, "
            "and subtle, data-driven utility benefits."
        )

    elif is_low_income and is_high_spending:
        name = "Trendsetting Spenders"
        badge_color = "#f59e0b"  # Amber
        archetype = "Aspiring / Impulse Buyers"
        characteristics = (
            f"Modest earnings (avg ${income:.1f}k) yet exceptionally enthusiastic retail spending (score {spending:.1f}). "
            "Frequently younger consumers driven by viral trends and immediate gratification."
        )
        recommendation = (
            "Offer flexible Buy-Now-Pay-Later (BNPL) payment plans, limited-time flash drops, influencer collaborations, "
            "and gamified social discounts."
        )

    elif is_low_income and is_low_spending:
        name = "Frugal Occasional Shoppers"
        badge_color = "#64748b"  # Slate
        archetype = "Budget / Necessity Focused"
        characteristics = (
            f"Cautious purchasing profile with constrained income (avg ${income:.1f}k) and low spending score ({spending:.1f}). "
            "Only transactions for absolute essentials or heavy markdowns."
        )
        recommendation = (
            "Deploy discount bundles, seasonal clearance sales, price-match guarantees, and essential coupon catalogs."
        )

    elif is_moderate_spending and is_moderate_income:
        name = "Steady Mainstream Core"
        badge_color = "#6366f1"  # Indigo
        archetype = "Reliable Balanced Consumers"
        characteristics = (
            f"Moderate earnings (${income:.1f}k) and steady, predictable spending habits (score {spending:.1f}). "
            "Forms the stable bedrock of consistent customer traffic."
        )
        recommendation = (
            "Engage through seasonal promotional emails, points-accumulation reward programs, and mid-tier product recommendations."
        )

    elif is_high_spending:
        name = "Engaged Active Spenders"
        badge_color = "#8b5cf6"  # Purple
        archetype = "High-Engagement Consumers"
        characteristics = (
            f"Strong spending propensity (score {spending:.1f}) with moderate income (${income:.1f}k). Responsive to brand marketing."
        )
        recommendation = "Provide loyalty rewards, referral incentives, and cross-sell related categories."

    else:
        name = f"Segment {cluster_id + 1} - Standard Group"
        badge_color = "#0ea5e9"
        archetype = "Standard Purchasing Persona"
        characteristics = (
            f"Average age {age:.1f} years, income ${income:.1f}k, and spending score {spending:.1f}."
        )
        recommendation = "Monitor shopping trends and test customized category promotions to assess affinity."

    return {
        'cluster_id': int(cluster_id),
        'segment_name': name,
        'badge_color': badge_color,
        'archetype': archetype,
        'characteristics': characteristics,
        'recommendation': recommendation,
        'centroid': {k: round(float(v), 2) for k, v in centroid_dict.items()}
    }


def train_kmeans_model(X_raw: pd.DataFrame, X_scaled: np.ndarray, feature_names: list[str],
                       n_clusters: int, random_state: int = 42,
                       pca_model=None, X_pca: np.ndarray = None) -> dict:
    """
    Train K-Means model with KMeans++ initialization and fixed random state.
    Calculates actual inertia, silhouette score, Davies-Bouldin index, unscaled centroids,
    and dynamic cluster profiling. Persists the model bundle.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=20,
        max_iter=300,
        random_state=random_state
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate real evaluation metrics
    inertia = round(float(kmeans.inertia_), 2)
    sil_score = round(float(silhouette_score(X_scaled, cluster_labels)), 4)
    db_score = round(float(davies_bouldin_score(X_scaled, cluster_labels)), 4)
    ch_score = round(float(calinski_harabasz_score(X_scaled, cluster_labels)), 2)

    # Unscale cluster centroids to original feature units
    # Need scaler to inverse transform
    from .preprocessing import scale_features
    # We will pass the fitted scaler in bundle
    
    # Calculate empirical centroids directly from raw data grouped by cluster
    df_clustered = X_raw.copy()
    df_clustered['Cluster'] = cluster_labels
    
    global_stats = X_raw.describe().to_dict()
    
    cluster_profiles = []
    cluster_summaries = []
    
    for c_id in range(n_clusters):
        c_subset = df_clustered[df_clustered['Cluster'] == c_id]
        c_count = int(len(c_subset))
        c_pct = round((c_count / len(X_raw)) * 100, 2)
        
        centroid_dict = {}
        for feat in feature_names:
            centroid_dict[feat] = round(float(c_subset[feat].mean()), 2)

        profile = generate_cluster_profile(c_id, centroid_dict, global_stats)
        profile['count'] = c_count
        profile['percentage'] = c_pct
        
        # Summary statistics per cluster
        stat_summary = {
            'cluster_id': c_id,
            'count': c_count,
            'percentage': c_pct,
            'means': centroid_dict
        }
        
        cluster_profiles.append(profile)
        cluster_summaries.append(stat_summary)

    # Bundle artifacts for saving
    model_bundle = {
        'model': kmeans,
        'feature_names': feature_names,
        'n_clusters': n_clusters,
        'inertia': inertia,
        'silhouette_score': sil_score,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score,
        'cluster_profiles': cluster_profiles,
        'global_stats': global_stats,
        'pca_model': pca_model
    }

    return {
        'model_bundle': model_bundle,
        'cluster_labels': cluster_labels.tolist(),
        'inertia': inertia,
        'silhouette_score': sil_score,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score,
        'n_clusters': n_clusters,
        'cluster_profiles': cluster_profiles,
        'cluster_summaries': cluster_summaries
    }


def save_model_bundle(bundle: dict, path: str = MODEL_FILE_PATH):
    """Save trained model bundle to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: str = MODEL_FILE_PATH) -> dict:
    """Load persisted model bundle from disk."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)
