"""
Flask Application for Customer Segmentation Web Project.
Provides RESTful APIs for dataset management, exploratory data analysis,
K-Means model training, cluster visualization, real-time prediction, and report exports.
"""

import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, send_from_directory

from backend.data_loader import (
    load_dataset, validate_dataset, get_dataset_summary,
    normalize_column_names, STANDARD_COLUMNS, DEFAULT_DATASET_PATH
)
from backend.preprocessing import (
    prepare_features, scale_features, apply_pca_reduction,
    ALL_NUMERICAL_FEATURES, DEFAULT_FEATURES
)
from backend.clustering import (
    evaluate_k_range, train_kmeans_model, save_model_bundle,
    load_model_bundle, MODEL_FILE_PATH
)
from backend.evaluation import compute_cluster_statistics, get_cluster_comparison_matrix
from backend.prediction import predict_customer_segment
from backend.report_generator import generate_pdf_report

# Initialize Flask app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

# Global state for active dataset
CURRENT_DATASET_PATH = DEFAULT_DATASET_PATH


def ensure_initial_model_trained():
    """Ensure a baseline model is trained on startup so dashboard KPIs are immediately live."""
    try:
        if not os.path.exists(MODEL_FILE_PATH):
            df = load_dataset(CURRENT_DATASET_PATH)
            # Default features: Annual Income and Spending Score
            features = [STANDARD_COLUMNS['annual_income'], STANDARD_COLUMNS['spending_score']]
            if STANDARD_COLUMNS['age'] in df.columns:
                features = [STANDARD_COLUMNS['age'], STANDARD_COLUMNS['annual_income'], STANDARD_COLUMNS['spending_score']]
            
            X, active_features = prepare_features(df, features)
            X_scaled, scaler = scale_features(X)
            X_pca, pca_model, _ = apply_pca_reduction(X_scaled, n_components=2)
            
            # Default K = 5 (canonical for Mall Customers)
            result = train_kmeans_model(
                X_raw=X,
                X_scaled=X_scaled,
                feature_names=active_features,
                n_clusters=5,
                random_state=42,
                pca_model=pca_model,
                X_pca=X_pca
            )
            # Add scaler to bundle and save
            bundle = result['model_bundle']
            bundle['scaler'] = scaler
            bundle['X_pca'] = X_pca
            bundle['cluster_labels'] = result['cluster_labels']
            save_model_bundle(bundle)
            print("Baseline K-Means model trained and persisted successfully.")
    except Exception as e:
        print(f"Warning during initial model training: {e}")


# Run initial training if needed
ensure_initial_model_trained()


# -------------------------------------------------------------
# Frontend Static Page Routes
# -------------------------------------------------------------
@app.route('/')
def index_page():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/dataset')
def dataset_page():
    return send_from_directory(FRONTEND_DIR, 'dataset.html')

@app.route('/analysis')
def analysis_page():
    return send_from_directory(FRONTEND_DIR, 'analysis.html')

@app.route('/training')
def training_page():
    return send_from_directory(FRONTEND_DIR, 'training.html')

@app.route('/clusters')
def clusters_page():
    return send_from_directory(FRONTEND_DIR, 'clusters.html')

@app.route('/prediction')
def prediction_page():
    return send_from_directory(FRONTEND_DIR, 'prediction.html')

@app.route('/reports')
def reports_page():
    return send_from_directory(FRONTEND_DIR, 'reports.html')


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route('/api/dataset', methods=['GET'])
def get_dataset_api():
    """Returns dataset records, schema, and basic stats."""
    try:
        df = load_dataset(CURRENT_DATASET_PATH)
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        total_rows = len(df)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_df = df.iloc[start_idx:end_idx]
        
        return jsonify({
            'status': 'success',
            'total_rows': total_rows,
            'page': page,
            'page_size': page_size,
            'columns': df.columns.tolist(),
            'data': paginated_df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_dataset_api():
    """Upload custom CSV dataset with validation."""
    global CURRENT_DATASET_PATH
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part in request.'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'status': 'error', 'message': 'Only CSV files are supported.'}), 400

        # Read into DataFrame to validate
        df_upload = pd.read_csv(file)
        is_valid, msg = validate_dataset(df_upload)
        if not is_valid:
            return jsonify({'status': 'error', 'message': msg}), 400

        # Save to datasets directory
        saved_filename = f"uploaded_{int(pd.Timestamp.now().timestamp())}.csv"
        saved_path = os.path.join(DATASETS_DIR, saved_filename)
        df_upload.to_csv(saved_path, index=False)
        CURRENT_DATASET_PATH = saved_path

        summary = get_dataset_summary(df_upload)
        return jsonify({
            'status': 'success',
            'message': 'Dataset uploaded and validated successfully.',
            'summary': summary
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Error parsing CSV: {str(e)}"}), 500


@app.route('/api/summary', methods=['GET'])
def get_summary_api():
    """Returns dataset summary metrics and active model KPIs."""
    try:
        df = load_dataset(CURRENT_DATASET_PATH)
        summary = get_dataset_summary(df)
        
        bundle = load_model_bundle()
        model_info = {}
        if bundle:
            model_info = {
                'trained': True,
                'n_clusters': bundle.get('n_clusters'),
                'silhouette_score': bundle.get('silhouette_score'),
                'inertia': bundle.get('inertia'),
                'davies_bouldin_score': bundle.get('davies_bouldin_score'),
                'calinski_harabasz_score': bundle.get('calinski_harabasz_score'),
                'features': bundle.get('feature_names'),
                'cluster_profiles': bundle.get('cluster_profiles')
            }
        else:
            model_info = {'trained': False}

        return jsonify({
            'status': 'success',
            'dataset_summary': summary,
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/eda', methods=['GET'])
def get_eda_api():
    """Computes distribution histograms, scatter points, and correlation matrix."""
    try:
        df = load_dataset(CURRENT_DATASET_PATH)
        
        age_col = STANDARD_COLUMNS['age']
        income_col = STANDARD_COLUMNS['annual_income']
        spending_col = STANDARD_COLUMNS['spending_score']
        gender_col = STANDARD_COLUMNS['gender']

        def get_hist(series, bins=10):
            if series is None or series.empty or not pd.api.types.is_numeric_dtype(series):
                return {'bins': [], 'counts': []}
            counts, bin_edges = np.histogram(series.dropna(), bins=bins)
            labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(counts))]
            return {'labels': labels, 'counts': counts.tolist()}

        age_dist = get_hist(df[age_col], 10) if age_col in df.columns else None
        income_dist = get_hist(df[income_col], 10) if income_col in df.columns else None
        spending_dist = get_hist(df[spending_col], 10) if spending_col in df.columns else None

        gender_counts = {}
        if gender_col in df.columns:
            gender_counts = df[gender_col].value_counts().to_dict()

        # Scatter points (Income vs Spending)
        scatter_points = []
        for _, row in df.iterrows():
            pt = {
                'id': int(row.get(STANDARD_COLUMNS['customer_id'], _ + 1)),
                'income': float(row[income_col]) if income_col in df.columns else 0.0,
                'spending': float(row[spending_col]) if spending_col in df.columns else 0.0,
                'age': float(row[age_col]) if age_col in df.columns else None,
                'gender': str(row[gender_col]) if gender_col in df.columns else None
            }
            scatter_points.append(pt)

        # Correlation Matrix
        num_cols = [c for c in [age_col, income_col, spending_col] if c in df.columns]
        corr_matrix = {}
        if len(num_cols) >= 2:
            corr_df = df[num_cols].corr().round(3)
            corr_matrix = {
                'columns': num_cols,
                'matrix': corr_df.values.tolist()
            }

        return jsonify({
            'status': 'success',
            'distributions': {
                'age': age_dist,
                'income': income_dist,
                'spending': spending_dist,
                'gender': gender_counts
            },
            'scatter': scatter_points,
            'correlation': corr_matrix
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/elbow', methods=['GET'])
def get_elbow_api():
    """Computes Inertia and Silhouette scores across K=2..10 to recommend optimal K."""
    try:
        df = load_dataset(CURRENT_DATASET_PATH)
        features_param = request.args.get('features')
        
        feature_names = None
        if features_param:
            feature_names = [f.strip() for f in features_param.split(',')]
            
        X, active_features = prepare_features(df, feature_names)
        X_scaled, _ = scale_features(X)

        elbow_results = evaluate_k_range(X_scaled, k_min=2, k_max=10)
        return jsonify({
            'status': 'success',
            'features': active_features,
            'results': elbow_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_model_api():
    """Trains K-Means model with chosen K and features."""
    try:
        data = request.get_json() or {}
        n_clusters = int(data.get('k', 5))
        features_req = data.get('features')

        if n_clusters < 2 or n_clusters > 15:
            return jsonify({'status': 'error', 'message': 'K must be between 2 and 15.'}), 400

        df = load_dataset(CURRENT_DATASET_PATH)
        X, active_features = prepare_features(df, features_req)
        X_scaled, scaler = scale_features(X)

        # PCA transformation
        X_pca, pca_model, explained_variance = apply_pca_reduction(X_scaled, n_components=2)

        # Train model
        train_result = train_kmeans_model(
            X_raw=X,
            X_scaled=X_scaled,
            feature_names=active_features,
            n_clusters=n_clusters,
            random_state=42,
            pca_model=pca_model,
            X_pca=X_pca
        )

        bundle = train_result['model_bundle']
        bundle['scaler'] = scaler
        bundle['X_pca'] = X_pca
        bundle['cluster_labels'] = train_result['cluster_labels']
        save_model_bundle(bundle)

        return jsonify({
            'status': 'success',
            'message': f"K-Means model trained successfully with K={n_clusters}.",
            'n_clusters': n_clusters,
            'features': active_features,
            'inertia': train_result['inertia'],
            'silhouette_score': train_result['silhouette_score'],
            'davies_bouldin_score': train_result['davies_bouldin_score'],
            'calinski_harabasz_score': train_result['calinski_harabasz_score'],
            'explained_variance': explained_variance,
            'cluster_profiles': train_result['cluster_profiles'],
            'cluster_summaries': train_result['cluster_summaries']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/clusters', methods=['GET'])
def get_clusters_api():
    """Returns clustered dataset points, PCA coordinates, and cluster centroids."""
    try:
        bundle = load_model_bundle()
        if not bundle:
            return jsonify({'status': 'error', 'message': 'No trained model found.'}), 404

        df = load_dataset(CURRENT_DATASET_PATH)
        feature_names = bundle['feature_names']
        labels = bundle['cluster_labels']
        n_clusters = bundle['n_clusters']
        profiles = bundle['cluster_profiles']
        pca_model = bundle.get('pca_model')
        scaler = bundle['scaler']
        model = bundle['model']

        # Cluster color mapping
        palette = [
            '#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
            '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#6366f1'
        ]
        
        # Centroids unscaled
        unscaled_centroids = scaler.inverse_transform(model.cluster_centers_)
        centroids_list = []
        for idx in range(n_clusters):
            c_dict = {feature_names[f]: round(float(unscaled_centroids[idx][f]), 2) for f in range(len(feature_names))}
            c_dict['cluster_id'] = idx
            
            # PCA projection of centroid
            if pca_model is not None:
                pca_pt = pca_model.transform(model.cluster_centers_[idx:idx+1])[0]
                c_dict['pca_x'] = round(float(pca_pt[0]), 3)
                c_dict['pca_y'] = round(float(pca_pt[1]), 3)
            centroids_list.append(c_dict)

        # Assemble points
        X_pca = bundle.get('X_pca')
        points = []
        for i, row in df.iterrows():
            cid = int(labels[i])
            pt = {
                'id': int(row.get(STANDARD_COLUMNS['customer_id'], i + 1)),
                'cluster': cid,
                'color': palette[cid % len(palette)],
                'age': float(row[STANDARD_COLUMNS['age']]) if STANDARD_COLUMNS['age'] in df.columns else None,
                'income': float(row[STANDARD_COLUMNS['annual_income']]) if STANDARD_COLUMNS['annual_income'] in df.columns else None,
                'spending': float(row[STANDARD_COLUMNS['spending_score']]) if STANDARD_COLUMNS['spending_score'] in df.columns else None,
                'gender': str(row[STANDARD_COLUMNS['gender']]) if STANDARD_COLUMNS['gender'] in df.columns else None,
            }
            if X_pca is not None and i < len(X_pca):
                pt['pca_x'] = round(float(X_pca[i][0]), 3)
                pt['pca_y'] = round(float(X_pca[i][1]), 3)
            points.append(pt)

        return jsonify({
            'status': 'success',
            'n_clusters': n_clusters,
            'features': feature_names,
            'profiles': profiles,
            'centroids': centroids_list,
            'points': points
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/cluster-summary', methods=['GET'])
def get_cluster_summary_api():
    """Returns cluster profiles, statistical distributions, and comparative matrices."""
    try:
        bundle = load_model_bundle()
        if not bundle:
            return jsonify({'status': 'error', 'message': 'No trained model found.'}), 404

        df = load_dataset(CURRENT_DATASET_PATH)
        feature_names = bundle['feature_names']
        labels = bundle['cluster_labels']

        stats = compute_cluster_statistics(df, labels, feature_names)
        matrix = get_cluster_comparison_matrix(df, labels, feature_names)

        return jsonify({
            'status': 'success',
            'n_clusters': bundle['n_clusters'],
            'profiles': bundle['cluster_profiles'],
            'cluster_statistics': stats,
            'comparison_matrix': matrix
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """Predicts customer segment for new input features."""
    try:
        payload = request.get_json() or {}
        prediction_result = predict_customer_segment(payload)
        return jsonify({
            'status': 'success',
            'prediction': prediction_result
        })
    except ValueError as ve:
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/export/csv', methods=['GET'])
def export_csv_api():
    """Exports dataset with assigned cluster IDs and segment names."""
    try:
        bundle = load_model_bundle()
        if not bundle:
            return jsonify({'status': 'error', 'message': 'Train a model first.'}), 400

        df = load_dataset(CURRENT_DATASET_PATH)
        labels = bundle['cluster_labels']
        profiles = {p['cluster_id']: p['segment_name'] for p in bundle['cluster_profiles']}

        df_export = df.copy()
        df_export['Cluster_ID'] = labels
        df_export['Segment_Name'] = [profiles.get(cid, f"Cluster {cid}") for cid in labels]

        output = io.StringIO()
        df_export.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='clustered_customers.csv'
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/export/report', methods=['GET'])
def export_report_api():
    """Generates and downloads the comprehensive academic PDF report."""
    try:
        bundle = load_model_bundle()
        if not bundle:
            return jsonify({'status': 'error', 'message': 'Train a model first.'}), 400

        df = load_dataset(CURRENT_DATASET_PATH)
        summary = get_dataset_summary(df)

        pdf_bytes = generate_pdf_report(summary, bundle)

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='Customer_Segmentation_ML_Report.pdf'
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("Starting Customer Segmentation K-Means Server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
