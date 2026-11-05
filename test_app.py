"""
Comprehensive Test Suite for Customer Segmentation Flask App.
Tests every API endpoint, model training, prediction, CSV export, and PDF generation.
"""

import sys
import unittest
import json
import io

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from app import app, MODEL_FILE_PATH


class CustomerSegmentationTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_static_routes(self):
        """Test all frontend page routes return HTTP 200."""
        routes = ['/', '/dataset', '/analysis', '/training', '/clusters', '/prediction', '/reports']
        for route in routes:
            response = self.app.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} failed with {response.status_code}")
            self.assertIn(b'CustomerAI', response.data)
        print("[PASS] All 7 frontend static routes responded with 200 OK.")

    def test_api_summary(self):
        """Test GET /api/summary."""
        response = self.app.get('/api/summary')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('dataset_summary', data)
        self.assertEqual(data['dataset_summary']['total_records'], 200)
        self.assertEqual(data['dataset_summary']['kpi']['avg_income'], 60.56)
        print("[PASS] GET /api/summary verified successfully.")

    def test_api_dataset(self):
        """Test GET /api/dataset with pagination."""
        response = self.app.get('/api/dataset?page=1&page_size=25')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['data']), 25)
        self.assertEqual(data['total_rows'], 200)
        print("[PASS] GET /api/dataset paginated response verified successfully.")

    def test_api_eda(self):
        """Test GET /api/eda."""
        response = self.app.get('/api/eda')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('distributions', data)
        self.assertIn('income', data['distributions'])
        self.assertIn('scatter', data)
        self.assertEqual(len(data['scatter']), 200)
        self.assertIn('correlation', data)
        print("[PASS] GET /api/eda distributions and correlation verified successfully.")

    def test_api_elbow(self):
        """Test GET /api/elbow."""
        response = self.app.get('/api/elbow')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('results', data)
        results = data['results']
        self.assertEqual(len(results['k_values']), 9)  # K=2..10
        self.assertGreater(results['best_silhouette'], 0.3)
        print(f"[PASS] GET /api/elbow verified: Recommended K={results['recommended_k']}, Silhouette={results['best_silhouette']}.")

    def test_api_train(self):
        """Test POST /api/train with selected K."""
        payload = {
            'k': 5,
            'features': ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        }
        response = self.app.post('/api/train', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['n_clusters'], 5)
        self.assertIn('inertia', data)
        self.assertIn('silhouette_score', data)
        self.assertEqual(len(data['cluster_profiles']), 5)
        print(f"[PASS] POST /api/train verified: Inertia={data['inertia']}, Silhouette={data['silhouette_score']}.")

    def test_api_clusters(self):
        """Test GET /api/clusters."""
        response = self.app.get('/api/clusters')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['points']), 200)
        self.assertEqual(len(data['centroids']), 5)
        self.assertIn('pca_x', data['points'][0])
        print("[PASS] GET /api/clusters verified: 200 points with PCA coordinates and centroids.")

    def test_api_cluster_summary(self):
        """Test GET /api/cluster-summary."""
        response = self.app.get('/api/cluster-summary')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['cluster_statistics']), 5)
        self.assertIn('comparison_matrix', data)
        print("[PASS] GET /api/cluster-summary statistical distributions verified.")

    def test_api_predict(self):
        """Test POST /api/predict for valid customer inputs."""
        payload = {
            'age': 29,
            'annual_income': 95,
            'spending_score': 88
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        pred = data['prediction']
        self.assertIn('predicted_cluster', pred)
        self.assertIn('segment_name', pred)
        self.assertIn('recommendation', pred)
        self.assertIn('coordinates', pred)
        print(f"[PASS] POST /api/predict verified: Predicted {pred['segment_name']} (Cluster {pred['predicted_cluster']}).")

    def test_api_predict_validation_errors(self):
        """Test POST /api/predict input boundary error handling."""
        # Age out of range
        bad_payload = {'age': 150, 'annual_income': 50, 'spending_score': 50}
        response = self.app.post('/api/predict', data=json.dumps(bad_payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('Age', data['message'])
        print("[PASS] POST /api/predict validation error handling verified.")

    def test_api_export_csv(self):
        """Test GET /api/export/csv."""
        response = self.app.get('/api/export/csv')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Content-Type'], 'text/csv; charset=utf-8')
        content = response.data.decode('utf-8')
        self.assertIn('Cluster_ID', content)
        self.assertIn('Segment_Name', content)
        lines = content.strip().split('\n')
        self.assertEqual(len(lines), 201)  # 200 rows + 1 header
        print("[PASS] GET /api/export/csv download verified (200 records + headers).")

    def test_api_export_report(self):
        """Test GET /api/export/report."""
        response = self.app.get('/api/export/report')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Content-Type'], 'application/pdf')
        self.assertTrue(response.data.startswith(b'%PDF'))
        self.assertGreater(len(response.data), 5000)
        print(f"[PASS] GET /api/export/report PDF generation verified ({len(response.data)} bytes).")

    def test_api_upload_validation(self):
        """Test POST /api/upload validation with invalid CSV."""
        invalid_csv = "col1,col2\n1,2\n3,4"
        data = {
            'file': (io.BytesIO(invalid_csv.encode('utf-8')), 'test.csv')
        }
        response = self.app.post('/api/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        res_data = json.loads(response.data)
        self.assertIn('Missing required columns', res_data['message'])
        print("[PASS] POST /api/upload missing column validation verified.")


if __name__ == '__main__':
    unittest.main()
