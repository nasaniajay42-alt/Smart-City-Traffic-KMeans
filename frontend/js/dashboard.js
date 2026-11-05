/**
 * Dashboard Logic for index.html
 */

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const summaryData = await App.getSummary();
    const dsSummary = summaryData.dataset_summary || {};
    const kpi = dsSummary.kpi || {};
    const model = summaryData.model_info || {};

    // 1. Update KPI Cards
    const totalCustomersEl = document.getElementById('kpi-total-customers');
    const nClustersEl = document.getElementById('kpi-n-clusters');
    const avgIncomeEl = document.getElementById('kpi-avg-income');
    const avgSpendingEl = document.getElementById('kpi-avg-spending');
    const silScoreEl = document.getElementById('kpi-sil-score');

    if (totalCustomersEl) totalCustomersEl.textContent = dsSummary.total_records || '0';
    if (nClustersEl) nClustersEl.textContent = model.n_clusters || 'N/A';
    if (avgIncomeEl) avgIncomeEl.textContent = `$${kpi.avg_income || 0}k`;
    if (avgSpendingEl) avgSpendingEl.textContent = `${kpi.avg_spending || 0}/100`;
    if (silScoreEl) silScoreEl.textContent = model.silhouette_score !== undefined ? model.silhouette_score : 'N/A';

    // Model Status Details
    const modelStatusEl = document.getElementById('model-status-text');
    const modelInertiaEl = document.getElementById('model-inertia-text');
    const modelFeaturesEl = document.getElementById('model-features-text');
    
    if (modelStatusEl) {
      if (model.trained) {
        modelStatusEl.innerHTML = `<span class="badge bg-success">Trained & Active</span>`;
      } else {
        modelStatusEl.innerHTML = `<span class="badge bg-warning text-dark">Untrained</span>`;
      }
    }
    if (modelInertiaEl) modelInertiaEl.textContent = model.inertia || 'N/A';
    if (modelFeaturesEl && model.features) {
      modelFeaturesEl.textContent = model.features.join(', ');
    }

    // 2. Fetch Clusters and render dashboard preview charts
    const clusterData = await App.getClusters();
    if (clusterData && clusterData.points) {
      // Cluster size distribution
      const counts = {};
      const labels = [];
      (clusterData.profiles || []).forEach(p => {
        labels.push(p.segment_name);
        counts[p.cluster_id] = p.count;
      });

      const clusterCounts = Object.keys(counts).sort((a,b) => a-b).map(k => counts[k]);
      Charts.renderDoughnut('dashboard-cluster-doughnut', labels, clusterCounts);

      // Mini Scatter Plot
      Charts.renderClusterScatter(
        'dashboard-cluster-scatter',
        clusterData.points,
        clusterData.centroids,
        false,
        clusterData.profiles
      );
    }

  } catch (err) {
    console.error('Error loading dashboard:', err);
    App.showAlert(`Could not load dashboard data: ${err.message}`, 'danger');
  }
});
