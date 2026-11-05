/**
 * Chart.js Integration Module for Customer Segmentation Dashboard.
 * Clean, modern visualization defaults with custom tooltips and academic styling.
 */

const Charts = {
  // Cohesive academic color palette
  palette: [
    '#2563eb', // Vibrant Blue
    '#10b981', // Emerald
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#8b5cf6', // Violet
    '#06b6d4', // Cyan
    '#ec4899', // Pink
    '#14b8a6', // Teal
    '#f97316', // Orange
    '#6366f1'  // Indigo
  ],

  // Instances tracker for proper teardown on re-renders
  instances: {},

  destroyChart(canvasId) {
    if (this.instances[canvasId]) {
      this.instances[canvasId].destroy();
      delete this.instances[canvasId];
    }
  },

  // 1. Histogram / Bar Chart
  renderHistogram(canvasId, labels, data, labelName = 'Count', color = '#2563eb') {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: labelName,
          data: data,
          backgroundColor: color,
          borderRadius: 6,
          barPercentage: 0.8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            padding: 10,
            cornerRadius: 6
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#64748b', font: { size: 11 } }
          },
          y: {
            grid: { color: '#f1f5f9' },
            ticks: { color: '#64748b', font: { size: 11 } },
            beginAtZero: true
          }
        }
      }
    });
  },

  // 2. Doughnut Chart (e.g. Gender, Cluster Share)
  renderDoughnut(canvasId, labels, data, customColors = null) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const colors = customColors || this.palette.slice(0, labels.length);

    this.instances[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: data,
          backgroundColor: colors,
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 12, padding: 14, color: '#475569', font: { size: 12 } }
          },
          tooltip: {
            backgroundColor: '#0f172a',
            padding: 10,
            cornerRadius: 6
          }
        },
        cutout: '65%'
      }
    });
  },

  // 3. Elbow Chart (Inertia vs K)
  renderElbowChart(canvasId, kValues, inertias, recommendedK = null) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const pointColors = kValues.map(k => k === recommendedK ? '#ef4444' : '#2563eb');
    const pointRadii = kValues.map(k => k === recommendedK ? 8 : 5);

    this.instances[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: kValues.map(k => `K=${k}`),
        datasets: [{
          label: 'Inertia (WCSS)',
          data: inertias,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.08)',
          fill: true,
          tension: 0.3,
          pointBackgroundColor: pointColors,
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: pointRadii,
          pointHoverRadius: 9
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            callbacks: {
              label: (context) => {
                const k = kValues[context.dataIndex];
                let text = ` Inertia: ${context.raw}`;
                if (k === recommendedK) text += ' (★ Recommended K)';
                return text;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#64748b', font: { weight: 600 } }
          },
          y: {
            grid: { color: '#f1f5f9' },
            ticks: { color: '#64748b' },
            title: { display: true, text: 'Within-Cluster Sum of Squares (Inertia)', color: '#475569', font: { size: 11.5, weight: '600' } }
          }
        }
      }
    });
  },

  // 4. Silhouette Score Chart
  renderSilhouetteChart(canvasId, kValues, silhouetteScores, recommendedK = null) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const barColors = kValues.map(k => k === recommendedK ? '#10b981' : '#93c5fd');

    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: kValues.map(k => `K=${k}`),
        datasets: [{
          label: 'Silhouette Score',
          data: silhouetteScores,
          backgroundColor: barColors,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            callbacks: {
              label: (context) => {
                const k = kValues[context.dataIndex];
                let text = ` Silhouette: ${context.raw}`;
                if (k === recommendedK) text += ' (Optimal Peak)';
                return text;
              }
            }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#64748b', font: { weight: 600 } } },
          y: {
            grid: { color: '#f1f5f9' },
            ticks: { color: '#64748b' },
            title: { display: true, text: 'Silhouette Coefficient', color: '#475569', font: { size: 11.5, weight: '600' } },
            beginAtZero: true
          }
        }
      }
    });
  },

  // 5. Clustered Customer Scatter Plot (Income vs Spending or PCA)
  renderClusterScatter(canvasId, points, centroids, usePCA = false, profiles = []) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const profileMap = {};
    profiles.forEach(p => {
      profileMap[p.cluster_id] = p.segment_name;
    });

    // Group points by cluster
    const clusterGroups = {};
    points.forEach(pt => {
      if (!clusterGroups[pt.cluster]) {
        clusterGroups[pt.cluster] = [];
      }
      const xVal = usePCA ? pt.pca_x : pt.income;
      const yVal = usePCA ? pt.pca_y : pt.spending;
      clusterGroups[pt.cluster].push({
        x: xVal,
        y: yVal,
        raw: pt
      });
    });

    const datasets = [];

    // Add customer data points per cluster
    Object.keys(clusterGroups).sort((a, b) => a - b).forEach(cId => {
      const c = Number(cId);
      const color = this.palette[c % this.palette.length];
      const name = profileMap[c] || `Cluster ${c}`;

      datasets.push({
        label: name,
        data: clusterGroups[c],
        backgroundColor: color,
        borderColor: '#ffffff',
        borderWidth: 1.5,
        pointRadius: 6,
        pointHoverRadius: 8,
        pointStyle: 'circle'
      });
    });

    // Add Centroid markers as separate dataset
    if (centroids && centroids.length > 0) {
      const centroidPoints = centroids.map(cent => {
        const xVal = usePCA ? cent.pca_x : cent['Annual Income (k$)'];
        const yVal = usePCA ? cent.pca_y : cent['Spending Score (1-100)'];
        return {
          x: xVal,
          y: yVal,
          cluster_id: cent.cluster_id
        };
      });

      datasets.push({
        label: 'Cluster Centroids (★)',
        data: centroidPoints,
        backgroundColor: '#0f172a',
        borderColor: '#ffffff',
        borderWidth: 2,
        pointRadius: 12,
        pointHoverRadius: 14,
        pointStyle: 'crossRot'
      });
    }

    const xLabel = usePCA ? 'Principal Component 1 (PCA 1)' : 'Annual Income (k$)';
    const yLabel = usePCA ? 'Principal Component 2 (PCA 2)' : 'Spending Score (1-100)';

    this.instances[canvasId] = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              boxWidth: 12,
              padding: 12,
              color: '#334155',
              font: { size: 12, weight: 600 }
            }
          },
          tooltip: {
            backgroundColor: '#0f172a',
            padding: 12,
            callbacks: {
              label: (context) => {
                const raw = context.raw;
                if (raw.cluster_id !== undefined) {
                  return ` Centroid C${raw.cluster_id}: (${raw.x}, ${raw.y})`;
                }
                const pt = raw.raw;
                if (!pt) return ` (${raw.x}, ${raw.y})`;
                return [
                  ` Customer #${pt.id} [${profileMap[pt.cluster] || 'C' + pt.cluster}]`,
                  ` Income: $${pt.income}k | Spending: ${pt.spending}/100`,
                  ` Age: ${pt.age} yrs | Gender: ${pt.gender}`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: xLabel, color: '#1e293b', font: { size: 13, weight: 'bold' } },
            grid: { color: '#e2e8f0' },
            ticks: { color: '#64748b' }
          },
          y: {
            title: { display: true, text: yLabel, color: '#1e293b', font: { size: 13, weight: 'bold' } },
            grid: { color: '#e2e8f0' },
            ticks: { color: '#64748b' }
          }
        }
      }
    });
  },

  // 6. Grouped Comparison Bar Chart
  renderComparisonBar(canvasId, comparisonMatrix) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    const clusters = comparisonMatrix.clusters;
    const features = comparisonMatrix.features;
    const datasets = [];

    const featureColors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];

    features.forEach((feat, idx) => {
      datasets.push({
        label: feat,
        data: comparisonMatrix.data[feat],
        backgroundColor: featureColors[idx % featureColors.length],
        borderRadius: 4
      });
    });

    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: clusters,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12, color: '#334155', font: { size: 12 } } },
          tooltip: { backgroundColor: '#0f172a', padding: 10 }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#64748b', font: { weight: '600' } } },
          y: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' }, beginAtZero: true }
        }
      }
    });
  }
};
