/**
 * Customer Segmentation Application - Shared Frontend Logic & API Client
 */

const API_BASE = '/api';

const App = {
  // Format currency
  formatCurrency(val) {
    if (val === null || val === undefined) return 'N/A';
    return `$${Number(val).toLocaleString()}k`;
  },

  // Format decimal
  formatDecimal(val, decimals = 2) {
    if (val === null || val === undefined) return 'N/A';
    return Number(val).toFixed(decimals);
  },

  // Show status notification
  showAlert(message, type = 'info', containerId = 'alert-container') {
    const container = document.getElementById(containerId);
    if (!container) return;

    const alertEl = document.createElement('div');
    alertEl.className = `alert alert-${type} alert-dismissible fade show shadow-sm mb-3`;
    alertEl.role = 'alert';
    alertEl.innerHTML = `
      <div class="d-flex align-items-center gap-2">
        <i class="bi bi-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-triangle' : 'info-circle'}"></i>
        <div>${message}</div>
      </div>
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    container.innerHTML = '';
    container.appendChild(alertEl);

    setTimeout(() => {
      if (alertEl.parentNode) {
        alertEl.classList.remove('show');
        setTimeout(() => alertEl.remove(), 150);
      }
    }, 6000);
  },

  // API Client Methods
  async getSummary() {
    const res = await fetch(`${API_BASE}/summary`);
    if (!res.ok) throw new Error('Failed to load summary');
    return await res.json();
  },

  async getDataset(page = 1, pageSize = 50) {
    const res = await fetch(`${API_BASE}/dataset?page=${page}&page_size=${pageSize}`);
    if (!res.ok) throw new Error('Failed to load dataset');
    return await res.json();
  },

  async getEDA() {
    const res = await fetch(`${API_BASE}/eda`);
    if (!res.ok) throw new Error('Failed to load exploratory data');
    return await res.json();
  },

  async getElbow(features = '') {
    const url = features ? `${API_BASE}/elbow?features=${encodeURIComponent(features)}` : `${API_BASE}/elbow`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to compute elbow curve');
    return await res.json();
  },

  async trainModel(k = 5, features = null) {
    const payload = { k: Number(k) };
    if (features) payload.features = features;

    const res = await fetch(`${API_BASE}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok || data.status === 'error') {
      throw new Error(data.message || 'Model training failed');
    }
    return data;
  },

  async getClusters() {
    const res = await fetch(`${API_BASE}/clusters`);
    if (!res.ok) throw new Error('Failed to load cluster points');
    return await res.json();
  },

  async getClusterSummary() {
    const res = await fetch(`${API_BASE}/cluster-summary`);
    if (!res.ok) throw new Error('Failed to load cluster summary');
    return await res.json();
  },

  async predictCustomer(data) {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const result = await res.json();
    if (!res.ok || result.status === 'error') {
      throw new Error(result.message || 'Prediction failed');
    }
    return result.prediction;
  },

  async uploadCSV(file) {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!res.ok || data.status === 'error') {
      throw new Error(data.message || 'Upload failed');
    }
    return data;
  },

  // Highlight active link in sidebar
  initNavigation() {
    const currentPath = window.location.pathname.replace(/\/$/, '') || '/';
    const navLinks = document.querySelectorAll('.sidebar-nav .nav-link');
    
    navLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (href === currentPath || (currentPath === '/' && href === '/index.html') || (href !== '/' && currentPath.endsWith(href))) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    // Mobile sidebar toggle if hamburger exists
    const toggleBtn = document.getElementById('sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    if (toggleBtn && sidebar) {
      toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('show');
      });
    }
  }
};

document.addEventListener('DOMContentLoaded', () => {
  App.initNavigation();
});
