# Customer Purchasing Behaviour Segmentation Using K-Means Clustering
### An Interactive Academic Machine Learning & Decision-Support System

---

## 1. Project Overview
This project presents an academic, fully functional Machine Learning web application designed to segment retail customers into distinct behavioural groups based on their historical purchasing patterns and socio-demographic characteristics.

Built with **Python, Scikit-Learn, Flask, Bootstrap 5, and Chart.js**, this platform automates the entire unsupervised learning lifecycle—from exploratory data analysis and feature standardization (`StandardScaler`), to hyperparameter optimization via the **Elbow Method** and **Silhouette Analysis**, **PCA (Principal Component Analysis)** 2D projection, dynamic cluster profiling, real-time single-customer segment classification, and academic PDF/CSV report generation.

Unlike static prototypes or synthetic demonstrations, this application operates on actual customer records (Mall Customer Segmentation Dataset), computing real mathematical inertia, cohesion scores, and data-driven segment archetypes without hardcoded assumptions.

---

## 2. Problem Statement
In modern retail management, mass marketing campaigns treat diverse customer bases uniformly, resulting in inefficient capital allocation, poor conversion rates, and suboptimal customer engagement. Retailers hold substantial transaction data (such as customer age, annual income, and purchasing spending ratings) but struggle to translate raw observations into structured customer segments.

The challenge is to apply unsupervised learning techniques to discover natural groupings within continuous multi-dimensional space, accurately identify behavioral personas (e.g., High-Value Champions vs. Budget-Conscious Savers), and provide marketing managers with actionable retention strategies.

---

## 3. Project Objectives
1. **Unsupervised Clustering Pipeline:** Construct a real-world pipeline implementing **K-Means++** to eliminate centroid initialization instability.
2. **Mathematical Validation:** Formulate and automate optimal cluster identification using the **Elbow Method (Within-Cluster Sum of Squares - WCSS)** and the **Silhouette Coefficient**.
3. **Data Standardization:** Apply `StandardScaler` to prevent high-magnitude features (e.g., Annual Income in \$k) from disproportionately skewing Euclidean distance metrics over bounded features (e.g., Spending Score from 1–100).
4. **Dimensionality Reduction:** Integrate **Principal Component Analysis (PCA)** to project multi-feature spaces (Age + Income + Spending) into an interpretable 2D visual coordinate plane.
5. **Data-Driven Persona Profiling:** Derive descriptive segment titles (e.g., *High-Value Champions*, *Prudent Affluent*, *Trendsetting Spenders*, *Frugal Occasional Shoppers*) dynamically from unscaled centroid statistics.
6. **Real-Time Inference Engine:** Enable users to input arbitrary customer parameters and instantly classify them into their nearest cluster centroid with actionable marketing directives.
7. **Academic Reporting:** Generate publication-grade PDF summaries and exportable CSV datasets.

---

## 4. Dataset Information
The default dataset utilized is the canonical **Mall Customer Segmentation Dataset**:
* **Source:** Kaggle / UCI Machine Learning Repository
* **Total Instances:** 200 customer observations
* **Target Feature Dimensions:**
  * `CustomerID`: Unique categorical identifier for each customer.
  * `Gender`: Categorical binary variable (`Male` / `Female`).
  * `Age`: Continuous numerical attribute spanning 18 to 70 years ($\mu \approx 38.85$).
  * `Annual Income (k$)`: Continuous earnings ranging from \$15k to \$137k ($\mu \approx \$60.56\text{k}$).
  * `Spending Score (1-100)`: Continuous retail index reflecting purchasing frequency and ticket size ($\mu \approx 50.20$).

*Custom Dataset Support:* Users can upload any CSV dataset via the **Dataset Explorer** tab. The system validates required numerical columns, handles alternative column naming conventions (e.g., `annual_income`, `spending`), checks for missing values, and imputes null entries using column medians.

---

## 5. Mathematical Working of K-Means Clustering

### 5.1 The Optimization Objective (Inertia / WCSS)
Given an observation matrix $X = \{x_1, x_2, \dots, x_n\} \in \mathbb{R}^d$, K-Means partitions $n$ samples into $K$ disjoint subsets $S = \{S_1, S_2, \dots, S_K\}$ by minimizing the Within-Cluster Sum of Squares (Inertia $J$):

$$J = \sum_{j=1}^{K} \sum_{x \in S_j} ||x - \mu_j||^2$$

where $\mu_j = \frac{1}{|S_j|} \sum_{x \in S_j} x$ represents the geometric centroid of cluster $S_j$, and $||\cdot||$ denotes standard Euclidean distance:

$$d(x, \mu_j) = \sqrt{\sum_{i=1}^{d} (x_i - \mu_{j,i})^2}$$

### 5.2 Lloyd's Iterative Optimization Algorithm
The algorithm alternates iteratively through two steps until convergence:
1. **Assignment Step:** Assign each customer $x_i$ to its nearest centroid:
   $$c^{(i)} := \arg\min_{j \in \{1,\dots,K\}} ||x_i - \mu_j||^2$$
2. **Update Step:** Recalculate centroids based on current cluster members:
   $$\mu_j := \frac{\sum_{i=1}^{n} \mathbf{1}\{c^{(i)} = j\} x_i}{\sum_{i=1}^{n} \mathbf{1}\{c^{(i)} = j\}}$$
Convergence is reached when cluster assignments stabilize ($\Delta J < \epsilon$) or maximum iterations ($300$) are met.

### 5.3 K-Means++ Initialization
Standard random centroid selection can trap the model in poor local minima. **K-Means++** mitigates this through probabilistic distance-weighted seeding:
1. Centroid $\mu_1$ is chosen uniformly at random from $X$.
2. For each sample $x \in X$, compute $D(x) = \min_{j} ||x - \mu_j||$.
3. Select the next centroid with probability proportional to the squared distance:
   $$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$
4. Repeat until $K$ initial seeds are designated.

### 5.4 StandardScaler Normalization
Because Euclidean distance is variance-dependent:
$$z = \frac{x - \mu}{\sigma}$$
Feature standardization ensures that Annual Income (\$15k–\$137k) does not dominate Spending Score (1–100) or Age (18–70).

### 5.5 Optimal Cluster Determination (Elbow & Silhouette)
1. **Elbow Method:** Evaluates the rate of change of inertia $\frac{\partial^2 J}{\partial K^2}$ across $K \in [2, 10]$ to locate the inflection point.
2. **Silhouette Coefficient:** For each observation $i$, intra-cluster cohesion $a(i)$ and nearest-neighbor separation $b(i)$ are evaluated:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$
   The global Silhouette Score is the mean over all samples. The system identifies optimal $K$ by maximizing this score.

---

## 6. System Architecture

```
customer-segmentation/
│
├── app.py                     # Flask application entry point, routing, and REST API controllers
├── requirements.txt           # Python dependency specifications
├── test_app.py                # Comprehensive test suite covering all 13 endpoints & features
├── README.md                  # Comprehensive academic engineering documentation
│
├── backend/                   # Modular backend architecture
│   ├── data_loader.py         # CSV loading, column normalization, schema validation & summary stats
│   ├── preprocessing.py       # Imputation, StandardScaler normalization, and PCA dimensionality reduction
│   ├── clustering.py          # K-Means++ training, Elbow evaluation, dynamic persona synthesizer, joblib persistence
│   ├── evaluation.py          # Cluster-wise statistics, distribution matrices, metrics (DB, CH, Silhouette)
│   ├── prediction.py          # Customer inference engine with fitted scaler & distance mapping
│   └── report_generator.py    # Academic PDF document generator (ReportLab) & CSV exporters
│
├── datasets/
│   └── Mall_Customers.csv     # Canonical benchmark Mall Customer dataset
│
├── models/
│   └── kmeans_model.joblib    # Persisted model bundle (KMeans, StandardScaler, PCA, metadata)
│
├── frontend/                  # Responsive web presentation tier
│   ├── index.html             # Executive Dashboard (5 KPI cards, scatter overview, model status)
│   ├── dataset.html           # Dataset Explorer (CSV uploader, schema, statistics, paginated table)
│   ├── analysis.html          # Exploratory Data Analysis (distributions, scatter, Pearson heatmap)
│   ├── training.html          # Model Optimization (K slider, Elbow chart, Silhouette comparison, train triggers)
│   ├── clusters.html          # Cluster Personas & PCA 2D projections, marketing action plans
│   ├── prediction.html        # Live Customer Classifier (interactive sliders, real-time spatial plot)
│   ├── reports.html           # PDF / CSV / JSON Export Center & Academic Theoretical Overview
│   │
│   ├── css/
│   │   └── style.css          # Navy-blue & white theme design system (responsive sidebar, cards, pills)
│   └── js/
│       ├── app.js             # Shared API client, navigation highlighting, notification toasts
│       ├── charts.js          # Chart.js reusable constructors (Scatter, Doughnut, Histograms, Elbow)
│       └── dashboard.js       # Main dashboard KPI data-binding and chart initialization
│
└── outputs/                   # Export directory for generated reports & artifacts
```

---

## 7. Installation Instructions

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git
* Modern web browser (Chrome, Edge, Firefox, Safari)

### Step 1: Clone or Navigate to the Workspace
```bash
cd c:\Users\nasan\OneDrive\Desktop\kmeans
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```
*Required core packages:*
* `flask>=3.0.0`
* `pandas>=2.0.0`
* `numpy>=1.24.0`
* `scikit-learn>=1.3.0`
* `matplotlib>=3.8.0`
* `seaborn>=0.13.0`
* `reportlab>=4.0.0`

---

## 8. How to Run the Project

### Execute the Application
```bash
python app.py
```

### Access via Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Run the Verification Test Suite
```bash
python test_app.py
```
This automated suite tests all 13 REST API endpoints, model training, prediction accuracy, and export capabilities.

---

## 9. API Documentation

| HTTP Method | Endpoint | Description | Sample Query / Payload |
|---|---|---|---|
| `GET` | `/api/summary` | Returns dataset KPIs and active model convergence metrics. | None |
| `GET` | `/api/dataset` | Returns paginated customer records, column list, and counts. | `?page=1&page_size=50` |
| `POST` | `/api/upload` | Validates and loads a user-provided CSV customer dataset. | `multipart/form-data` (`file`) |
| `GET` | `/api/eda` | Returns histogram bins, 2D scatter coordinates, and correlation matrix. | None |
| `GET` | `/api/elbow` | Computes Inertia and Silhouette scores across $K \in [2, 10]$. | `?features=Age,Annual+Income+(k$),Spending+Score+(1-100)` |
| `POST` | `/api/train` | Standardizes features, fits K-Means++, and persists model. | `{"k": 5, "features": ["Age", "Annual Income (k$)", "Spending Score (1-100)"]}` |
| `GET` | `/api/clusters` | Returns 2D points, PCA coordinates, and cluster centroids. | None |
| `GET` | `/api/cluster-summary` | Returns statistical distributions and feature comparison matrices. | None |
| `POST` | `/api/predict` | Predicts cluster ID and persona for a new customer record. | `{"age": 28, "annual_income": 85, "spending_score": 90}` |
| `GET` | `/api/export/csv` | Downloads full dataset with appended Cluster ID and Segment Name. | None |
| `GET` | `/api/export/report` | Generates and streams academic PDF project report. | None |

---

## 10. Expected Customer Segments & Business Strategies

When clustering on the canonical benchmark features ($K = 5$), the empirical centroids produce the following personas:

1. **High-Value Champions (High Income, High Spending):**
   * *Profile:* Earning \$85k–\$137k with spending scores $> 70$.
   * *Business Strategy:* VIP loyalty tiers, concierge support, exclusive previews, private showroom access.
2. **Budget-Conscious Affluent (High Income, Low Spending):**
   * *Profile:* Earning \$70k–\$137k with conservative spending scores $< 40$.
   * *Business Strategy:* High-ROI promotions, premium durability messaging, calculated value propositions.
3. **Trendsetting Spenders (Low Income, High Spending):**
   * *Profile:* Earning \$15k–\$39k with spending scores $> 60$. Frequently younger demographics.
   * *Business Strategy:* Buy-Now-Pay-Later (BNPL) payment plans, limited drops, flash discounts, social media marketing.
4. **Frugal Occasional Shoppers (Low Income, Low Spending):**
   * *Profile:* Modest earnings (\$15k–\$39k) with spending scores $< 35$.
   * *Business Strategy:* Essential product bundles, seasonal clearance sales, coupon loyalty discounts.
5. **Steady Mainstream Core (Moderate Income, Moderate Spending):**
   * *Profile:* Average earnings (\$40k–\$65k) with balanced spending scores ($40–60$).
   * *Business Strategy:* Points accumulation programs, quarterly newsletters, cross-category recommendations.

---

## 11. Limitations
1. **Spherical Cluster Assumption:** K-Means implicitly assumes isotropic (spherical) clusters with similar variance. Non-spherical geometries (e.g., elongated manifold structures) may require DBSCAN or Gaussian Mixture Models (GMM).
2. **Sensitivity to Outliers:** Because centroids are calculated as geometric means, extreme income or spending outliers can shift cluster centers.
3. **Cross-Sectional Data:** The Mall Customer dataset represents a single point in time. It does not reflect longitudinal temporal dynamics (e.g., changes in customer spending across seasons).

---

## 12. Future Enhancements
* **RFM (Recency, Frequency, Monetary) Integration:** Connect to real e-commerce transactional databases to compute live RFM metrics.
* **Hierarchical & Density-Based Clustering:** Provide comparative models (Agglomerative Clustering, DBSCAN) alongside K-Means.
* **Automated Re-training Pipelines:** Set up cron tasks to re-evaluate cluster stability as new customer orders stream in.
* **LLM-Powered Marketing Copy Generation:** Connect to generative AI APIs to draft tailored marketing email copy directly inside the persona cards.
