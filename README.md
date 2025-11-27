# Azure ML Engineer prep – summary

## 1. Role & goals

**Target role:** Machine Learning Engineer / Data Solutions Architect (Azure-focused)

**My goals with this 2-week prep:**

- Demonstrate hands-on ability to train, deploy, and operate ML models on **Azure Machine Learning**.
- Show I can build and orchestrate **data pipelines** using **Azure Data Factory / Synapse**.
- Speak clearly about **MLOps, governance, and risk-oriented use cases** (e.g., delinquency / credit risk) using the **CATER** framework.

---

## 2. Tech stack snapshot

**Languages & libraries**

- Python (pandas, scikit-learn, numpy, matplotlib)
- SQL (for analytics / feature engineering)
- Optional: MLflow / other experiment tools (if used)

**Azure services used**

- **Storage & data**
  - Azure Blob Storage / Data Lake Storage
  - Azure SQL DB / Synapse (if used)

- **ML & compute**
  - Azure Machine Learning workspace
  - Azure ML compute clusters
  - Azure ML jobs (training, batch scoring)
  - Azure ML model registry
  - Azure ML online and/or batch endpoints

- **Data pipelines**
  - Azure Data Factory or Synapse pipelines

- **Ops & tooling**
  - GitHub + GitHub Actions (or Azure DevOps)
  - Azure Monitor / logs (conceptually, or hands-on if used)
  - Power BI or simple reporting tools for model outputs

---

## 3. Project A – Delinquency-style classifier on Azure ML

### 3.1 Problem statement

- **Business-style goal:**  
  (Example) Predict whether a customer is likely to become delinquent / default in the next N days so that risk/collections teams can prioritize outreach.

- **Dataset used:**  
  - Name / source:
  - Target variable:
  - Key features:

### 3.2 Approach

- Data preprocessing:
  - Handling missing values:
  - Encoding categorical variables:
  - Train/test split:

- Model(s) tried:
  - Baseline (e.g., logistic regression)
  - Tree-based (e.g., random forest, XGBoost, etc.)
  - Final choice & why:

- Metrics:
  - Primary metric(s) (AUC, accuracy, precision/recall, etc.):
  - Business-relevant interpretation (e.g., “lift in high-risk detection,” etc.):

### 3.3 Azure implementation

- **Data storage:** Blob / Data Lake containers used and folder structure (raw, curated, etc.).
- **Training:**
  - Training script: `train.py`
  - How it’s submitted as an Azure ML job
  - Metrics and artifacts logged

- **Model registration:**
  - How model is registered in Azure ML model registry
  - Versioning approach

- **Deployment:**
  - Online endpoint setup (managed endpoint)
  - Scoring script: `score.py`
  - Sample request/response format

- **Client:**
  - Python client `client.py` to send JSON and get predictions.

### 3.4 Key talking points (for interviews)

- How this maps to a real credit risk / delinquency use case.
- Tradeoffs made (model complexity vs interpretability, latency vs cost).
- How monitoring and retraining would work in a production version.

---

## 4. Project B – Batch scoring & data pipeline on Azure

### 4.1 Problem statement

- **Goal:**  
  (Example) Run nightly batch scoring of accounts to update risk scores and feed dashboards / workflows.

- **Inputs / outputs:**
  - Input: daily snapshot CSVs in `raw/` storage.
  - Output: scored files in `curated/` or a database table.

### 4.2 Pipeline design

- **Data Factory / Synapse pipeline:**
  - Ingestion / copy activities:
  - Scoring activity (Python script or Azure ML batch job):
  - Output write step:
  - Scheduling (manual / daily schedule):

- **Scoring script:**
  - File name (e.g., `batch_score.py`)
  - How it loads the model and data:
  - Output schema (e.g., original columns + `prediction`, `score`).

### 4.3 Monitoring & reporting

- Operational logs:
  - What gets logged each run (row counts, timestamps, min/max scores, etc.).
  - Where logs are stored (CSV, table, etc.).

- Reporting:
  - Simple chart / report created:
    - e.g., distribution of risk scores over time, counts by risk bucket.

- Future improvements:
  - Hooking into Azure Monitor / Log Analytics.
  - Adding drift detection and automatic alerts.

---

## 5. Architecture overview

### 5.1 High-level architecture (bullet view)

- **Sources:** core systems → flat files / APIs
- **Ingestion:** Azure Data Factory / Synapse pipelines
- **Storage layers:** raw → staged/curated in Blob/Data Lake/Synapse
- **ML:**
  - Training on Azure ML (jobs, registry)
  - Online endpoint for real-time scoring
  - Batch jobs for nightly scoring
- **Consumption:**
  - Applications / APIs using online endpoint
  - Dashboards (Power BI) using batch outputs
- **Monitoring & governance:**
  - Azure Monitor / logs for operational metrics
  - RBAC, Key Vault, and basic governance practices

### 5.2 (Optional) Diagram

> Include or link a simple architecture diagram (Mermaid, PNG, etc.) that shows the full flow.

---

## 6. Top 10 Azure concepts I’ll reference in interviews

1. Azure Blob Storage / Data Lake
2. Azure Machine Learning workspace
3. Azure ML jobs (training & batch)
4. Azure ML model registry
5. Azure ML online endpoints
6. Azure Data Factory / Synapse pipelines
7. Azure SQL / Synapse Analytics
8. Azure Monitor / Log Analytics (for monitoring)
9. Azure Key Vault (secrets) and RBAC (access control)
10. CI/CD with GitHub Actions or Azure DevOps for ML code and deployments

---

## 7. CATER cheat sheet for this work

- **C – Context & constraints:**  
  Business problem (risk/delinquency), success metrics, users, latency & regulatory constraints.

- **A – Architecture & data:**  
  Azure storage, pipelines, ML workspace, how data flows end-to-end.

- **T – Tradeoffs & technology choices:**  
  Online vs batch, model choice, Azure service selection (ADF vs Synapse vs Functions, etc.).

- **E – Execution & MLOps:**  
  Training scripts, jobs, endpoints, CI/CD, monitoring, retraining.

- **R – Risks, reliability & results:**  
  Failure modes, governance, security (Key Vault, RBAC), business KPIs and review cadence.