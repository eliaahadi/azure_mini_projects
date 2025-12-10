# Azure ML engineer prep – summary

## 1. Role and goals

**Target role:** Machine learning engineer (Azure-focused, data + MLOps)

**My goals with this prep:**

- Show end-to-end ability to **design, train, deploy, and operate** ML models using patterns that map cleanly to **Azure Machine Learning**.
- Demonstrate comfort with **data pipelines** and **batch / online scoring** in a way that matches a risk / analytics use case (credit / delinquency style).
- Be able to clearly explain tradeoffs and architecture using my **CATER** framework.

---

## 2. Tech stack snapshot

**Languages and libs**

- Python (pandas, numpy, scikit-learn, joblib)
- MLflow (local, to simulate experiment tracking)
- FastAPI + uvicorn (local online scoring)
- pytest (basic tests)

**“Cloud” and infra (local + emulator)**

- Azurite (Azure Storage emulator) via Docker
- Azure Storage SDK (`azure-storage-blob`)
- Local filesystem “model registry”
- GitHub Actions (CI skeleton, optional)

**Azure mapping**

- Azurite Blob ⇢ Azure Blob Storage / Data Lake
- Local MLflow runs ⇢ Azure ML experiments/runs
- Local registry ⇢ Azure ML model registry
- FastAPI app ⇢ Azure ML managed online endpoint
- Batch scripts ⇢ Azure ML batch jobs + ADF/Synapse pipelines

---

## 3. Project A – Online risk-style classifier (Azure-style, local sim)

### 3.1 Problem

- **Business-style goal:**  
  Simulate a **delinquency / risk** classifier: predict whether an entity (e.g., customer / account) is “high risk” (binary classification).
- **Why:**  
  Mirrors what this ML engineer role would do for credit union / financial risk models: support proactive outreach, better risk-adjusted decisions.

### 3.2 Data

- **Source:** sklearn `breast_cancer` dataset, exported to `day1_breast_cancer.csv`.
- **Target:** `target` (0/1).
- **Features:** All non-target numeric columns (engineered / scaled in pipeline).

*(Replace with a real risk dataset later if desired.)*

### 3.3 Training pipeline

- **Script:** `train.py`
- **Flow:**
  1. Load data from `day1_breast_cancer.csv` if present, otherwise from sklearn.
  2. Train/test split (stratified, default 80/20).
  3. Pipeline: `StandardScaler` → `LogisticRegression(max_iter=1000)`.
  4. Compute core metrics: accuracy, ROC AUC, classification report.
  5. Save:
     - `artifacts/model.joblib`
     - `artifacts/metrics.json`
  6. “Register” the model into a **local registry** as a new version:
     - `local_registry/model_v001/`
     - `local_registry/model_v002/`  
       each containing `model.joblib`, `metrics.json`, `metadata.json`.

### 3.4 Local model registry (simulated)

- **Directory:** `local_registry/`
- **Versioning pattern:** `model_vNNN` directories.
- **Each version contains:**
  - `model.joblib` – full sklearn Pipeline (scaler + classifier).
  - `metrics.json` – accuracy, ROC AUC, train/test sizes, random_state.
  - `metadata.json` – model name, version, registration timestamp.

This simulates what Azure ML’s **model registry** gives you (name + version + metadata).

### 3.5 Online scoring (local “endpoint”)

- **Core scorer:** `score.py`
  - Loads **latest** model from `local_registry/`.
  - Accepts a list of JSON-like records (`[{feature_name: value, ...}]`).
  - Returns predictions + probabilities for each record.

- **FastAPI service:** `app_score_api.py`
  - Endpoint: `POST /predict`
  - Request:

    ```json
    {
      "records": [
        {
          "feature_1": 0.1,
          "feature_2": 2.3
        }
      ]
    }
    ```

  - Response:

    ```json
    {
      "results": [
        {
          "input": { "feature_1": 0.1, "feature_2": 2.3 },
          "prediction": 0,
          "probability": 0.12
        }
      ]
    }
    ```

- **Client:** `client.py`
  - Sends a sample payload to `http://localhost:8000/predict`.
  - Prints predictions so you can demo the flow end-to-end.

### 3.6 Azure mapping

- `train.py`  
  ⇢ Azure ML **command job** that logs metrics and saves artifacts.

- `local_registry/model_vNNN/`  
  ⇢ Azure ML **model registry** (named model + version).

- FastAPI (`app_score_api.py`)  
  ⇢ Azure ML **managed online endpoint** (with an equivalent `score.py` + environment).

- `client.py`  
  ⇢ Any downstream **service / application** calling the Azure endpoint.

---

## 4. Project B – Batch scoring and pipeline (Azure-style, local sim)

### 4.1 Problem

- **Goal:**  
  Simulate a **nightly batch scoring pipeline** that computes risk scores for a full population and writes them to curated storage for analytics and operations (e.g., daily delinquency / risk lists).

### 4.2 Batch scoring flow

- **Script:** `batch_score.py` (to be implemented / refined)
- **Intended behavior:**
  1. Load latest model from `local_registry`.
  2. Read `raw/new_data_YYYYMMDD.csv`.
  3. Apply model to each row.
  4. Write `curated/predictions_YYYYMMDD.csv` with:
     - Original features
     - `prediction`
     - `score` (probability).

- **Optional “pipeline driver”:** `pipeline_run_batch.py`
  - Accepts `--date YYYY-MM-DD` or similar.
  - Finds the right input file.
  - Calls `batch_score.py`.
  - Logs success/failure.

### 4.3 Monitoring and reporting

- **Logs:** `logs/batch_runs_log.csv`  
  Each run appends:
  - timestamp
  - input file
  - row count
  - min/max score
  - model version

- **Analysis script:** `analyze_predictions.py` (or notebook)
  - Reads `curated/*predictions*.csv`.
  - Outputs:
    - Risk bucket counts (e.g., low/med/high).
    - Simple trend lines over dates.

### 4.4 Azure mapping

- Local CSV folder structure (`raw/`, `curated/`)  
  ⇢ Azure **Blob Storage / Data Lake** containers and folders.

- `pipeline_run_batch.py` + `batch_score.py`  
  ⇢ **Azure Data Factory / Synapse** pipeline that:
    - Copies / stages data.
    - Triggers an **Azure ML batch job** or Python activity.
    - Writes predictions into a curated zone (Blob, Lake, or Synapse table).

- `logs/batch_runs_log.csv`  
  ⇢ Azure **Log Analytics / Azure Monitor**, or a logging table.

- `analyze_predictions.py`  
  ⇢ **Power BI** dashboards / reports wired to the curated prediction store.

---

## 5. Architecture overview

### 5.1 High-level architecture (local sim)

- **Sources**
  - Local CSVs (`day1_breast_cancer.csv`, `raw/new_data_*.csv`)
  - (In real life: core banking systems, CRM, transaction data, etc.)

- **Storage**
  - Local: Azurite (Blob) or filesystem `raw/` and `curated/`
  - Azure: Blob Storage / Data Lake

- **Training**
  - Local: `train.py` + MLflow for experiment tracking
  - Azure: Azure ML jobs (command) with experiment runs

- **Model registry**
  - Local: `local_registry/model_vNNN/`
  - Azure: Azure ML model registry (name + version)

- **Serving**
  - Local: FastAPI (`app_score_api.py`) running via uvicorn
  - Azure: Azure ML managed online endpoints

- **Batch**
  - Local: `batch_score.py` + `pipeline_run_batch.py`
  - Azure: Data Factory / Synapse pipelines + Azure ML batch endpoints/jobs

- **Consumption**
  - Applications / services calling online endpoint
  - Dashboards / reports reading curated predictions

- **Monitoring & governance**
  - Local: simple CSV logs + Python plots
  - Azure: Azure Monitor, Log Analytics, Key Vault, RBAC

### 5.2 (Optional) Mermaid diagram

You can embed a Mermaid diagram in `architecture_diagram.md`, for example:

```mermaid
flowchart LR
    A[Source systems / CSVs] --> B[Blob / Data Lake (raw)]
    B --> C[Data prep / ADF or notebooks]
    C --> D[Azure ML training job]
    D --> E[Azure ML model registry]
    E --> F[Managed online endpoint]
    E --> G[Batch scoring jobs]
    G --> H[Blob / Lake (curated predictions)]
    F --> I[Apps / APIs]
    H --> J[Power BI / reports]