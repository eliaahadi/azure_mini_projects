# Mini project Q&A – azure_mini_projects

Use these prompts + bullets as talking points during interviews.

---

## Q1. Walk me through an end-to-end ML project you’ve built.

**Answer**

- **Context**
  - Built a small risk-style classifier to simulate a delinquency / credit-risk use case.
- **Data**
  - Binary dataset stored as CSV in Blob/Azurite (locally emulated).
- **Training**
  - `train_and_register_model`:
    - Loads the CSV, splits features/target.
    - Trains a `StandardScaler + LogisticRegression` pipeline.
    - Computes metrics (accuracy, ROC AUC).
    - Registers each model version in `local_registry/model_vNNN/` with metadata.
- **Online serving**
  - FastAPI `/predict` endpoint:
    - Loads the latest registered model at startup.
    - Accepts JSON with a list of feature dicts.
    - Returns prediction + probability per record.
- **Batch scoring**
  - `batch_score` + `run_daily_pipeline`:
    - Nightly job reads `raw/accounts_{date}.csv`.
    - Writes `curated/accounts_{date}_predictions.csv`.
    - Logs run info in `logs/batch_runs_log.csv`.
- **Azure mapping**
  - Same pattern maps to:
    - Azure Blob / Data Lake for storage.
    - Azure ML command jobs (training, batch).
    - Azure ML model registry.
    - Managed online endpoints for real-time scoring.
    - ADF/Synapse pipelines for orchestration.

---

## Q2. How do you move from notebooks to production-ready code?

**Answer**

- Start with an exploratory notebook to:
  - Understand the data.
  - Test a baseline model quickly.
- Gradually extract logic into reusable functions:
  - `train_and_register_model` for training + registration.
  - `batch_score` for batch scoring.
- Wrap functions into CLI-style scripts so they can be:
  - Run from the command line.
  - Parameterized for different datasets/environments.
- Add a FastAPI layer on top for HTTP-based scoring (`/predict`).
- Add basic tests and CI to catch regressions.
- In Azure:
  - These scripts become **command jobs** and **endpoint handlers**.
  - Pipelines orchestrate them in dev / staging / production.

---

## Q3. How would you productionize this on Azure?

**Answer**

- **Data layer**
  - Use Azure Blob / Data Lake for raw and curated data zones.
- **Training**
  - Run `train_and_register_model` as an Azure ML command job.
  - Log metrics and register the model in the Azure ML model registry.
- **Online inference**
  - Package the scoring logic (from FastAPI) into an Azure ML managed online endpoint.
  - Expose a `/score` or `/predict` route secured via AAD / tokens.
- **Batch inference**
  - Run `batch_score` (or equivalent) as:
    - An Azure ML batch job, or
    - A step in an ADF/Synapse pipeline.
- **Orchestration**
  - Use ADF/Synapse pipelines as the equivalent of `run_daily_pipeline`:
    - Parameterize by date and environment.
    - Trigger on a schedule or via events.
- **Monitoring & governance**
  - Send logs and metrics to Azure Monitor / Log Analytics.
  - Use RBAC and Key Vault for secrets and access control.

---

## Q4. When do you prefer online vs batch scoring?

**Answer**

- **Online scoring (FastAPI / managed endpoint)**
  - Low-latency decisions (e.g. fraud checks at transaction time, instant credit decisions).
  - Integrates directly with front-end / transactional systems.
  - Endpoint like `POST /predict` with per-request features.
- **Batch scoring (`batch_score`, nightly pipeline)**
  - Large populations scored on a schedule (hourly/daily/weekly).
  - Use cases: daily delinquency lists, churn risk lists, marketing campaigns.
- **In this project**
  - Implemented both:
    - `/predict` for interactive / real-time needs.
    - `batch_score` + `run_daily_pipeline` for scheduled, large-volume scoring.

---

## Q5. How do you version and manage models?

**Answer**

- Each training run:
  - Produces a model artifact (`model.joblib`) and metrics (`metrics.json`).
  - Registers a new directory under `local_registry/model_vNNN/`.
  - Writes metadata (`model_name`, `version`, `registered_at`) to `metadata.json`.
- “Latest model”:
  - Determined by sorting `model_vNNN` folders and taking the highest version.
  - Used by both batch and online scoring logic.
- In Azure ML:
  - Same idea with **model registry**:
    - Models are named and versioned centrally.
    - Online endpoints and pipelines reference specific model versions.
    - You can roll back or A/B test by switching model version bindings.

---

## Q6. How would you monitor this in production?

**Answer**

- **Model performance**
  - Log metrics like accuracy and AUC at training time (`metrics.json`).
  - Periodically evaluate the model on fresh labeled data.
  - Track performance over time to detect degradation.
- **Operational metrics**
  - Batch:
    - Log each run in `logs/batch_runs_log.csv` (or a database / Log Analytics).
    - Capture timestamp, model version, rows processed, success/failure.
  - Online:
    - Track latency, error rates, request volume, and payload sizes.
- **Data / model drift**
  - Compare feature distributions between training data and recent scoring data.
  - Trigger investigations or retraining jobs when drift thresholds are exceeded.
- **Azure mapping**
  - Use Azure Monitor + Log Analytics + Application Insights for:
    - Central log collection.
    - Dashboards and alerts.

---

## Q7. How do you think about data and feature pipelines for this project?

**Answer**

- **Layered storage**
  - `raw/` zone (or Blob/Data Lake “raw” container) for ingested data as-is.
  - `curated/` zone for cleaned, feature-ready data and predictions.
- **Feature pipeline**
  - In this mini-project, features are mostly direct from the CSV plus scaling.
  - In a real system, you’d add:
    - Joins across multiple sources.
    - Feature engineering (aggregations, ratios, time windows).
- **Batch pipeline (`run_daily_pipeline`)**
  - Reads raw accounts file for a given date.
  - Calls `batch_score` with the latest model.
  - Writes curated predictions for downstream use (reports, dashboards, ops tools).
- **Azure mapping**
  - Use ADF/Synapse to orchestrate:
    - Ingestion to raw.
    - Transformation to feature/curated.
    - Invocation of Azure ML batch jobs.
  - Store intermediate outputs in different containers/folders for clarity and governance.

---

## Q8. How would you explain this project to a non-technical stakeholder?

**Answer**

- “I built a small prototype that:
  - Takes your historical data from storage.
  - Trains a model to identify which customers/accounts are higher risk.
  - Then uses that model in two ways:
    - As an API the system can call in real time when it needs a quick decision.
    - As a nightly batch job that produces a list of high-risk customers for your team to review.
  - The design mirrors how we’d implement it on Azure, so it’s directly transferable to a production environment.”