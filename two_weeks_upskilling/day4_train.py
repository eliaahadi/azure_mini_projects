#!/usr/bin/env python3
"""
train.py

Day 4: Refactor into a reusable training script and simulate a model registry.

- Loads binary classification data from a CSV (if provided), otherwise from sklearn.
- Trains a Pipeline: StandardScaler + LogisticRegression.
- Prints metrics.
- Saves the trained pipeline to an artifacts directory.
- Copies the artifacts into a versioned "local registry" folder:
    local_registry/model_v001/
    local_registry/model_v002/
    ...

Later, this maps to:
- Azure ML command job for training.
- Azure ML model registry for versioned models.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(data_path: Path | None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from a CSV if data_path exists, otherwise use sklearn breast_cancer.
    CSV is expected to have a 'target' column (0/1) and feature columns.
    """
    if data_path is not None and data_path.exists():
        print(f"[data] Loading data from CSV: {data_path}")
        df = pd.read_csv(data_path)
        if "target" not in df.columns:
            raise ValueError("CSV must contain a 'target' column.")
    else:
        print("[data] CSV not found, using sklearn breast_cancer dataset.")
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        # Ensure target column is named 'target'
        if "target" not in df.columns:
            df["target"] = data.target

    X = df.drop(columns=["target"])
    y = df["target"]
    print(f"[data] Shape X: {X.shape}, y: {y.shape}")
    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a simple Pipeline: StandardScaler + LogisticRegression.
    Returns trained pipeline and metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[train] Train size: {len(y_train)}, Test size: {len(y_test)}")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"[metrics] Accuracy: {acc:.3f}")
    print(f"[metrics] ROC AUC:  {auc:.3f}")
    print("[metrics] Classification report:")
    print(classification_report(y_test, y_pred))

    metrics = {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "random_state": random_state,
        "test_size": test_size,
    }

    return pipeline, metrics


def save_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, Any],
    artifacts_dir: Path,
) -> Path:
    """
    Save the trained pipeline and metrics to the artifacts directory.
    Returns the path to the artifacts directory.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[artifacts] Saved model to {model_path}")

    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[artifacts] Saved metrics to {metrics_path}")

    return artifacts_dir


def get_next_version(registry_dir: Path, prefix: str = "model_v") -> str:
    """
    Inspect existing versioned folders and return the next version string, e.g. 'model_v001'.
    """
    registry_dir.mkdir(parents=True, exist_ok=True)
    existing = glob(str(registry_dir / f"{prefix}*"))

    if not existing:
        return f"{prefix}001"

    versions = []
    for path in existing:
        name = Path(path).name
        try:
            num = int(name.replace(prefix, ""))
            versions.append(num)
        except ValueError:
            continue

    next_num = max(versions) + 1 if versions else 1
    return f"{prefix}{next_num:03d}"


def register_model_locally(
    artifacts_dir: Path,
    registry_dir: Path,
    model_name: str = "risk_classifier",
) -> Path:
    """
    Simulate 'model registration' by copying artifacts into a versioned directory
    under local_registry/.

    Layout:
        local_registry/
            model_v001/
                model.joblib
                metrics.json
                metadata.json
    """
    version_name = get_next_version(registry_dir)
    target_dir = registry_dir / version_name
    target_dir.mkdir(parents=True, exist_ok=False)

    # Copy model + metrics
    for file_name in ["model.joblib", "metrics.json"]:
        src = artifacts_dir / file_name
        if src.exists():
            shutil.copy2(src, target_dir / file_name)

    # Write a simple metadata file
    metadata = {
        "model_name": model_name,
        "version": version_name,
        "registered_at": datetime.utcnow().isoformat() + "Z",
    }
    metadata_path = target_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[registry] Registered model as {version_name} in {registry_dir}")
    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 4: train model and simulate a local model registry."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="day1_breast_cancer.csv",
        help="Path to CSV with 'target' column (defaults to day1_breast_cancer.csv).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory to save model and metrics.",
    )
    parser.add_argument(
        "--registry-dir",
        type=str,
        default="local_registry",
        help="Directory for local simulated model registry.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    artifacts_dir = Path(args.artifacts_dir)
    registry_dir = Path(args.registry_dir)

    X, y = load_data(data_path if data_path.exists() else None)
    pipeline, metrics = train_model(
        X,
        y,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    artifacts_path = save_artifacts(pipeline, metrics, artifacts_dir)
    register_model_locally(artifacts_path, registry_dir)


if __name__ == "__main__":
    main()