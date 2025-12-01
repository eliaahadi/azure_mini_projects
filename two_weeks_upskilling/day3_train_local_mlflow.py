#!/usr/bin/env python3
"""
day3_train_local_mlflow.py

Train a simple baseline classifier and log metrics to MLflow.
This simulates an Azure ML experiment locally.
"""

from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_experiment(random_state: int = 42, max_iter: int = 1000) -> None:
    # Configure local MLflow tracking (file-based)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("day3_azure_style_experiment")

    # Load data
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Start MLflow run
    with mlflow.start_run(run_name="logreg_baseline"):
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"Accuracy: {acc:.3f}")
        print(f"ROC AUC:  {auc:.3f}")

        # Log params & metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 3: local MLflow experiment (Azure-style)."
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(random_state=args.random_state, max_iter=args.max_iter)


if __name__ == "__main__":
    main()