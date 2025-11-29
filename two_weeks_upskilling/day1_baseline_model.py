#!/usr/bin/env python3
"""
day1_baseline_model.py

Baseline binary classifier using scikit-learn.
This is your local 'sandbox' model that you will later port to Azure ML.
"""

from __future__ import annotations

import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(random_state: int = 42) -> Dict[str, Any]:
    # 1) Load data
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    # By default, df has a 'target' column
    X = df.drop(columns=["target"])
    y = df["target"]

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    # 3) Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC:  {auc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    summary = {
        "dataset": "breast_cancer (sklearn)",
        "model": "LogisticRegression",
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "random_state": random_state,
    }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 1 baseline model (local / open source)."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_and_evaluate(random_state=args.random_state)
    print("\nSummary:", summary)


if __name__ == "__main__":
    main()