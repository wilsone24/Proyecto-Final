"""
test_training.py
----------------
Runs the full training pipeline locally without Databricks.

Requirements
    pip install xgboost scikit-learn mlflow pandas matplotlib

Usage
    python tests/test_training.py
    # or, from the repo root:
    python -m tests.test_training
"""

import os
import sys
import pathlib

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = pathlib.Path(__file__).parent.parent
DATA_FILE   = REPO_ROOT / "data" / "cardio_train.csv"
MLFLOW_DB   = REPO_ROOT / "mlruns" / "mlflow.db"
ARTIFACT_DIR = REPO_ROOT / "mlruns" / "artifacts"

# SQLite tracking URI — works without a running MLflow server
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

# ---------------------------------------------------------------------------
# 1. Load & pre-process (mirrors gold.cardiofeatures)
# ---------------------------------------------------------------------------
def load_features(csv_path: pathlib.Path) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "Place cardio_train.csv in the data/ folder and re-run."
        )

    raw = pd.read_csv(csv_path, sep=";")

    # ── Replicates silver transformations ──────────────────────────────────
    raw = raw[
        raw["age"].between(365, 365 * 120) &
        raw["height"].between(100, 250) &
        raw["weight"].between(10, 200) &
        raw["ap_hi"].between(60, 300) &
        raw["ap_lo"].between(40, 200) &
        (raw["ap_lo"] < raw["ap_hi"]) &
        raw["cardio"].notna()
    ].copy()

    # ── Replicates gold feature engineering ────────────────────────────────
    raw["age_years"]   = (raw["age"] / 365.25).round(1)
    raw["age_group_id"] = pd.cut(
        raw["age_years"],
        bins=[0, 30, 45, 60, 75, 999],
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    raw["height_cm"]      = raw["height"].astype(float)
    raw["weight_kg"]      = raw["weight"].astype(float)
    raw["bmi"]            = (raw["weight_kg"] / (raw["height_cm"] / 100) ** 2).round(2)
    raw["systolic_bp"]    = raw["ap_hi"]
    raw["diastolic_bp"]   = raw["ap_lo"]
    raw["pulse_pressure"] = raw["ap_hi"] - raw["ap_lo"]
    raw["hypertension"]   = ((raw["ap_hi"] >= 140) | (raw["ap_lo"] >= 90)).astype(int)
    raw["is_smoker"]            = raw["smoke"].astype(int)
    raw["drinks_alcohol"]       = raw["alco"].astype(int)
    raw["is_physically_active"] = raw["active"].astype(int)

    FEATURES = [
        "age_years", "age_group_id", "gender",
        "height_cm", "weight_kg", "bmi",
        "systolic_bp", "diastolic_bp", "pulse_pressure",
        "hypertension", "cholesterol", "gluc",
        "is_smoker", "drinks_alcohol", "is_physically_active",
    ]
    TARGET = "cardio"

    X = raw[FEATURES].fillna(raw[FEATURES].median())
    y = raw[TARGET].astype(int)

    print(f"Dataset loaded  — rows: {len(X):,}  |  features: {len(FEATURES)}")
    print(f"Class balance   — 0: {(y==0).sum():,}  1: {(y==1).sum():,}")
    return X, y


# ---------------------------------------------------------------------------
# 2. Train with MLflow (local tracking)
# ---------------------------------------------------------------------------
def train(X: pd.DataFrame, y: pd.Series) -> None:
    # Ensure the mlruns directory exists before SQLite creates the DB file
    MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("cardiovascular_local")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    params = {
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           -1,
    }

    mlflow.xgboost.autolog()

    with mlflow.start_run(run_name="xgboost_cardiovascular_v1") as run:
        mlflow.set_tags({
            "dataset": "ocgn_cardio_train",
            "source":  "local_csv",
            "modelo":  "XGBoost",
            "version": "1.0",
        })

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50,
        )

        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_pred_proba),
            "f1_score":  f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std",  cv_scores.std())

        # Confusion matrix artifact
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=["Sin afección", "Con afección"],
        ).plot(ax=ax, colorbar=False)
        plt.title("Confusion Matrix - Cardiovascular XGBoost")
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        # Feature importance artifact
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        xgb.plot_importance(model, ax=ax2, max_num_features=15, importance_type="gain")
        plt.title("Feature Importance (Gain)")
        plt.tight_layout()
        mlflow.log_figure(fig2, "feature_importance.png")
        plt.close()

        # Log model (no registry; Unity Catalog is Databricks-only)
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        print("\nMetrics:", metrics)
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\nRun ID : {run.info.run_id}")
        print(f"DB     : {MLFLOW_DB}")
        print(f"Launch : mlflow ui --backend-store-uri {TRACKING_URI} --default-artifact-root {ARTIFACT_DIR.as_uri()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y = load_features(DATA_FILE)
    train(X, y)
