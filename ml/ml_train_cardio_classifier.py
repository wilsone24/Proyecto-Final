# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Cardiovascular Disease — ML Training Pipeline
# MAGIC
# MAGIC Trains XGBoost vs Random Forest, picks the winner, registers it as `@candidate`.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# Install ML libraries not available in the standard Databricks runtime.
%pip install xgboost optuna shap

# COMMAND ----------

# DBTITLE 1,Restart Python
# Restart the Python kernel so the freshly-installed libraries are picked up.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Setup and configuration

# COMMAND ----------

# DBTITLE 1,Catalog
# Set the active Unity Catalog for all subsequent table references.
spark.sql("USE CATALOG `databricks_service_pf`")

# COMMAND ----------

# DBTITLE 1,Libraries
# ML stack: sklearn for splits/metrics/RF, XGBoost, Optuna for tuning, SHAP for interpretability, MLflow for tracking.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
)

from xgboost import XGBClassifier
import xgboost as xgb

import optuna
from optuna.samplers import TPESampler

import shap

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

import sklearn

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
optuna.logging.set_verbosity(optuna.logging.WARNING)

# COMMAND ----------

# DBTITLE 1,Parameters
# Job-level inputs: source feature table, target column, registered model name and MLflow experiment.
dbutils.widgets.text("source_schema", "gold")
dbutils.widgets.text("source_table", "cardio_features")
dbutils.widgets.text("target_column", "cardio")
dbutils.widgets.text(
    "registered_model_name", "databricks_service_pf.gold.cardio_classifier"
)
dbutils.widgets.text("mlflow_experiment_path", "/Shared/cardio_ml")

# COMMAND ----------

# DBTITLE 1,Constants
# Validate widgets and derive runtime constants (I/O, split sizes, tuning budget, thresholds, SHAP config).
SOURCE_SCHEMA          = dbutils.widgets.get("source_schema")
SOURCE_TABLE           = dbutils.widgets.get("source_table")
TARGET_COLUMN          = dbutils.widgets.get("target_column")
REGISTERED_MODEL_NAME  = dbutils.widgets.get("registered_model_name")
MLFLOW_EXPERIMENT_PATH = dbutils.widgets.get("mlflow_experiment_path")

# Fail fast if any required widget was not provided
if not all([SOURCE_SCHEMA, SOURCE_TABLE, TARGET_COLUMN, REGISTERED_MODEL_NAME, MLFLOW_EXPERIMENT_PATH]):
    raise ValueError(
        f"Missing required widgets: source_schema='{SOURCE_SCHEMA}', "
        f"source_table='{SOURCE_TABLE}', target_column='{TARGET_COLUMN}', "
        f"registered_model_name='{REGISTERED_MODEL_NAME}', "
        f"mlflow_experiment_path='{MLFLOW_EXPERIMENT_PATH}'"
    )

# Registry conventions
CANDIDATE_ALIAS = "candidate"

# Split configuration — fixed by experimental design
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
RANDOM_STATE = 42

# Hyperparameter tuning
N_OPTUNA_TRIALS = 20
CV_FOLDS        = 5

# XGBoost early stopping (active inside the Optuna objective only)
XGB_EARLY_STOPPING_ROUNDS = 15

# Threshold sweep
THRESHOLD_MIN  = 0.10
THRESHOLD_MAX  = 0.90
THRESHOLD_STEP = 0.01

# Interpretability
SHAP_SAMPLE_SIZE = 1000
SHAP_LOCAL_N     = 3

# Derived constants
FULL_SOURCE         = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
PIPELINE_VERSION    = "1.0.0"
OPTIMIZATION_METRIC = "roc_auc"   # Optuna objective
THRESHOLD_METRIC    = "f1"        # threshold optimization criterion

THRESHOLD_GRID = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)

RUN_TAGS = {
    "pipeline.version":    PIPELINE_VERSION,
    "data.source":         FULL_SOURCE,
    "data.layer":          "gold",
    "data.subject":        "cardiovascular-disease",
    "data.owner":          "data-engineering",
    "data.purpose":        "ml-classifier",
    "optimization.metric": OPTIMIZATION_METRIC,
    "threshold.metric":    THRESHOLD_METRIC,
    "stage":               "training",
}

print(f"Source:                 {FULL_SOURCE}")
print(f"Registered model:       {REGISTERED_MODEL_NAME}")
print(f"Candidate alias:        @{CANDIDATE_ALIAS}")
print(f"Split (train/val/test): {1-TEST_SIZE-VAL_SIZE:.2f} / {VAL_SIZE:.2f} / {TEST_SIZE:.2f}")
print(f"Optuna trials per algo: {N_OPTUNA_TRIALS}")
print(f"CV folds:               {CV_FOLDS}")

# COMMAND ----------

# DBTITLE 1,MLflow configuration
# Point MLflow at the Unity Catalog model registry and the shared experiment path.
try:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
    mlflow_client = MlflowClient()
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_PATH}")
    print(f"Registry URI:      databricks-uc")

except Exception as e:
    raise Exception(f"[MLflow Setup] Failed to configure MLflow: {e}")

# COMMAND ----------

# DBTITLE 1,Helper functions
# Reusable utilities for metrics, cross-validation, threshold sweep and MLflow metric logging.
def compute_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute the full classification metric suite for binary classification.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_proba: Predicted probabilities for the positive class.
        threshold: Decision threshold used to derive binary predictions from probabilities.

    Returns:
        Dict with keys: accuracy, precision, recall, f1 (threshold-dependent),
        roc_auc and pr_auc (threshold-independent).
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba),
        "pr_auc":    average_precision_score(y_true, y_proba),
    }


def cv_evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: str = "roc_auc",
) -> tuple:
    """Run cross-validation and return aggregated scores.

    Args:
        model: sklearn-compatible estimator.
        X: Feature matrix.
        y: Target vector.
        cv: Cross-validation splitter (e.g. StratifiedKFold instance).
        scoring: sklearn scoring string (e.g. "roc_auc", "f1").

    Returns:
        A (mean, std, raw_scores) tuple — mean and std as floats, raw_scores
        as a numpy array of per-fold scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return float(scores.mean()), float(scores.std()), scores


def find_optimal_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    metric_fn=f1_score,
) -> tuple:
    """Sweep decision thresholds and return the one that maximises `metric_fn`.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_proba: Predicted probabilities for the positive class.
        thresholds: Array of thresholds to evaluate (e.g. np.arange(0.1, 0.9, 0.01)).
        metric_fn: sklearn metric callable to maximise (default f1_score).
            Must accept y_true, y_pred and a zero_division kwarg.

    Returns:
        A (best_threshold, best_score, results_df) tuple. results_df has one
        row per threshold with columns: threshold, score, precision, recall.
    """
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "score":     float(metric_fn(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        })
    results_df = pd.DataFrame(rows)
    best_row   = results_df.loc[results_df["score"].idxmax()]
    return float(best_row["threshold"]), float(best_row["score"]), results_df


def log_metric_dict(metrics: dict, prefix: str = "") -> None:
    """Log a dict of metrics to the active MLflow run with an optional prefix.

    Args:
        metrics: Mapping from metric name to numeric value.
        prefix: Optional prefix prepended to every metric name (e.g. "test"
            yields "test_accuracy", "test_f1", ...). Use an empty string to
            log without a prefix.
    """
    for name, value in metrics.items():
        key = f"{prefix}_{name}" if prefix else name
        mlflow.log_metric(key, float(value))

# COMMAND ----------

# DBTITLE 1,Custom pyfunc wrapper
# Wrapper that embeds the optimal threshold so serving consumers don't need to know it.
class CardioModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow pyfunc wrapper for the cardiovascular disease classifier.

    Exposes a uniform predict() interface over any sklearn-compatible classifier
    (XGBoost, Random Forest, ...). The optimal decision threshold is baked into
    the model object so downstream consumers (Model Serving endpoints, batch
    inference jobs, etc.) get binary predictions without having to know the
    threshold externally.

    Predict output is a DataFrame with two columns:
        - probability: probability of CVD (positive class).
        - prediction:  binary class derived from the embedded threshold.
    """

    def __init__(self, model, optimal_threshold: float):
        """Initialise the wrapper with a fitted estimator and decision threshold.

        Args:
            model: A fitted sklearn-compatible classifier exposing predict_proba().
            optimal_threshold: Decision threshold (between 0 and 1) used to derive
                binary predictions from probabilities.
        """
        self.model = model
        self.optimal_threshold = float(optimal_threshold)

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        """Predict probabilities and binary classes for a batch of inputs.

        Args:
            context: MLflow PythonModelContext (unused; required by the pyfunc API).
            model_input: Features as a pandas DataFrame or numpy array.
            params: Optional inference parameters (unused).

        Returns:
            DataFrame with `probability` (float, P(CVD)) and `prediction`
            (int, 0/1) columns.
        """
        if isinstance(model_input, np.ndarray):
            model_input = pd.DataFrame(model_input)
        proba      = self.model.predict_proba(model_input)[:, 1]
        prediction = (proba >= self.optimal_threshold).astype(int)
        return pd.DataFrame({
            "probability": proba,
            "prediction":  prediction,
        })

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data loading

# COMMAND ----------

# DBTITLE 1,Load features table
# Pull the feature table into pandas — 70k rows fit comfortably in driver memory.
try:
    features_spark = spark.table(FULL_SOURCE)
    features_df    = features_spark.toPandas()
    source_count   = len(features_df)

    if source_count == 0:
        raise Exception(f"Source table {FULL_SOURCE} returned 0 rows.")

    print(f"Loaded {source_count:,} rows × {features_df.shape[1]} columns from {FULL_SOURCE}")

except Exception as e:
    raise Exception(f"[Load] Failed to read {FULL_SOURCE}: {e}")

# COMMAND ----------

# DBTITLE 1,Feature/target split
# Separate features from the target and confirm no nulls (silver should have imputed them).
# Features excluded from training — derived from other features (see EDA).
# Mutual information confirmed they have less info than their raw components,
# and Spearman correlation >0.75 with their parent features → multicollinearity.
EXCLUDED_FEATURES = ["hypertension", "pulse_pressure", "age_group_id"]

try:
    if TARGET_COLUMN not in features_df.columns:
        raise Exception(f"Target column '{TARGET_COLUMN}' not found in {FULL_SOURCE}.")

    feature_columns = [c for c in features_df.columns if c != TARGET_COLUMN and c != "hypertension"  and c not in EXCLUDED_FEATURES]
    X = features_df[feature_columns].copy()
    y = features_df[TARGET_COLUMN].astype(int).copy()

    if X.isnull().any().any():
        raise Exception(f"Found null values in features. Silver layer should have imputed these.")

    class_balance = y.value_counts(normalize=True).sort_index().to_dict()

    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print(f"Target balance:  {class_balance}")

except Exception as e:
    raise Exception(f"[Feature-Target Split] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Train / Validation / Test split (70/15/15 stratified)

# COMMAND ----------

# DBTITLE 1,Three-way stratified split
# Two-pass split: first carve out test (15%), then split the remaining 85% into train/val.
try:
    # Pass 1: separate test (15%) from the rest (85%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Pass 2: split the remaining 85% into train and validation.
    # VAL_SIZE is expressed relative to the full dataset, so adjust it.
    val_size_relative = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_relative,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    split_summary = pd.DataFrame({
        "split":   ["train", "validation", "test"],
        "rows":    [len(X_train), len(X_val), len(X_test)],
        "pct":     [
            len(X_train) / len(X),
            len(X_val)   / len(X),
            len(X_test)  / len(X),
        ],
        "cardio_rate": [
            float(y_train.mean()),
            float(y_val.mean()),
            float(y_test.mean()),
        ],
    })
    display(split_summary)

except Exception as e:
    raise Exception(f"[Split] Train/val/test split failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Baseline models

# COMMAND ----------

# DBTITLE 1,XGBoost baseline
# Untuned XGBoost — establishes a reference performance for the tuned model to beat.
try:
    baseline_xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="auc",
        use_label_encoder=False,
    )
    baseline_xgb.fit(X_train, y_train)

    baseline_xgb_val_proba = baseline_xgb.predict_proba(X_val)[:, 1]
    baseline_xgb_metrics   = compute_metrics(y_val, baseline_xgb_val_proba)

    print("Baseline XGBoost — validation metrics:")
    for k, v in baseline_xgb_metrics.items():
        print(f"  {k:<10s}: {v:.4f}")

except Exception as e:
    raise Exception(f"[Baseline XGBoost] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,Random Forest baseline
# Untuned Random Forest — reference performance for the second tuned candidate.
try:
    baseline_rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    baseline_rf.fit(X_train, y_train)

    baseline_rf_val_proba = baseline_rf.predict_proba(X_val)[:, 1]
    baseline_rf_metrics   = compute_metrics(y_val, baseline_rf_val_proba)

    print("Baseline Random Forest — validation metrics:")
    for k, v in baseline_rf_metrics.items():
        print(f"  {k:<10s}: {v:.4f}")

except Exception as e:
    raise Exception(f"[Baseline Random Forest] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Cross validation of baselines

# COMMAND ----------

# DBTITLE 1,Stratified KFold setup
# Stratified folds preserve class balance across splits — essential for imbalanced binary targets.
cv_splitter = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# COMMAND ----------

# DBTITLE 1,CV - baseline XGBoost
# Cross-validate the untuned XGBoost to get a robust performance reference.
try:
    cv_baseline_xgb_mean, cv_baseline_xgb_std, _ = cv_evaluate(
        XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                      eval_metric="auc", use_label_encoder=False),
        X_train, y_train, cv_splitter, scoring=OPTIMIZATION_METRIC,
    )
    print(f"CV baseline XGBoost  {OPTIMIZATION_METRIC}: "
          f"{cv_baseline_xgb_mean:.4f} ± {cv_baseline_xgb_std:.4f}")

except Exception as e:
    raise Exception(f"[CV Baseline XGBoost] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,CV - baseline Random Forest
# Cross-validate the untuned Random Forest for symmetry with XGBoost.
try:
    cv_baseline_rf_mean, cv_baseline_rf_std, _ = cv_evaluate(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        X_train, y_train, cv_splitter, scoring=OPTIMIZATION_METRIC,
    )
    print(f"CV baseline RF       {OPTIMIZATION_METRIC}: "
          f"{cv_baseline_rf_mean:.4f} ± {cv_baseline_rf_std:.4f}")

except Exception as e:
    raise Exception(f"[CV Baseline RF] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Hyperparameter tuning with Optuna

# COMMAND ----------

# DBTITLE 1,Objective - XGBoost
# Optuna objective for XGBoost — early stopping inside the trial speeds tuning ~2-3x.
def objective_xgboost(trial: optuna.Trial) -> float:
    """Optuna objective for the XGBoost search.

    Suggests a hyperparameter combination, fits on `X_train` with early stopping
    against `X_val`, and returns the validation ROC-AUC. Stores the actual
    `best_iteration` as a user attribute so downstream steps can use the
    truncated tree count instead of the upper-bound `n_estimators`.

    Args:
        trial: Optuna Trial object.

    Returns:
        Validation ROC-AUC achieved by this hyperparameter combination.
    """
    suggested_params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha",  1e-8, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
    }
    # Early stopping inside the Optuna trial only — speeds up tuning by 2-3x.
    # The actual best_iteration is captured so the downstream CV / final
    # training use the truncated tree count, not the original suggestion.
    model = XGBClassifier(
        **suggested_params,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="auc",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    trial.set_user_attr("best_iteration", int(model.best_iteration) + 1)
    val_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_proba)

# COMMAND ----------

# DBTITLE 1,Run Optuna - XGBoost
# Launch the XGBoost tuning study; afterward replace the suggested n_estimators with the actual best_iteration.
try:
    sampler_xgb = TPESampler(seed=RANDOM_STATE)
    study_xgb   = optuna.create_study(direction="maximize", sampler=sampler_xgb,
                                      study_name="cardio_xgboost")
    study_xgb.optimize(objective_xgboost, n_trials=N_OPTUNA_TRIALS,
                       show_progress_bar=False)

    best_params_xgb = study_xgb.best_params.copy()
    best_score_xgb  = study_xgb.best_value

    # Replace the suggested n_estimators with the actual best_iteration from
    # early stopping. This guarantees downstream steps train the right number
    # of trees instead of the upper-bound originally suggested by Optuna.
    suggested_n_estimators = best_params_xgb["n_estimators"]
    best_iteration         = int(study_xgb.best_trial.user_attrs["best_iteration"])
    best_params_xgb["n_estimators"] = best_iteration

    print(f"XGBoost best validation {OPTIMIZATION_METRIC}: {best_score_xgb:.4f}")
    print(f"Early stopping: suggested n_estimators={suggested_n_estimators}, "
          f"actual best_iteration={best_iteration}")
    print(f"Best params: {best_params_xgb}")

except Exception as e:
    raise Exception(f"[Optuna XGBoost] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,Objective - Random Forest
# Optuna objective for Random Forest — search space narrowed deliberately for speed (see inline notes).
def objective_rf(trial: optuna.Trial) -> float:
    """Optuna objective for the Random Forest search.

    Search space deliberately narrowed for speed:
      - n_estimators capped at 300 (more rarely helps on tabular data).
      - max_depth capped at 20 (deeper trees overfit + train slower).
      - max_features excludes None (using all features per split is slow and overfits).
      - bootstrap fixed to True (False disables subsampling — slow + worse generalisation).

    Args:
        trial: Optuna Trial object.

    Returns:
        Validation ROC-AUC achieved by this hyperparameter combination.
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 300),
        "max_depth":         trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight":      trial.suggest_categorical("class_weight", [None, "balanced"]),
        "bootstrap":         True,
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_proba)

# COMMAND ----------

# DBTITLE 1,Run Optuna - Random Forest
# Launch the Random Forest tuning study (same trial budget as XGBoost).
try:
    sampler_rf = TPESampler(seed=RANDOM_STATE)
    study_rf   = optuna.create_study(direction="maximize", sampler=sampler_rf,
                                     study_name="cardio_random_forest")
    study_rf.optimize(objective_rf, n_trials=N_OPTUNA_TRIALS,
                      show_progress_bar=False)

    best_params_rf = study_rf.best_params
    best_score_rf  = study_rf.best_value

    print(f"Random Forest best validation {OPTIMIZATION_METRIC}: {best_score_rf:.4f}")
    print(f"Best params: {best_params_rf}")

except Exception as e:
    raise Exception(f"[Optuna RF] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Cross validation of tuned models — pick winner

# COMMAND ----------

# DBTITLE 1,CV - tuned XGBoost
# Cross-validate the tuned XGBoost to confirm the gain over baseline is robust.
try:
    tuned_xgb = XGBClassifier(
        **best_params_xgb,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="auc", use_label_encoder=False,
    )
    cv_tuned_xgb_mean, cv_tuned_xgb_std, _ = cv_evaluate(
        tuned_xgb, X_train, y_train, cv_splitter, scoring=OPTIMIZATION_METRIC,
    )
    print(f"CV tuned XGBoost  {OPTIMIZATION_METRIC}: "
          f"{cv_tuned_xgb_mean:.4f} ± {cv_tuned_xgb_std:.4f}")

except Exception as e:
    raise Exception(f"[CV Tuned XGBoost] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,CV - tuned Random Forest
# Cross-validate the tuned Random Forest under the same protocol as XGBoost.
try:
    tuned_rf = RandomForestClassifier(
        **best_params_rf,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    cv_tuned_rf_mean, cv_tuned_rf_std, _ = cv_evaluate(
        tuned_rf, X_train, y_train, cv_splitter, scoring=OPTIMIZATION_METRIC,
    )
    print(f"CV tuned RF       {OPTIMIZATION_METRIC}: "
          f"{cv_tuned_rf_mean:.4f} ± {cv_tuned_rf_std:.4f}")

except Exception as e:
    raise Exception(f"[CV Tuned RF] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,Comparison and winner selection
# Pick the algorithm with the higher CV ROC-AUC as the winner for the rest of the pipeline.
try:
    comparison_df = pd.DataFrame([
        {"model": "baseline_xgboost", "cv_mean": cv_baseline_xgb_mean, "cv_std": cv_baseline_xgb_std},
        {"model": "baseline_rf",      "cv_mean": cv_baseline_rf_mean,  "cv_std": cv_baseline_rf_std},
        {"model": "tuned_xgboost",    "cv_mean": cv_tuned_xgb_mean,    "cv_std": cv_tuned_xgb_std},
        {"model": "tuned_rf",         "cv_mean": cv_tuned_rf_mean,     "cv_std": cv_tuned_rf_std},
    ]).sort_values("cv_mean", ascending=False).reset_index(drop=True)
    display(comparison_df)

    # Winner between the two tuned models
    if cv_tuned_xgb_mean >= cv_tuned_rf_mean:
        WINNER_ALGORITHM   = "XGBoost"
        winner_estimator   = tuned_xgb
        winner_best_params = best_params_xgb
        winner_cv_mean     = cv_tuned_xgb_mean
        winner_cv_std      = cv_tuned_xgb_std
    else:
        WINNER_ALGORITHM   = "RandomForest"
        winner_estimator   = tuned_rf
        winner_best_params = best_params_rf
        winner_cv_mean     = cv_tuned_rf_mean
        winner_cv_std      = cv_tuned_rf_std

    print(f"\nWINNER: {WINNER_ALGORITHM}")
    print(f"  CV {OPTIMIZATION_METRIC}: {winner_cv_mean:.4f} ± {winner_cv_std:.4f}")

except Exception as e:
    raise Exception(f"[Winner Selection] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. Threshold optimization (on validation)

# COMMAND ----------

# DBTITLE 1,Sweep thresholds with winner trained on train-only
# Train a fresh copy of the winner on train only, then sweep thresholds against validation F1.
try:
    winner_train_only = type(winner_estimator)(**winner_estimator.get_params())
    winner_train_only.fit(X_train, y_train)

    winner_val_proba = winner_train_only.predict_proba(X_val)[:, 1]

    optimal_threshold, best_threshold_score, threshold_results = find_optimal_threshold(
        y_val, winner_val_proba, THRESHOLD_GRID, metric_fn=f1_score,
    )

    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"Validation F1 at optimal threshold: {best_threshold_score:.4f}")
    display(threshold_results.head(10))

except Exception as e:
    raise Exception(f"[Threshold Optimization] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,Threshold sweep plot
# Visualise how F1, precision and recall trade off across the threshold grid.
try:
    fig_threshold, ax = plt.subplots(figsize=(9, 5))
    ax.plot(threshold_results["threshold"], threshold_results["score"],
            label="F1 score", linewidth=2)
    ax.plot(threshold_results["threshold"], threshold_results["precision"],
            label="Precision", linewidth=1.5, linestyle="--")
    ax.plot(threshold_results["threshold"], threshold_results["recall"],
            label="Recall", linewidth=1.5, linestyle="--")
    ax.axvline(optimal_threshold, color="red", linestyle=":", linewidth=2,
               label=f"Optimal = {optimal_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(f"{WINNER_ALGORITHM}: threshold optimization on validation")
    ax.legend()
    plt.tight_layout()
    plt.show()

except Exception as e:
    raise Exception(f"[Threshold Plot] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 9. Final training (winner on train + validation)

# COMMAND ----------

# DBTITLE 1,Retrain winner on train + val combined
# Now that hyperparameters and threshold are fixed, refit on train+val for maximum signal.
try:
    X_trainval_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_trainval_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    final_estimator = type(winner_estimator)(**winner_estimator.get_params())
    final_estimator.fit(X_trainval_combined, y_trainval_combined)

    print(f"Final model retrained on {len(X_trainval_combined):,} rows "
          f"(train + validation combined)")

except Exception as e:
    raise Exception(f"[Final Training] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 10. Final evaluation on test

# COMMAND ----------

# DBTITLE 1,Compute final metrics
# Evaluate the final model on the held-out test set using the embedded optimal threshold.
try:
    final_test_proba   = final_estimator.predict_proba(X_test)[:, 1]
    final_test_metrics = compute_metrics(y_test, final_test_proba,
                                         threshold=optimal_threshold)

    print(f"Final evaluation on test (threshold = {optimal_threshold:.2f}):")
    for k, v in final_test_metrics.items():
        print(f"  {k:<10s}: {v:.4f}")

except Exception as e:
    raise Exception(f"[Final Evaluation] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,Confusion matrix
# Confusion matrix and per-class report — reveals where the model errs (FP vs FN).
try:
    y_test_pred = (final_test_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)

    fig_cm, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No CVD", "CVD"], yticklabels=["No CVD", "CVD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{WINNER_ALGORITHM} — confusion matrix on test "
                 f"(threshold = {optimal_threshold:.2f})")
    plt.tight_layout()
    plt.show()

    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred,
                                target_names=["No CVD", "CVD"]))

except Exception as e:
    raise Exception(f"[Confusion Matrix] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,ROC and PR curves
# Visualise discrimination (ROC) and class-imbalance-aware performance (PR) on test.
try:
    fpr, tpr, _                      = roc_curve(y_test, final_test_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, final_test_proba)

    fig_curves, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))

    ax_roc.plot(fpr, tpr, linewidth=2,
                label=f"ROC-AUC = {final_test_metrics['roc_auc']:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")

    ax_pr.plot(recall_curve, precision_curve, linewidth=2,
               label=f"PR-AUC = {final_test_metrics['pr_auc']:.3f}")
    ax_pr.axhline(y_test.mean(), linestyle="--", color="grey", linewidth=1,
                  label=f"Baseline = {y_test.mean():.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curve")
    ax_pr.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

except Exception as e:
    raise Exception(f"[ROC/PR Curves] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 11. Interpretability

# COMMAND ----------

# DBTITLE 1,Feature importance
# Tree-based feature importance (gain) — quick global view of what drives predictions.
try:
    feature_importance = pd.DataFrame({
        "feature":    feature_columns,
        "importance": final_estimator.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fig_fi, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feature_importance["feature"][::-1],
            feature_importance["importance"][::-1],
            color="#4C72B0", edgecolor="white")
    for i, value in enumerate(feature_importance["importance"][::-1]):
        ax.text(value + max(feature_importance["importance"]) * 0.01, i,
                f"{value:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"{WINNER_ALGORITHM} — feature importance")
    plt.tight_layout()
    plt.show()

    display(feature_importance)

except Exception as e:
    raise Exception(f"[Feature Importance] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 12. Register as @candidate
# MAGIC
# MAGIC Always registers a new version and moves the `@candidate` alias to it.
# MAGIC Promotion to `@champion` is decided in `ml_promote_cardio_classifier.py`.

# COMMAND ----------

# DBTITLE 1,Build pyfunc artifact
# Wrap the fitted estimator with CardioModel (embeds threshold) and infer the input/output signature.
try:
    wrapped_model = CardioModel(final_estimator, optimal_threshold)

    sample_input    = X_train.head(5)
    sample_output   = wrapped_model.predict(None, sample_input)
    model_signature = infer_signature(sample_input, sample_output)

    pip_requirements = [
        f"xgboost=={xgb.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"mlflow=={mlflow.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
    ]

    print(f"Wrapped model ready ({WINNER_ALGORITHM}, threshold={optimal_threshold:.2f})")

except Exception as e:
    raise Exception(f"[Model Build] Failed to build pyfunc wrapper: {e}")

# COMMAND ----------

# DBTITLE 1,Log run, register and set @candidate
# Single MLflow run: tags + params + metrics + figures + tables + model, then register and set @candidate alias.
try:
    with mlflow.start_run(run_name=f"cardio_training_{WINNER_ALGORITHM.lower()}") as run:
        # Tags
        mlflow.set_tags({
            **RUN_TAGS,
            "winner.algorithm":  WINNER_ALGORITHM,
            "optimal.threshold": f"{optimal_threshold:.4f}",
        })

        # Parameters (ml_promote_cardio_classifier.py reads random_state/test_size/val_size
        # from these to reconstruct the same test split)
        mlflow.log_params({
            "winner_algorithm":  WINNER_ALGORITHM,
            "optimal_threshold": optimal_threshold,
            "n_optuna_trials":   N_OPTUNA_TRIALS,
            "cv_folds":          CV_FOLDS,
            "test_size":         TEST_SIZE,
            "val_size":          VAL_SIZE,
            "random_state":      RANDOM_STATE,
            "source_table":      FULL_SOURCE,
            "target_column":     TARGET_COLUMN,
            "feature_count":     len(feature_columns),
            "train_rows":        len(X_train),
            "validation_rows":   len(X_val),
            "test_rows":         len(X_test),
            **{f"winner_{k}": v for k, v in winner_best_params.items()},
        })

        # Final metrics on test
        log_metric_dict(final_test_metrics, prefix="test")

        # CV metrics
        mlflow.log_metric("cv_baseline_xgb_mean", cv_baseline_xgb_mean)
        mlflow.log_metric("cv_baseline_xgb_std",  cv_baseline_xgb_std)
        mlflow.log_metric("cv_baseline_rf_mean",  cv_baseline_rf_mean)
        mlflow.log_metric("cv_baseline_rf_std",   cv_baseline_rf_std)
        mlflow.log_metric("cv_tuned_xgb_mean",    cv_tuned_xgb_mean)
        mlflow.log_metric("cv_tuned_xgb_std",     cv_tuned_xgb_std)
        mlflow.log_metric("cv_tuned_rf_mean",     cv_tuned_rf_mean)
        mlflow.log_metric("cv_tuned_rf_std",      cv_tuned_rf_std)

        # Artifacts — log only the figures that were successfully created.
        # Any optional step (e.g. SHAP) that failed earlier won't block registration.
        _optional_figures = {
            "plots/threshold_sweep.png":    globals().get("fig_threshold"),
            "plots/confusion_matrix.png":   globals().get("fig_cm"),
            "plots/roc_pr_curves.png":      globals().get("fig_curves"),
            "plots/feature_importance.png": globals().get("fig_fi"),
            "plots/shap_summary.png":       globals().get("fig_shap_summary"),
        }
        for path, fig in _optional_figures.items():
            if fig is not None:
                mlflow.log_figure(fig, path)

        for i, fig_local in enumerate(globals().get("local_figs", []) or []):
            mlflow.log_figure(fig_local, f"plots/shap_local_patient_{i+1}.png")

        comparison_df.to_csv("/tmp/cv_comparison.csv", index=False)
        mlflow.log_artifact("/tmp/cv_comparison.csv", "tables")

        feature_importance.to_csv("/tmp/feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance.csv", "tables")

        threshold_results.to_csv("/tmp/threshold_sweep.csv", index=False)
        mlflow.log_artifact("/tmp/threshold_sweep.csv", "tables")

        # Log the model artifact
        mlflow.pyfunc.log_model(
            artifact_path=    "model",
            python_model=     wrapped_model,
            signature=        model_signature,
            input_example=    sample_input,
            pip_requirements= pip_requirements,
        )
        run_id    = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"Model logged to run {run_id}")

        # Register and set @candidate alias
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )
        mlflow_client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=CANDIDATE_ALIAS,
            version=registered.version,
        )

        # Version-level metadata (consumed by ml_promote_cardio_classifier.py)
        mlflow_client.update_model_version(
            name=REGISTERED_MODEL_NAME,
            version=registered.version,
            description=(
                f"{WINNER_ALGORITHM} candidate. "
                f"Test ROC-AUC: {final_test_metrics['roc_auc']:.4f}. "
                f"Optimal threshold: {optimal_threshold:.2f}. "
                f"Trained on {len(X_trainval_combined):,} rows from {FULL_SOURCE}."
            ),
        )
        mlflow_client.set_model_version_tag(REGISTERED_MODEL_NAME, registered.version,
                                            "algorithm", WINNER_ALGORITHM)
        mlflow_client.set_model_version_tag(REGISTERED_MODEL_NAME, registered.version,
                                            "optimal_threshold", f"{optimal_threshold:.6f}")
        mlflow_client.set_model_version_tag(REGISTERED_MODEL_NAME, registered.version,
                                            "pipeline_version", PIPELINE_VERSION)
        mlflow_client.set_model_version_tag(REGISTERED_MODEL_NAME, registered.version,
                                            "test_roc_auc", f"{final_test_metrics['roc_auc']:.6f}")
        mlflow_client.set_model_version_tag(REGISTERED_MODEL_NAME, registered.version,
                                            "training_run_id", run_id)

        print(f"\nREGISTERED: {REGISTERED_MODEL_NAME} version {registered.version} "
              f"is now @{CANDIDATE_ALIAS}")

        candidate_version = registered.version

except Exception as e:
    raise Exception(f"[Registry] Failed to log/register model: {e}")

# COMMAND ----------

# DBTITLE 1,Training summary
# Print a final summary block so the run is easy to interpret from the notebook output.
print("=" * 70)
print("TRAINING PIPELINE SUMMARY")
print("=" * 70)
print(f"Source:                  {FULL_SOURCE}")
print(f"Rows used:               {len(X):,}")
print(f"Winning algorithm:       {WINNER_ALGORITHM}")
print(f"Optimal threshold:       {optimal_threshold:.4f}")
print(f"Test ROC-AUC:            {final_test_metrics['roc_auc']:.4f}")
print(f"Test F1:                 {final_test_metrics['f1']:.4f}")
print(f"Test PR-AUC:             {final_test_metrics['pr_auc']:.4f}")
print("-" * 70)
print(f"Registered model:        {REGISTERED_MODEL_NAME} v{candidate_version}")
print(f"Candidate alias:         @{CANDIDATE_ALIAS} → v{candidate_version}")
print(f"MLflow run id:           {run_id}")
print("-" * 70)
print("NEXT STEP:")
print("  Run ml_promote_cardio_classifier.py to decide if this candidate becomes")
print("  the new @champion. The serving endpoint always tracks @champion only.")
print("=" * 70)
