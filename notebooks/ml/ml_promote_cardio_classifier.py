# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Cardiovascular Disease — Model Promotion
# MAGIC
# MAGIC Compares `@candidate` vs `@champion` on the same test split; promotes the candidate if it wins.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# Install ML libraries required to load the candidate / champion pyfunc models.
%pip install xgboost

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
# sklearn for metrics + reconstructing the same train/test split, MLflow for model loading and registry ops.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# COMMAND ----------

# DBTITLE 1,Parameters
# Job-level inputs: target model and MLflow experiment (only fields that change between dev/prod).
dbutils.widgets.text("registered_model_name",  "databricks_service_pf.gold.cardio_classifier")
dbutils.widgets.text("mlflow_experiment_path", "/Shared/cardio_ml")

# COMMAND ----------

# DBTITLE 1,Constants
# Validate widgets and derive runtime constants (registry conventions, comparison policy, run tags).
REGISTERED_MODEL_NAME  = dbutils.widgets.get("registered_model_name")
MLFLOW_EXPERIMENT_PATH = dbutils.widgets.get("mlflow_experiment_path")

# Fail fast if any required widget was not provided
if not all([REGISTERED_MODEL_NAME, MLFLOW_EXPERIMENT_PATH]):
    raise ValueError(
        f"Missing required widgets: registered_model_name='{REGISTERED_MODEL_NAME}', "
        f"mlflow_experiment_path='{MLFLOW_EXPERIMENT_PATH}'"
    )

# Registry conventions (MLflow Model Registry aliases — not environment-specific)
CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS  = "champion"

# Promotion policy
COMPARISON_METRIC = "roc_auc"   # metric used to compare candidate vs champion
MIN_IMPROVEMENT   = 0.0         # candidate must beat champion by strictly more than this

# Features excluded from training — must match ml_train_cardio_classifier.py.
# Derived features that cause multicollinearity / information leakage:
#   - hypertension: ap_hi >= 140 OR ap_lo >= 90 (function of systolic_bp + diastolic_bp)
#   - pulse_pressure: systolic_bp - diastolic_bp
#   - age_group_id: bucket of age_years
EXCLUDED_FEATURES = ["hypertension", "pulse_pressure", "age_group_id"]

# Pipeline metadata
PIPELINE_VERSION = "1.0.0"

RUN_TAGS = {
    "pipeline.version":  PIPELINE_VERSION,
    "data.subject":      "cardiovascular-disease",
    "data.layer":        "gold",
    "stage":             "promotion",
    "comparison.metric": COMPARISON_METRIC,
    "min.improvement":   f"{MIN_IMPROVEMENT:.4f}",
}

print(f"Registered model:    {REGISTERED_MODEL_NAME}")
print(f"Candidate alias:     @{CANDIDATE_ALIAS}")
print(f"Champion alias:      @{CHAMPION_ALIAS}")
print(f"Comparison metric:   {COMPARISON_METRIC}")
print(f"Minimum improvement: {MIN_IMPROVEMENT} (strict > if 0)")

# COMMAND ----------

# DBTITLE 1,MLflow configuration
# Point MLflow at the Unity Catalog model registry and the shared experiment path.
try:
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
    mlflow_client = MlflowClient()
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_PATH}")

except Exception as e:
    raise Exception(f"[MLflow Setup] Failed to configure MLflow: {e}")

# COMMAND ----------

# DBTITLE 1,Helper functions
# Reusable utilities for metrics, probability extraction and version-metadata lookup.
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


def get_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Extract positive-class probabilities from a pyfunc-loaded model.

    The CardioModel wrapper returns a DataFrame with `probability` and
    `prediction` columns; older or different wrappers may return a plain array.
    This helper handles both cases.

    Args:
        model: An MLflow pyfunc-loaded model (output of mlflow.pyfunc.load_model).
        X: Feature matrix to score.

    Returns:
        1-D numpy array of positive-class probabilities, one per row of X.
    """
    output = model.predict(X)
    if isinstance(output, pd.DataFrame):
        if "probability" in output.columns:
            return output["probability"].values.astype(float)
        return output.iloc[:, 0].values.astype(float)
    return np.asarray(output, dtype=float).ravel()


def read_version_metadata(model_name: str, version: str) -> dict:
    """Pull the metadata ml_train_cardio_classifier.py wrote at registration time.

    Args:
        model_name: Fully-qualified Unity Catalog model name.
        version: Model version number (as string or int).

    Returns:
        Dict with keys: version, run_id, description, algorithm, optimal_threshold,
        test_roc_auc, training_run_id, pipeline_version. Missing tags fall back to
        sensible defaults (e.g. NaN for test_roc_auc, 0.5 for optimal_threshold).
    """
    info = mlflow_client.get_model_version(model_name, version)
    tags = dict(info.tags) if info.tags else {}
    return {
        "version":           info.version,
        "run_id":            info.run_id,
        "description":       info.description or "",
        "algorithm":         tags.get("algorithm",        "unknown"),
        "optimal_threshold": float(tags.get("optimal_threshold", 0.5)),
        "test_roc_auc":      float(tags.get("test_roc_auc", "nan")) if tags.get("test_roc_auc") else float("nan"),
        "training_run_id":   tags.get("training_run_id",  info.run_id),
        "pipeline_version":  tags.get("pipeline_version", "unknown"),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Resolve candidate and champion versions

# COMMAND ----------

# DBTITLE 1,Resolve @candidate (must exist)
# The promotion notebook requires a candidate — if missing, training never ran successfully.
try:
    candidate_info = mlflow_client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CANDIDATE_ALIAS,
    )
    candidate_meta = read_version_metadata(REGISTERED_MODEL_NAME, candidate_info.version)
    print(f"Candidate resolved: v{candidate_meta['version']} "
          f"({candidate_meta['algorithm']}, threshold={candidate_meta['optimal_threshold']:.2f})")

except mlflow.exceptions.RestException as e:
    raise Exception(
        f"[Candidate] No @{CANDIDATE_ALIAS} alias found on {REGISTERED_MODEL_NAME}. "
        f"Run ml_train_cardio_classifier.py first to produce a candidate. Details: {e}"
    )

except Exception as e:
    raise Exception(f"[Candidate] Failed to resolve candidate alias: {e}")

# COMMAND ----------

# DBTITLE 1,Resolve @champion (may not exist → cold start)
# Missing champion is the COLD START path — the candidate is promoted unconditionally.
champion_meta = None
is_cold_start = False

try:
    champion_info = mlflow_client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS,
    )
    champion_meta = read_version_metadata(REGISTERED_MODEL_NAME, champion_info.version)
    print(f"Champion resolved:  v{champion_meta['version']} "
          f"({champion_meta['algorithm']}, threshold={champion_meta['optimal_threshold']:.2f})")

    if str(champion_meta["version"]) == str(candidate_meta["version"]):
        print(f"\nNote: @{CANDIDATE_ALIAS} and @{CHAMPION_ALIAS} already point "
              f"to the same version (v{candidate_meta['version']}). Nothing to do.")

except mlflow.exceptions.RestException:
    is_cold_start = True
    print(f"No @{CHAMPION_ALIAS} alias yet — this is a COLD START.")

except Exception as e:
    raise Exception(f"[Champion] Failed to resolve champion alias: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Reconstruct the candidate's test split
# MAGIC
# MAGIC We read `random_state`, `test_size`, `val_size`, `source_table` and
# MAGIC `target_column` from the candidate's training run and replay the same
# MAGIC stratified split. Because the seed and parameters are identical, the
# MAGIC reconstructed test set is byte-equal to the one ml_train_cardio_classifier.py
# MAGIC evaluated on.

# COMMAND ----------

# DBTITLE 1,Read training params from candidate's run
# Pull the exact source table, target, split sizes and seed used by training.
try:
    training_run = mlflow_client.get_run(candidate_meta["training_run_id"])
    train_params = training_run.data.params

    SOURCE_TABLE_FROM_RUN = train_params["source_table"]
    TARGET_COLUMN         = train_params["target_column"]
    TEST_SIZE             = float(train_params["test_size"])
    VAL_SIZE              = float(train_params["val_size"])
    RANDOM_STATE          = int(train_params["random_state"])

    print(f"Source table:   {SOURCE_TABLE_FROM_RUN}")
    print(f"Target column:  {TARGET_COLUMN}")
    print(f"Test size:      {TEST_SIZE}")
    print(f"Val size:       {VAL_SIZE}")
    print(f"Random state:   {RANDOM_STATE}")

except Exception as e:
    raise Exception(f"[Training Params] Failed to read params from candidate run: {e}")

# COMMAND ----------

# DBTITLE 1,Load source and rebuild test set
# Replay the first split (test vs trainval) — that's enough since both models are evaluated on test.
try:
    features_df = spark.table(SOURCE_TABLE_FROM_RUN).toPandas()

    # Mirror the exclusion applied during training so X has the same columns
    # the candidate model expects (otherwise predict() fails on schema mismatch).
    feature_columns = [
        c for c in features_df.columns
        if c != TARGET_COLUMN and c not in EXCLUDED_FEATURES
    ]
    X = features_df[feature_columns].copy()
    y = features_df[TARGET_COLUMN].astype(int).copy()

    # Replay exact same split as ml_train_cardio_classifier.py
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Reconstructed test set: {len(X_test):,} rows ({y_test.mean():.4f} cardio rate)")

except Exception as e:
    raise Exception(f"[Test Reconstruction] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Evaluate candidate and champion on the test set

# COMMAND ----------

# DBTITLE 1,Evaluate candidate
# Load the candidate via its alias and score the reconstructed test set.
try:
    candidate_uri     = f"models:/{REGISTERED_MODEL_NAME}@{CANDIDATE_ALIAS}"
    candidate_model   = mlflow.pyfunc.load_model(candidate_uri)
    candidate_proba   = get_probabilities(candidate_model, X_test)
    candidate_metrics = compute_metrics(
        y_test, candidate_proba, threshold=candidate_meta["optimal_threshold"],
    )

    print(f"Candidate (v{candidate_meta['version']}) test metrics:")
    for k, v in candidate_metrics.items():
        print(f"  {k:<10s}: {v:.4f}")

except Exception as e:
    raise Exception(f"[Candidate Evaluation] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,Evaluate champion (if exists)
# Skip silently on cold start; otherwise score the champion on the same test set.
champion_metrics = None
champion_proba   = None

if not is_cold_start:
    try:
        champion_uri     = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
        champion_model   = mlflow.pyfunc.load_model(champion_uri)
        champion_proba   = get_probabilities(champion_model, X_test)
        champion_metrics = compute_metrics(
            y_test, champion_proba, threshold=champion_meta["optimal_threshold"],
        )

        print(f"Champion (v{champion_meta['version']}) test metrics:")
        for k, v in champion_metrics.items():
            print(f"  {k:<10s}: {v:.4f}")

    except Exception as e:
        raise Exception(f"[Champion Evaluation] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,Side-by-side comparison
# Build a comparison DataFrame with champion / candidate / delta for every metric.
try:
    if champion_metrics is not None:
        rows = []
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            rows.append({
                "metric":    metric,
                "champion":  champion_metrics[metric],
                "candidate": candidate_metrics[metric],
                "delta":     candidate_metrics[metric] - champion_metrics[metric],
            })
        comparison_df = pd.DataFrame(rows)
    else:
        comparison_df = pd.DataFrame([
            {"metric": m, "champion": None,
             "candidate": v, "delta": None}
            for m, v in candidate_metrics.items()
        ])

    display(comparison_df)

except Exception as e:
    raise Exception(f"[Comparison Table] Failed: {e}")

# COMMAND ----------

# DBTITLE 1,Comparison plot (ROC curves side by side)
# Overlay the ROC curves of champion and candidate on the same axes — visual sanity check.
fig_compare = None
try:
    if champion_metrics is not None:
        fpr_c, tpr_c, _ = roc_curve(y_test, candidate_proba)
        fpr_h, tpr_h, _ = roc_curve(y_test, champion_proba)

        fig_compare, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr_h, tpr_h, linewidth=2, color="#888888",
                label=f"Champion v{champion_meta['version']} "
                      f"(AUC = {champion_metrics['roc_auc']:.4f})")
        ax.plot(fpr_c, tpr_c, linewidth=2, color="#2E86C1",
                label=f"Candidate v{candidate_meta['version']} "
                      f"(AUC = {candidate_metrics['roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Champion vs Candidate — ROC on test set")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

except Exception as e:
    raise Exception(f"[Comparison Plot] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Decision logic

# COMMAND ----------

# DBTITLE 1,Decide whether to promote
# Four branches: cold_start → promote, same version → skip, beats threshold → promote, else keep champion.
candidate_score = float(candidate_metrics[COMPARISON_METRIC])
champion_score  = float(champion_metrics[COMPARISON_METRIC]) if champion_metrics else None

if is_cold_start:
    should_promote  = True
    decision_reason = "cold_start"
    delta_score     = None
elif str(champion_meta["version"]) == str(candidate_meta["version"]):
    should_promote  = False
    decision_reason = "same_version"
    delta_score     = 0.0
else:
    delta_score = candidate_score - champion_score
    if delta_score > MIN_IMPROVEMENT:
        should_promote  = True
        decision_reason = "beats_champion"
    else:
        should_promote  = False
        decision_reason = "below_threshold"

print("=" * 60)
print(f"DECISION: {'PROMOTE' if should_promote else 'KEEP CHAMPION'}")
print(f"Reason:   {decision_reason}")
print("-" * 60)
print(f"Candidate {COMPARISON_METRIC}: {candidate_score:.6f}")
if champion_score is not None:
    print(f"Champion  {COMPARISON_METRIC}: {champion_score:.6f}")
    print(f"Delta:                  {delta_score:+.6f}")
    print(f"Min improvement needed: {MIN_IMPROVEMENT:+.6f}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Apply decision and log the run

# COMMAND ----------

# DBTITLE 1,Move @champion alias if promoted
# Single MLflow run: log tags + params + metrics + comparison artifacts, then apply the alias move.
try:
    with mlflow.start_run(run_name=f"cardio_promotion_v{candidate_meta['version']}") as run:
        # Tags
        mlflow.set_tags({
            **RUN_TAGS,
            "decision":                  "promote" if should_promote else "keep_champion",
            "decision.reason":           decision_reason,
            "candidate.version":         str(candidate_meta["version"]),
            "candidate.algorithm":       candidate_meta["algorithm"],
            "champion.version.previous": str(champion_meta["version"]) if champion_meta else "none",
            "cold.start":                str(is_cold_start).lower(),
        })

        # Parameters
        mlflow.log_params({
            "registered_model_name": REGISTERED_MODEL_NAME,
            "candidate_alias":       CANDIDATE_ALIAS,
            "champion_alias":        CHAMPION_ALIAS,
            "candidate_version":     candidate_meta["version"],
            "champion_version_prev": champion_meta["version"] if champion_meta else "none",
            "candidate_threshold":   candidate_meta["optimal_threshold"],
            "champion_threshold":    champion_meta["optimal_threshold"] if champion_meta else "none",
            "comparison_metric":     COMPARISON_METRIC,
            "min_improvement":       MIN_IMPROVEMENT,
            "test_rows":             len(X_test),
        })

        # Metrics
        for name, value in candidate_metrics.items():
            mlflow.log_metric(f"candidate_{name}", float(value))
        if champion_metrics is not None:
            for name, value in champion_metrics.items():
                mlflow.log_metric(f"champion_{name}", float(value))
            mlflow.log_metric("delta_score", float(delta_score))

        # Artifacts
        comparison_df.to_csv("/tmp/promotion_comparison.csv", index=False)
        mlflow.log_artifact("/tmp/promotion_comparison.csv", "tables")
        if fig_compare is not None:
            mlflow.log_figure(fig_compare, "plots/roc_comparison.png")

        promotion_run_id = run.info.run_id
        print(f"Promotion run logged: {promotion_run_id}")

        # Apply the decision
        if should_promote:
            mlflow_client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias=CHAMPION_ALIAS,
                version=candidate_meta["version"],
            )
            mlflow_client.set_model_version_tag(
                REGISTERED_MODEL_NAME, candidate_meta["version"],
                "promoted_at_run", promotion_run_id,
            )
            mlflow_client.set_model_version_tag(
                REGISTERED_MODEL_NAME, candidate_meta["version"],
                "champion_since", str(pd.Timestamp.utcnow().isoformat()),
            )

            if is_cold_start:
                print(f"\nPROMOTED (cold start): v{candidate_meta['version']} "
                      f"is now @{CHAMPION_ALIAS}.")
            else:
                print(f"\nPROMOTED: v{candidate_meta['version']} is now "
                      f"@{CHAMPION_ALIAS}. Previous champion v{champion_meta['version']} "
                      f"is unaliased but remains in the registry for audit/rollback.")

        else:
            print(f"\nNOT PROMOTED: candidate v{candidate_meta['version']} did "
                  f"not beat champion v{champion_meta['version']}. "
                  f"Champion alias remains on v{champion_meta['version']}.")

except Exception as e:
    raise Exception(f"[Promotion] Failed to log/apply decision: {e}")

# COMMAND ----------

# DBTITLE 1,Promotion summary
# Final summary block so the run is easy to interpret from the notebook output.
print("=" * 70)
print("PROMOTION SUMMARY")
print("=" * 70)
print(f"Registered model:       {REGISTERED_MODEL_NAME}")
print(f"Comparison metric:      {COMPARISON_METRIC}")
print("-" * 70)
print(f"Candidate version:      v{candidate_meta['version']} ({candidate_meta['algorithm']})")
print(f"Candidate {COMPARISON_METRIC}:       {candidate_score:.6f}")
if champion_score is not None:
    print(f"Previous champion:      v{champion_meta['version']} ({champion_meta['algorithm']})")
    print(f"Previous {COMPARISON_METRIC}:        {champion_score:.6f}")
    print(f"Delta:                  {delta_score:+.6f}")
print("-" * 70)
print(f"Decision:               {'PROMOTE' if should_promote else 'KEEP CHAMPION'}")
print(f"Reason:                 {decision_reason}")
if should_promote:
    print(f"Active @{CHAMPION_ALIAS}:        v{candidate_meta['version']}")
else:
    print(f"Active @{CHAMPION_ALIAS}:        v{champion_meta['version']} (unchanged)")
print(f"Promotion run id:       {promotion_run_id}")
print("=" * 70)
print()
print("NEXT STEP:")
if should_promote:
    print(f"  Re-run ml_serve_cardio_classifier.py to update the serving endpoint with the new")
    print(f"  champion version (v{candidate_meta['version']}).")
else:
    print(f"  No action needed. The serving endpoint already serves the active champion.")
