# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
spark.sql(f"USE CATALOG `pf1`")

# COMMAND ----------

# DBTITLE 1,Import libraries
import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and setup

# COMMAND ----------

# DBTITLE 1,Data preparation
# ── Read Gold feature table and convert to pandas ───────────────────────────
pdf = spark.table("gold.cardiofeatures").toPandas()

FEATURES = [
    "age_years", "age_group_id", "gender",
    "height_cm", "weight_kg", "bmi",
    "systolic_bp", "diastolic_bp", "pulse_pressure",
    "hypertension", "cholesterol", "gluc",
    "is_smoker", "drinks_alcohol", "is_physically_active",
]
TARGET = "cardio"

X = pdf[FEATURES].fillna(pdf[FEATURES].median())
y = pdf[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
print(f"Class balance (train) — 0: {(y_train==0).sum():,}  1: {(y_train==1).sum():,}")

# ── Parámetros del modelo ───────────────────────────────────────────────────
params = {
    "n_estimators":      300,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "scale_pos_weight":  (y_train == 0).sum() / (y_train == 1).sum(),
    "use_label_encoder": False,
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1
}

mlflow.xgboost.autolog()

# ── Entrenamiento con MLflow ────────────────────────────────────────────────
with mlflow.start_run(run_name="xgboost_cardiovascular_v1") as run:

    # --- Tags descriptivos ---
    mlflow.set_tags({
        "dataset":   "ocgn_cardio_train",
        "source":    "gold.cardiofeatures",
        "modelo":    "XGBoost",
        "version":   "1.0",
    })

    # --- Log de parámetros, no necesario con autolog ---
    # mlflow.log_params(params)
    # mlflow.log_param("features_count", len(FEATURES))
    # mlflow.log_param("train_size", len(X_train))
    # mlflow.log_param("test_size", len(X_test))

    # --- Entrenamiento ---
    model = xgb.XGBClassifier(**params)
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=50
    )

    # --- Métricas ---
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
    print("\n📊 Métricas:", metrics)

    # --- Cross-validation ---
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
    mlflow.log_metric("cv_roc_auc_std",  cv_scores.std())

    # --- Artefactos: Confusion Matrix ---
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sin afección", "Con afección"])
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix - Cardiovascular XGBoost")
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close()

    # --- Artefactos: Feature Importance ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(model, ax=ax2, max_num_features=15, importance_type="gain")
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    mlflow.log_figure(fig2, "feature_importance.png")
    plt.close()

    # --- Registrar modelo con firma ---
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    mlflow.xgboost.log_model(
        model,
        name="model",
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="pf1.models.xgboost_cardio"
        # ↑ Unity Catalog format: catalog.schema.model_name
    )

    print(f"\n✅ Run ID: {run.info.run_id}")
    print(f"🔗 UI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model registry

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "pf1.models.xgboost_cardio"

# ── Obtener la última versión registrada ────────────────────────────────────
latest = client.get_registered_model(model_name)
latest_version = client.get_latest_versions(model_name)[0].version

# ── Transición de alias (Unity Catalog usa aliases, no stages) ──────────────
client.set_registered_model_alias(
    name=model_name,
    alias="champion",          # "challenger", "staging", etc.
    version=latest_version
)

print(f"Modelo v{latest_version} promovido como 'champion'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------

# ── Load model by alias ─────────────────────────────────────────────────────
champion = mlflow.xgboost.load_model(f"models:/{model_name}@champion")

# ── Score new patients from the Gold feature table ───────────────────────────
nuevos_pacientes = spark.table("gold.cardiofeatures").toPandas()

predicciones = champion.predict_proba(nuevos_pacientes[FEATURES])[:, 1]
nuevos_pacientes["cardiovascular_risk"] = predicciones
print(nuevos_pacientes[["age_years", "gender", "cardiovascular_risk"]].head(10))
