# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
spark.sql(f"USE CATALOG `pf1`")

# COMMAND ----------

# DBTITLE 1,Import libraries
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code (copiado de gold, revisar y borrar)

# COMMAND ----------

# DBTITLE 1,Define source and target
sourceSchema = "silver"
sourceTable  = "cardio_silver_train"

schemaName = "gold"
tableName  = "cardio_gold_train"

# COMMAND ----------

# DBTITLE 1,Read from Gold table
df = spark.table(f"{sourceSchema}.{sourceTable}")

# COMMAND ----------

# DBTITLE 1,Feature engineering - BMI
df = df.withColumn(
    "bmi",
    F.round(F.col("weight") / F.pow(F.col("height") / 100.0, 2), 2)
)

# COMMAND ----------

# DBTITLE 1,Feature engineering - BMI category
df = df.withColumn(
    "bmi_category",
    F.when(F.col("bmi") < 18.5, F.lit(0))   # Underweight
     .when(F.col("bmi") < 25.0, F.lit(1))   # Normal
     .when(F.col("bmi") < 30.0, F.lit(2))   # Overweight
     .otherwise(F.lit(3))                    # Obese
)

# COMMAND ----------

# DBTITLE 1,Feature engineering - age group
df = df.withColumn(
    "age_group",
    F.when(F.col("age") < 40, F.lit(0))      # Under 40
     .when(F.col("age") < 50, F.lit(1))      # 40-49
     .when(F.col("age") < 60, F.lit(2))      # 50-59
     .otherwise(F.lit(3))                    # 60+
)

# COMMAND ----------

# DBTITLE 1,Feature engineering - blood pressure category
# Based on standard hypertension staging
df = df.withColumn(
    "bp_category",
    F.when((F.col("ap_hi") < 120) & (F.col("ap_lo") < 80),  F.lit(0))  # Normal
     .when((F.col("ap_hi") < 130) & (F.col("ap_lo") < 80),  F.lit(1))  # Elevated
     .when((F.col("ap_hi") < 140) | (F.col("ap_lo") < 90),  F.lit(2))  # Hypertension Stage 1
     .otherwise(F.lit(3))                                                # Hypertension Stage 2
)

# COMMAND ----------

# DBTITLE 1,Feature engineering - pulse pressure
df = df.withColumn("pulse_pressure", F.col("ap_hi") - F.col("ap_lo"))

# COMMAND ----------

# DBTITLE 1,Normalize continuous features (MinMax scaling)
continuous_features = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure"]

assembler = VectorAssembler(inputCols=continuous_features, outputCol="features_vec")
scaler    = MinMaxScaler(inputCol="features_vec", outputCol="features_scaled")

pipeline  = Pipeline(stages=[assembler, scaler])
model     = pipeline.fit(df)
df_scaled = model.transform(df)

# Unpack scaled vector back to individual columns
for i, col_name in enumerate(continuous_features):
    df_scaled = df_scaled.withColumn(
        f"{col_name}_scaled",
        F.udf(lambda v, idx=i: float(v[idx]) if v is not None else None, DoubleType())("features_scaled")
    )

df_scaled = df_scaled.drop("features_vec", "features_scaled")

# COMMAND ----------

# DBTITLE 1,Select final feature set for ML consumption
feature_columns = [
    "id",
    # Scaled continuous features
    "age_scaled", "height_scaled", "weight_scaled",
    "ap_hi_scaled", "ap_lo_scaled", "bmi_scaled", "pulse_pressure_scaled",
    # Original categorical features
    "gender", "cholesterol", "gluc", "smoke", "alco", "active",
    # Engineered features
    "bmi_category", "age_group", "bp_category",
    # Target variable
    "cardio",
    # Audit
    "ingestion_date", "source_table"
]

df_gold = df_scaled.select(feature_columns)

# COMMAND ----------

# DBTITLE 1,Class distribution of target variable
print("Cardio class distribution:")
display(
    df_gold.groupBy("cardio")
           .count()
           .withColumn("percentage", F.round(F.col("count") / df_gold.count() * 100, 2))
           .orderBy("cardio")
)

# COMMAND ----------

# DBTITLE 1,Feature importance preview - correlations with target
print("Correlation of each feature with target 'cardio':")
for col_name in [c for c in feature_columns if c not in ("id", "cardio", "ingestion_date", "source_table")]:
    try:
        corr_val = df_gold.stat.corr(col_name, "cardio")
        print(f"  {col_name:30s}  corr = {corr_val:.4f}")
    except Exception:
        pass

# COMMAND ----------

# DBTITLE 1,Write to Gold Delta table
(
    df_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{schemaName}.{tableName}")
)

#print(f"Gold table '{schemaName}.{tableName}' written successfully.")
#display(spark.table(f"{schemaName}.{tableName}").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and setup

# COMMAND ----------

# DBTITLE 1,Imports
import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,Data preparation
# ── Preparar datos ──────────────────────────────────────────────────────────
FEATURES = [
    "edad", "sexo", "tipo_dolor_pecho", "azucar_ayunas", "ecg_reposo",
    "presion_sistolica", "colesterol", "frecuencia_cardiaca_max",
    "depresion_st", "pendiente_st", "vasos_principales", "thal",
    "ratio_presion_fc", "colesterol_alto", "edad_grupo"
]

X = df[FEATURES].fillna(df[FEATURES].median())
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
        "schema":    "star_schema",
        "dataset":   "heart_disease_uci",
        "modelo":    "XGBoost",
        "version":   "1.0",
        "engineer":  "tu_nombre"
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
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="cardiovascular.models.xgboost_heart_disease"
        # ↑ Formato Unity Catalog: catalog.schema.model_name
    )

    print(f"\n✅ Run ID: {run.info.run_id}")
    print(f"🔗 UI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model registry

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "cardiovascular.models.xgboost_heart_disease"

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

# ── Cargar modelo por alias ─────────────────────────────────────────────────
champion = mlflow.xgboost.load_model(f"models:/{model_name}@champion")

# ── Predicción en nuevos pacientes (desde el star schema) ──────────────────
nuevos_pacientes = spark.sql("""
    SELECT p.edad, p.sexo, p.tipo_dolor_pecho, ...
    FROM cardiovascular.fact_examen f
    JOIN cardiovascular.dim_paciente p ON f.paciente_id = p.paciente_id
    WHERE f.fecha_id = 2
""").toPandas()

predicciones = champion.predict_proba(nuevos_pacientes[FEATURES])[:, 1]
nuevos_pacientes["riesgo_cardiovascular"] = predicciones
print(nuevos_pacientes[["edad", "sexo", "riesgo_cardiovascular"]].head(10))
