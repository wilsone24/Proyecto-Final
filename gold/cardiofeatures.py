# Databricks notebook source
# DBTITLE 1,Catalog
spark.sql("USE CATALOG `databricks_service_pf`")

# COMMAND ----------

# DBTITLE 1,Libraries
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("source_schema", "gold")
dbutils.widgets.text("source_table",  "factcardio")
dbutils.widgets.text("target_schema", "gold")
dbutils.widgets.text("target_table",  "cardiofeatures")

# COMMAND ----------

# DBTITLE 1,Variables
SOURCE_SCHEMA = dbutils.widgets.get("source_schema")
SOURCE_TABLE  = dbutils.widgets.get("source_table")
TARGET_SCHEMA = dbutils.widgets.get("target_schema")
TARGET_TABLE  = dbutils.widgets.get("target_table")
FULL_SOURCE   = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
FULL_TARGET   = f"{TARGET_SCHEMA}.{TARGET_TABLE}"
PIPELINE_VER  = "1.0.0"
 
DROP_COLS = ["PatientId", "SilverIngestTime"]
 
RENAME_MAP = {
    "AgeYears":               "age_years",
    "IdAgeGroup":             "age_group_id",
    "IdGender":               "gender",
    "HeightCm":               "height_cm",
    "WeightKg":               "weight_kg",
    "BMI":                    "bmi",
    "SystolicBP":             "systolic_bp",
    "DiastolicBP":            "diastolic_bp",
    "PulsePressure":          "pulse_pressure",
    "HasHypertension":        "hypertension",
    "IdCholesterolType":      "cholesterol",
    "IdGlucoseType":          "gluc",
    "IsSmoker":               "is_smoker",
    "DrinksAlcohol":          "drinks_alcohol",
    "IsPhysicallyActive":     "is_physically_active",
    "HasCardiovascularDisease": "cardio",
}
 
BOOL_TO_INT_COLS = ["hypertension", "is_smoker", "drinks_alcohol", "is_physically_active", "cardio"]

# COMMAND ----------

# DBTITLE 1,Metadata
column_comments = {
    "age_years":            "Patient age in full years. Source: AgeYears.",
    "age_group_id":         "Age group FK: 1=<30, 2=30-44, 3=45-59, 4=60-74, 5=>=75. Source: IdAgeGroup.",
    "gender":               "Coded gender: 1=female, 2=male. Source: IdGender.",
    "height_cm":            "Height in centimetres. Source: HeightCm.",
    "weight_kg":            "Weight in kilograms. Source: WeightKg.",
    "bmi":                  "Body Mass Index = weight_kg / (height_cm/100)^2. Source: BMI.",
    "systolic_bp":          "Systolic blood pressure in mmHg. Source: SystolicBP.",
    "diastolic_bp":         "Diastolic blood pressure in mmHg. Source: DiastolicBP.",
    "pulse_pressure":       "Pulse pressure = systolic - diastolic. Source: PulsePressure.",
    "hypertension":         "1 if systolic >= 140 OR diastolic >= 90 (ACC/AHA 2017). Source: HasHypertension.",
    "cholesterol":          "Cholesterol level: 1=normal, 2=above normal, 3=well above normal. Source: IdCholesterolType.",
    "gluc":                 "Glucose level: 1=normal, 2=above normal, 3=well above normal. Source: IdGlucoseType.",
    "is_smoker":            "1 if patient smokes. Source: IsSmoker.",
    "drinks_alcohol":       "1 if patient drinks alcohol. Source: DrinksAlcohol.",
    "is_physically_active": "1 if patient is physically active. Source: IsPhysicallyActive.",
    "cardio":               "Target — 1=cardiovascular disease present, 0=absent. Source: HasCardiovascularDisease.",
}
 
table_comment = (
    "Gold layer — Feature table for XGBoost cardiovascular disease classifier. "
    "Derived from gold.factcardio: non-predictive columns dropped (PatientId, SilverIngestTime), "
    "columns renamed to snake_case ML convention, boolean fields cast to int. "
    "Overwritten on every pipeline run — not a historical table. "
    f"Source: pf1.{FULL_SOURCE}. Pipeline version: {PIPELINE_VER}."
)
 
table_properties = {
    "data.domain":           "health",
    "data.layer":            "gold",
    "data.source":           FULL_SOURCE,
    "data.owner":            "data-engineering",
    "data.pii":              "false",
    "data.contains.phi":     "true",
    "data.sensitivity":      "medium",
    "data.subject":          "cardiovascular-disease",
    "data.purpose":          "ml-features",
    "pipeline.version":      PIPELINE_VER,
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact":   "true",
}

# COMMAND ----------

# DBTITLE 1,Read source
try:
    fact_df    = spark.table(FULL_SOURCE)
    source_count = fact_df.count()
 
    if source_count == 0:
        raise Exception(f"Source table {FULL_SOURCE} returned 0 rows.")
 
except Exception as e:
    raise Exception(f"[Extract] Failed to read {FULL_SOURCE}: {e}")

# COMMAND ----------

# DBTITLE 1,Build feature table
try:
    features_df = fact_df.drop(*DROP_COLS)
 
    for original, renamed in RENAME_MAP.items():
        features_df = features_df.withColumnRenamed(original, renamed)
 
    for col in BOOL_TO_INT_COLS:
        features_df = features_df.withColumn(col, F.col(col).cast("int"))
 
except Exception as e:
    raise Exception(f"[Transform] Feature table build failed: {e}")

# COMMAND ----------

# DBTITLE 1,Write feature table (overwrite)
try:
    (
        features_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FULL_TARGET)
    )
 
except Exception as e:
    raise Exception(f"[Write] Failed to write feature table {FULL_TARGET}: {e}")

# COMMAND ----------

# DBTITLE 1,Validate row count
actual_count = spark.table(FULL_TARGET).count()
 
if actual_count != source_count:
    raise Exception(
        f"[Validation] Row count mismatch — source: {source_count}, target: {actual_count}"
    )
 
print(f"Row count OK: {actual_count:,} rows written to {FULL_TARGET}")

# COMMAND ----------

# DBTITLE 1,Apply  Metadata
try:
    for column, comment in column_comments.items():
        safe_comment = comment.replace("'", "\\'")
        spark.sql(
            f"""
                ALTER TABLE {FULL_TARGET}
                ALTER COLUMN `{column}` COMMENT '{safe_comment}'
            """
        )
 
except Exception as e:
    raise Exception(f"[Column Comments] Failed to apply comments on {FULL_TARGET}: {e}")

try:
    safe_table_comment = table_comment.replace("'", "\\'")
    spark.sql(f"COMMENT ON TABLE {FULL_TARGET} IS '{safe_table_comment}'")
 
except Exception as e:
    raise Exception(f"[Table Comment] Failed to apply table comment on {FULL_TARGET}: {e}")

try:
    props_ddl = ", ".join([f"'{k}' = '{v}'" for k, v in table_properties.items()])
    spark.sql(
        f"""
            ALTER TABLE {FULL_TARGET}
            SET TBLPROPERTIES (
                {props_ddl}
            )
        """
    )
 
except Exception as e:
    raise Exception(f"[Table Properties] Failed to apply properties on {FULL_TARGET}: {e}")

