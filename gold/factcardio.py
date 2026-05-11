# Databricks notebook source
# DBTITLE 1,Catalog
spark.sql("USE CATALOG `pf1`")

# COMMAND ----------

# DBTITLE 1,Libraries
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("source_schema", "silver")
dbutils.widgets.text("source_table",  "cardiosilver")
dbutils.widgets.text("target_schema", "gold")
dbutils.widgets.text("target_table",  "factcardio")

# COMMAND ----------

# DBTITLE 1,Variables
SOURCE_SCHEMA = dbutils.widgets.get("source_schema")
SOURCE_TABLE  = dbutils.widgets.get("source_table")
TARGET_SCHEMA = dbutils.widgets.get("target_schema")
TARGET_TABLE  = dbutils.widgets.get("target_table")
FULL_SOURCE   = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
FULL_TARGET   = f"{TARGET_SCHEMA}.{TARGET_TABLE}"
PIPELINE_VER  = "1.0.0"
 
CURRENT_YEAR  = spark.sql("SELECT YEAR(CURRENT_DATE)").collect()[0][0]
YEAR_FILTER   = CURRENT_YEAR - 2
 
SELECT_MAP = {
    "id":                "PatientId",
    "age_years":         "AgeYears",
    "age_group_id":      "IdAgeGroup",
    "gender":            "IdGender",
    "height_cm":         "HeightCm",
    "weight_kg":         "WeightKg",
    "bmi":               "BMI",
    "ap_hi":             "SystolicBP",
    "ap_lo":             "DiastolicBP",
    "pulse_pressure":    "PulsePressure",
    "hypertension":      "HasHypertension",
    "cholesterol":       "IdCholesterolType",
    "gluc":              "IdGlucoseType",
    "smoke":             "IsSmoker",
    "alco":              "DrinksAlcohol",
    "active":            "IsPhysicallyActive",
    "cardio":            "HasCardiovascularDisease",
    "_silverIngestTime": "SilverIngestTime",
}

# COMMAND ----------

# DBTITLE 1,Metadata
column_comments = {
    "PatientId":               "Unique patient identifier from the source dataset.",
    "AgeYears":                "Patient age in full years (age / 365.25), rounded to 1 decimal.",
    "IdAgeGroup":              "Age group FK: 1=<30, 2=30-44, 3=45-59, 4=60-74, 5=>=75. Join with dim_age_group.",
    "IdGender":                "Coded gender: 1=female, 2=male. Join with gold.dimgender.",
    "HeightCm":                "Height in centimetres.",
    "WeightKg":                "Weight in kilograms.",
    "BMI":                     "Body Mass Index = weight_kg / (height_cm/100)^2.",
    "SystolicBP":              "Systolic blood pressure in mmHg.",
    "DiastolicBP":             "Diastolic blood pressure in mmHg.",
    "PulsePressure":           "Pulse pressure = systolic - diastolic.",
    "HasHypertension":         "True when systolic >= 140 OR diastolic >= 90 (ACC/AHA 2017 criteria).",
    "IdCholesterolType":       "Cholesterol level: 1=normal, 2=above normal, 3=well above normal. Join with gold.dimcholesterol.",
    "IdGlucoseType":           "Glucose level: 1=normal, 2=above normal, 3=well above normal. Join with gold.dimglucose.",
    "IsSmoker":                "True if patient smokes. Self-reported.",
    "DrinksAlcohol":           "True if patient drinks alcohol. Self-reported.",
    "IsPhysicallyActive":      "True if patient is physically active. Self-reported.",
    "HasCardiovascularDisease":"Target — true=cardiovascular disease present, false=absent.",
    "SilverIngestTime":        "Timestamp (COT) when this record was written to the silver layer.",
}
 
table_comment = (
    "Gold layer — Cardiovascular fact table. "
    "Contains only current SCD-2 records (is_current=TRUE) "
    f"from the last 3 years (>= {YEAR_FILTER}). "
    "Overwritten on every pipeline run. "
    f"Source: pf1.{FULL_SOURCE}. Pipeline version: {PIPELINE_VER}."
)
 
table_properties = {
    "data.domain":                       "health",
    "data.layer":                        "gold",
    "data.source":                       FULL_SOURCE,
    "data.owner":                        "data-engineering",
    "data.pii":                          "false",
    "data.contains.phi":                 "true",
    "data.sensitivity":                  "medium",
    "data.subject":                      "cardiovascular-disease",
    "pipeline.version":                  PIPELINE_VER,
    "delta.autoOptimize.optimizeWrite":  "true",
    "delta.autoOptimize.autoCompact":    "true",
}

# COMMAND ----------

# DBTITLE 1,Validate count
try:
    silver_df    = spark.table(FULL_SOURCE)
    source_count = silver_df.count()
 
    if source_count == 0:
        raise Exception(f"Source table {FULL_SOURCE} returned 0 rows.")
 
except Exception as e:
    raise Exception(f"[Extract] Failed to read {FULL_SOURCE}: {e}")

# COMMAND ----------

# DBTITLE 1,Filter fact table
try:
    filtered_df = (
        silver_df
        .filter(F.col("is_current") == True)
        .filter(F.year(F.col("_silverIngestTime")) >= YEAR_FILTER)
    )
 
    fact_df = filtered_df.select(
        [F.col(src).alias(tgt) for src, tgt in SELECT_MAP.items()]
    )
 
except Exception as e:
    raise Exception(f"[Transform] Filter and column mapping failed: {e}")

# COMMAND ----------

# DBTITLE 1,Overwrite table
try:
    (
        fact_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FULL_TARGET)
    )
 
except Exception as e:
    raise Exception(f"[Write] Failed to write fact table {FULL_TARGET}: {e}")

# COMMAND ----------

# DBTITLE 1,Row count validation
actual_count = spark.table(FULL_TARGET).count()
 
if actual_count != filtered_df.count():
    raise Exception(
        f"[Validation] Row count mismatch — "
        f"expected: {filtered_df.count()}, actual: {actual_count}"
    )

# COMMAND ----------

# DBTITLE 1,Apply Metadata
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
