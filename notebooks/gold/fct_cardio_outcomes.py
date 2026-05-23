# Databricks notebook source
# DBTITLE 1,Catalog
# Set the active Unity Catalog for all subsequent table references.
spark.sql("USE CATALOG `databricks_service_pf`")

# COMMAND ----------

# DBTITLE 1,Libraries
# Spark APIs for transformations and Python datetime for the pipeline-run timestamp.
from pyspark.sql import functions as F
from datetime import datetime, timezone

# COMMAND ----------

# DBTITLE 1,Parameters
# Job-level inputs: source silver table and target gold fact table.
dbutils.widgets.text("source_schema", "silver")
dbutils.widgets.text("source_table",  "cardio_patients")
dbutils.widgets.text("target_schema", "gold")
dbutils.widgets.text("target_table",  "fct_cardio_outcomes")

# COMMAND ----------

# DBTITLE 1,Constants
# Validate widgets and derive runtime constants (tables, year filter, column mapping).
SOURCE_SCHEMA = dbutils.widgets.get("source_schema")
SOURCE_TABLE  = dbutils.widgets.get("source_table")
TARGET_SCHEMA = dbutils.widgets.get("target_schema")
TARGET_TABLE  = dbutils.widgets.get("target_table")

# Fail fast if any required widget was not provided
if not all([SOURCE_SCHEMA, SOURCE_TABLE, TARGET_SCHEMA, TARGET_TABLE]):
    raise ValueError(
        f"Missing required widgets: source_schema='{SOURCE_SCHEMA}', "
        f"source_table='{SOURCE_TABLE}', target_schema='{TARGET_SCHEMA}', "
        f"target_table='{TARGET_TABLE}'"
    )

FULL_SOURCE      = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
FULL_TARGET      = f"{TARGET_SCHEMA}.{TARGET_TABLE}"
PIPELINE_VERSION = "1.0.0"

# Filter window: how many years back to keep in the fact (relative to current year).
YEARS_BACK = 2

# Filter the fact to records ingested in the last `YEARS_BACK + 1` years.
# Computed dynamically so the table always reflects the rolling window.
CURRENT_YEAR = datetime.now(timezone.utc).year
YEAR_FILTER  = CURRENT_YEAR - YEARS_BACK

# Source-to-target column mapping (silver snake_case → gold PascalCase business names).
# Note: omitted columns (is_current, _sourceFileName, _sourceFileModificationTime) are
# intentionally left out of the fact — they are silver/bronze operational metadata.
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

# DBTITLE 1,Helpers
# Reusable utilities for safe DDL execution and SQL string escaping.
def execute_sql_safely(label: str, query: str) -> None:
    """Execute a single SQL statement and re-raise with a labelled error message.

    Wraps spark.sql() so callers don't have to repeat the same try/except
    pattern for every DDL statement they emit.

    Args:
        label: Short context tag used in the error message (e.g. "Column Comments").
            Helps locate the failing section when reading logs.
        query: Full SQL statement to execute. Must be a single statement.

    Raises:
        Exception: Re-raises any error from spark.sql(), prefixed with the label
            and the target table name.
    """
    try:
        spark.sql(query)
    except Exception as e:
        raise Exception(f"[{label}] Failed on {FULL_TARGET}: {e}")


def escape_sql_string(s: str) -> str:
    """Escape single quotes so a string can be embedded inside a SQL literal.

    Args:
        s: Raw string that may contain unescaped single quotes.

    Returns:
        The same string with every "'" replaced by "\\'", safe to use inside
        single-quoted SQL string literals.
    """
    return s.replace("'", "\\'")

# COMMAND ----------

# DBTITLE 1,Metadata
# Column comments, table comment and Unity Catalog properties applied after the write.
column_comments = {
    "PatientId":                "Unique patient identifier from the source dataset.",
    "AgeYears":                 "Patient age in full years (age / 365.25), rounded to 1 decimal.",
    "IdAgeGroup":               "Age group FK: 1=<30, 2=30-44, 3=45-59, 4=60-74, 5=>=75. Join with gold.dim_age_group.",
    "IdGender":                 "Coded gender: 1=female, 2=male. Join with gold.dim_gender.",
    "HeightCm":                 "Height in centimetres.",
    "WeightKg":                 "Weight in kilograms.",
    "BMI":                      "Body Mass Index = weight_kg / (height_cm/100)^2.",
    "SystolicBP":               "Systolic blood pressure in mmHg.",
    "DiastolicBP":              "Diastolic blood pressure in mmHg.",
    "PulsePressure":            "Pulse pressure = systolic - diastolic.",
    "HasHypertension":          "True when systolic >= 140 OR diastolic >= 90 (ACC/AHA 2017 criteria).",
    "IdCholesterolType":        "Cholesterol level: 1=normal, 2=above normal, 3=well above normal. Join with gold.dim_cholesterol.",
    "IdGlucoseType":            "Glucose level: 1=normal, 2=above normal, 3=well above normal. Join with gold.dim_glucose.",
    "IsSmoker":                 "True if patient smokes. Self-reported.",
    "DrinksAlcohol":            "True if patient drinks alcohol. Self-reported.",
    "IsPhysicallyActive":       "True if patient is physically active. Self-reported.",
    "HasCardiovascularDisease": "Target — true=cardiovascular disease present, false=absent.",
    "SilverIngestTime":         "UTC timestamp when this record was written to the silver layer.",
}

table_comment = (
    "Gold layer — Cardiovascular fact table. "
    "Contains only current SCD-2 records (is_current=TRUE) "
    f"ingested in the last {YEARS_BACK + 1} years (>= {YEAR_FILTER}). "
    "Overwritten on every pipeline run. "
    f"Source: {FULL_SOURCE}. Pipeline version: {PIPELINE_VERSION}."
)

table_properties = {
    "data.domain":       "health",
    "data.layer":        "gold",
    "data.source":       FULL_SOURCE,
    "data.owner":        "data-engineering",
    "data.pii":          "true",
    "data.contains.phi": "true",
    "data.sensitivity":  "medium",
    "data.subject":      "cardiovascular-disease",
    "pipeline.version":  PIPELINE_VERSION,
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact":   "true",
}

# COMMAND ----------

# DBTITLE 1,Read silver table
# Load source silver table and abort early if it returned no rows.
try:
    silver_df    = spark.table(FULL_SOURCE)
    source_count = silver_df.count()

    if source_count == 0:
        raise Exception(f"Source table {FULL_SOURCE} returned 0 rows.")

except Exception as e:
    raise Exception(f"[Extract] Failed to read {FULL_SOURCE}: {e}")

# COMMAND ----------

# DBTITLE 1,Filter and project to fact schema
# Keep only current SCD-2 records within the rolling year window, then rename to business names.
try:
    filtered_df = (
        silver_df
        .filter(F.col("is_current") == True)
        .filter(F.year(F.col("_silverIngestTime")) >= YEAR_FILTER)
    )

    fact_df = filtered_df.select(
        [F.col(source).alias(target) for source, target in SELECT_MAP.items()]
    )

    expected_count = filtered_df.count()

except Exception as e:
    raise Exception(f"[Transform] Filter and column mapping failed: {e}")

# COMMAND ----------

# DBTITLE 1,Overwrite fact table
# Atomic full-refresh write — the fact is a snapshot, not an SCD table.
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
# Confirm the write produced exactly the expected number of rows.
actual_count = spark.table(FULL_TARGET).count()

if actual_count != expected_count:
    raise Exception(
        f"[Validation] Row count mismatch — "
        f"expected: {expected_count}, actual: {actual_count}"
    )

# COMMAND ----------

# DBTITLE 1,Add metadata to the target
# Apply column COMMENTs, table COMMENT and TBLPROPERTIES via execute_sql_safely().

# Column comments
for column, comment in column_comments.items():
    safe_comment = escape_sql_string(comment)
    execute_sql_safely(
        "Column Comments",
        f"ALTER TABLE {FULL_TARGET} ALTER COLUMN `{column}` COMMENT '{safe_comment}'",
    )

# Table comment
safe_table_comment = escape_sql_string(table_comment)
execute_sql_safely(
    "Table Comment",
    f"COMMENT ON TABLE {FULL_TARGET} IS '{safe_table_comment}'",
)

# Table properties
props_ddl = ", ".join([f"'{k}' = '{v}'" for k, v in table_properties.items()])
execute_sql_safely(
    "Table Properties",
    f"ALTER TABLE {FULL_TARGET} SET TBLPROPERTIES ({props_ddl})",
)
