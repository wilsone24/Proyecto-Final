# Databricks notebook source
# DBTITLE 1,Catalog
# Set the active Unity Catalog for all subsequent table references.
spark.sql("USE CATALOG `databricks_service_pf`")

# COMMAND ----------

# DBTITLE 1,Libraries
# Spark APIs for column transformations.
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Parameters
# Job-level inputs: source gold fact table and target ML feature table.
dbutils.widgets.text("source_schema", "gold")
dbutils.widgets.text("source_table",  "fct_cardio_outcomes")
dbutils.widgets.text("target_schema", "gold")
dbutils.widgets.text("target_table",  "cardio_features")

# COMMAND ----------

# DBTITLE 1,Constants
# Validate widgets and derive runtime constants (tables, drop list, rename map, boolean cast list).
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

# Columns dropped from the fact — non-predictive for ML (identifier and operational timestamp).
DROP_COLS = ["PatientId", "SilverIngestTime"]

# Fact (PascalCase business names) → feature table (snake_case ML convention).
RENAME_MAP = {
    "AgeYears":                 "age_years",
    "IdAgeGroup":               "age_group_id",
    "IdGender":                 "gender",
    "HeightCm":                 "height_cm",
    "WeightKg":                 "weight_kg",
    "BMI":                      "bmi",
    "SystolicBP":               "systolic_bp",
    "DiastolicBP":              "diastolic_bp",
    "PulsePressure":            "pulse_pressure",
    "HasHypertension":          "hypertension",
    "IdCholesterolType":        "cholesterol",
    "IdGlucoseType":            "gluc",
    "IsSmoker":                 "is_smoker",
    "DrinksAlcohol":            "drinks_alcohol",
    "IsPhysicallyActive":       "is_physically_active",
    "HasCardiovascularDisease": "cardio",
}

# Boolean columns cast to int because most ML libraries expect numerical inputs.
BOOL_TO_INT_COLS = ["hypertension", "is_smoker", "drinks_alcohol", "is_physically_active", "cardio"]

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
    "Gold layer — Feature table for the cardiovascular disease ML classifier. "
    "Derived from gold.fct_cardio_outcomes: non-predictive columns dropped (PatientId, SilverIngestTime), "
    "columns renamed to snake_case ML convention, boolean fields cast to int. "
    "De-identified — does not contain patient identifiers. "
    "Overwritten on every pipeline run — not a historical table. "
    f"Source: {FULL_SOURCE}. Pipeline version: {PIPELINE_VERSION}."
)

table_properties = {
    "data.domain":       "health",
    "data.layer":        "gold",
    "data.source":       FULL_SOURCE,
    "data.owner":        "data-engineering",
    "data.pii":          "false",
    "data.contains.phi": "true",
    "data.sensitivity":  "medium",
    "data.subject":      "cardiovascular-disease",
    "data.purpose":      "ml-features",
    "pipeline.version":  PIPELINE_VERSION,
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact":   "true",
}

# COMMAND ----------

# DBTITLE 1,Read source
# Load source fact table and abort early if it returned no rows.
try:
    fact_df      = spark.table(FULL_SOURCE)
    source_count = fact_df.count()

    if source_count == 0:
        raise Exception(f"Source table {FULL_SOURCE} returned 0 rows.")

except Exception as e:
    raise Exception(f"[Extract] Failed to read {FULL_SOURCE}: {e}")

# COMMAND ----------

# DBTITLE 1,Build feature table
# Drop non-predictive columns, rename to ML convention, cast booleans to int.
try:
    features_df = fact_df.drop(*DROP_COLS)

    for original, renamed in RENAME_MAP.items():
        features_df = features_df.withColumnRenamed(original, renamed)

    for column in BOOL_TO_INT_COLS:
        features_df = features_df.withColumn(column, F.col(column).cast("int"))

except Exception as e:
    raise Exception(f"[Transform] Feature table build failed: {e}")

# COMMAND ----------

# DBTITLE 1,Write feature table (overwrite)
# Atomic full-refresh write — the feature table is a snapshot, not historical.
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

# DBTITLE 1,Row count validation
# Confirm the write produced exactly the expected number of rows.
actual_count = spark.table(FULL_TARGET).count()

if actual_count != source_count:
    raise Exception(
        f"[Validation] Row count mismatch — source: {source_count}, target: {actual_count}"
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
