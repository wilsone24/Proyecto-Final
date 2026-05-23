# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Enviroment
# Set the active Unity Catalog for all subsequent table references.
spark.sql("USE CATALOG `databricks_service_pf`")

# COMMAND ----------

# DBTITLE 1,Libraries
# Spark APIs for transformations and schema declaration, plus datetime for ingest timestamps.
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from datetime import datetime, timezone

# COMMAND ----------

# DBTITLE 1,Parameters
# Job-level inputs: source CSV path and target bronze table.
dbutils.widgets.text("source_path", "/Volumes/databricks_service_pf/bronze/dataset/cardio_train.csv")
dbutils.widgets.text("schema_name", "bronze")
dbutils.widgets.text("table_name",  "raw_kaggle__cardio_patients")

# COMMAND ----------

# DBTITLE 1,Constants
# Validate widgets and derive runtime constants (table, schema, CSV options, metadata columns).
SOURCE_PATH = dbutils.widgets.get("source_path")
SCHEMA_NAME = dbutils.widgets.get("schema_name")
TABLE_NAME = dbutils.widgets.get("table_name")

# Fail fast if any required widget was not provided
if not all([SOURCE_PATH, SCHEMA_NAME, TABLE_NAME]):
    raise ValueError(
        f"Missing required widgets: source_path='{SOURCE_PATH}', "
        f"schema_name='{SCHEMA_NAME}', table_name='{TABLE_NAME}'"
    )

FULL_TABLE       = f"{SCHEMA_NAME}.{TABLE_NAME}"
INGEST_TS        = datetime.now(timezone.utc).isoformat()
PIPELINE_VERSION = "1.0.0"

# Pipeline-level configuration (kept separate from table metadata)
EXPECTED_ROWS = 70000

# CSV reader options — change here if the source format ever changes
CSV_OPTIONS = {
    "header":    "true",
    "delimiter": ";",
}

# Source data schema
schema = StructType(
    [
        StructField("id", IntegerType(), nullable=True),
        StructField("age", IntegerType(), nullable=True),
        StructField("gender", IntegerType(), nullable=True),
        StructField("height", IntegerType(), nullable=True),
        StructField("weight", DoubleType(), nullable=True),
        StructField("ap_hi", IntegerType(), nullable=True),
        StructField("ap_lo", IntegerType(), nullable=True),
        StructField("cholesterol", IntegerType(), nullable=True),
        StructField("gluc", IntegerType(), nullable=True),
        StructField("smoke", IntegerType(), nullable=True),
        StructField("alco", IntegerType(), nullable=True),
        StructField("active", IntegerType(), nullable=True),
        StructField("cardio", IntegerType(), nullable=True),
    ]
)

# Operational metadata columns added at ingest time.
# To add a new metadata column: add an entry here AND an entry in column_comments.
METADATA_COLUMNS = {
    "_ingestTime":                 F.lit(INGEST_TS).cast("timestamp"),
    "_sourceFileName":             F.col("_metadata.file_name"),
    "_sourceFileModificationTime": F.col("_metadata.file_modification_time"),
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
        raise Exception(f"[{label}] Failed on {FULL_TABLE}: {e}")


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
    "id": "Unique patient identifier in the original dataset.",
    "age": "Patient age in days.",
    "gender": "Gender: 1 = female, 2 = male.",
    "height": "Height in centimeters.",
    "weight": "Weight in kilograms.",
    "ap_hi": "Systolic blood pressure (mmHg). Examination feature.",
    "ap_lo": "Diastolic blood pressure (mmHg). Examination feature.",
    "cholesterol": "Cholesterol level: 1=normal, 2=above normal, 3=well above normal.",
    "gluc": "Glucose level: 1=normal, 2=above normal, 3=well above normal.",
    "smoke": "Smoking status: 0=no, 1=yes. Self-reported.",
    "alco": "Alcohol intake: 0=no, 1=yes. Self-reported.",
    "active": "Physical activity: 0=no, 1=yes. Self-reported.",
    "cardio": "Target variable: 0=no cardiovascular disease, 1=cardiovascular disease present.",
    "_ingestTime": "UTC timestamp of ingestion into the bronze layer.",
    "_sourceFileName": "Name of the source file this row was ingested from. Useful for traceability and debugging.",
    "_sourceFileModificationTime": "Modification timestamp of the source file at ingest time. Distinguishes between an original file and a re-uploaded version.",
}

table_comment = (
    "Bronze layer — Raw cardiovascular disease data (70,000 patients). "
    "Source: cardio_train.csv, Kaggle Cardiovascular Disease dataset. "
    "Contains objective features (age, weight, height), examination features (blood pressure, cholesterol, glucose) "
    "and self-reported features (smoking, alcohol, physical activity). "
    "Target: cardio (presence or absence of cardiovascular disease). "
    "_* columns are operational ingestion metadata added at ingest time "
    "(_ingestTime, _sourceFileName, _sourceFileModificationTime)."
)

table_properties = {
    "data.domain":         "health",
    "data.layer":          "bronze",
    "data.source":         "cardio_train.csv",
    "data.source.type":    "csv",
    "data.owner":          "data-engineering",
    "data.pii":            "true",
    "data.contains.phi":   "true",
    "data.sensitivity":    "medium",
    "data.subject":        "cardiovascular-disease",
    "pipeline.version":    PIPELINE_VERSION,
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact":   "true",
}

# COMMAND ----------

# DBTITLE 1,Ingestion
# Read CSV with declared schema, attach metadata columns, overwrite bronze Delta table.
try:
    reader = spark.read.schema(schema)
    for option_key, option_value in CSV_OPTIONS.items():
        reader = reader.option(option_key, option_value)

    stage = reader.csv(SOURCE_PATH)

    for column_name, column_expression in METADATA_COLUMNS.items():
        stage = stage.withColumn(column_name, column_expression)

    (
        stage.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FULL_TABLE)
    )

except Exception as e:
    raise Exception(f"[Ingestion] Failed to write table {FULL_TABLE}: {e}")

# COMMAND ----------

# DBTITLE 1,Rows Validation
# Fail loudly if row count differs from EXPECTED_ROWS (catches corrupted/truncated source).
actual_rows = spark.table(FULL_TABLE).count()

if actual_rows != EXPECTED_ROWS:
    raise Exception(
        f"[Validation] Row count mismatch — expected: {EXPECTED_ROWS}, actual: {actual_rows}"
    )

# COMMAND ----------

# DBTITLE 1,Add metadata to the table
# Apply column COMMENTs, table COMMENT and TBLPROPERTIES via execute_sql_safely().

# Column comments
for column, comment in column_comments.items():
    safe_comment = escape_sql_string(comment)
    execute_sql_safely(
        "Column Comments",
        f"ALTER TABLE {FULL_TABLE} ALTER COLUMN `{column}` COMMENT '{safe_comment}'",
    )

# Table comment
safe_table_comment = escape_sql_string(table_comment)
execute_sql_safely(
    "Table Comment",
    f"COMMENT ON TABLE {FULL_TABLE} IS '{safe_table_comment}'",
)

# Table properties
props_ddl = ", ".join([f"'{k}' = '{v}'" for k, v in table_properties.items()])
execute_sql_safely(
    "Table Properties",
    f"ALTER TABLE {FULL_TABLE} SET TBLPROPERTIES ({props_ddl})",
)
