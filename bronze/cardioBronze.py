# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Enviroment
spark.sql("USE CATALOG `pf1`")

# COMMAND ----------

# DBTITLE 1,Libraries
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from datetime import datetime

# COMMAND ----------

dbutils.widgets.text("source_path", "")
dbutils.widgets.text("schema_name", "")
dbutils.widgets.text("table_name", "")

# COMMAND ----------

# DBTITLE 1,Constants
SOURCE_PATH = dbutils.widgets.get("source_path")
SCHEMA_NAME = dbutils.widgets.get("schema_name")
TABLE_NAME = dbutils.widgets.get("table_name")
FULL_TABLE = f"{SCHEMA_NAME}.{TABLE_NAME}"
INGEST_TS = datetime.utcnow().isoformat()

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

# COMMAND ----------

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
}

table_comment = (
    "Bronze layer — Raw cardiovascular disease data (70,000 patients). "
    "Source: cardio_train.csv, Kaggle Cardiovascular Disease dataset. "
    "Contains objective features (age, weight, height), examination features (blood pressure, cholesterol, glucose) "
    "and self-reported features (smoking, alcohol, physical activity). "
    "Target: cardio (presence or absence of cardiovascular disease). "
    "_* columns are operational ingestion metadata added at ingest time."
)

table_properties = {
    "data.domain": "health",
    "data.layer": "bronze",
    "data.source": "cardio_train.csv",
    "data.source.type": "csv",
    "data.owner": "data-engineering",
    "data.pii": "false",
    "data.contains.phi": "true",
    "data.sensitivity": "medium",
    "data.subject": "cardiovascular-disease",
    "data.rows.expected": "70000",
    "pipeline.version": "1.0.0",
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
}

EXPECTED_ROWS = int(table_properties["data.rows.expected"])

# COMMAND ----------

# DBTITLE 1,Ingestion
try:
    stage = (
        spark.read.option("header", True)
        .option("delimiter", ";")
        .schema(schema)
        .csv(SOURCE_PATH)
        .withColumn("_ingestTime", F.lit(INGEST_TS).cast("timestamp"))
    )

    (
        stage.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FULL_TABLE)
    )

except Exception as e:
    raise Exception(f"[Ingestion] Failed to write table {FULL_TABLE}: {e}")

# COMMAND ----------

actual_rows = spark.table(FULL_TABLE).count()

if actual_rows != EXPECTED_ROWS:
    raise Exception(
        f"[Validation] Row count mismatch — expected: {EXPECTED_ROWS}, actual: {actual_rows}"
    )

# COMMAND ----------

try:
    for column, comment in column_comments.items():
        spark.sql(
            f"""
                ALTER TABLE {FULL_TABLE}
                ALTER COLUMN `{column}` COMMENT '{comment}'
            """
        )

except Exception as e:
    raise Exception(f"[Column Comments] Failed to apply comments on {FULL_TABLE}: {e}")

try:
    spark.sql(f"COMMENT ON TABLE {FULL_TABLE} IS '{table_comment}'")

except Exception as e:
    raise Exception(
        f"[Table Comment] Failed to apply table comment on {FULL_TABLE}: {e}"
    )

try:
    props_ddl = ", ".join([f"'{k}' = '{v}'" for k, v in table_properties.items()])

    spark.sql(
        f"""
            ALTER TABLE {FULL_TABLE}
            SET TBLPROPERTIES (
                {props_ddl}
            )
        """
    )

except Exception as e:
    raise Exception(
        f"[Table Properties] Failed to apply properties on {FULL_TABLE}: {e}"
    )

