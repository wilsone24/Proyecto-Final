# Databricks notebook source
# DBTITLE 1,Catalog
spark.sql("USE CATALOG `pf1`")

# COMMAND ----------

# DBTITLE 1,Libraries
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, TimestampType
from delta.tables import DeltaTable

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("source_schema", "bronze")
dbutils.widgets.text("source_table", "cardioBronze")
dbutils.widgets.text("target_schema", "silver")
dbutils.widgets.text("target_table", "cardioSilver")

# COMMAND ----------

# DBTITLE 1,Variables
SOURCE_SCHEMA = dbutils.widgets.get("source_schema")
SOURCE_TABLE = dbutils.widgets.get("source_table")
TARGET_SCHEMA = dbutils.widgets.get("target_schema")
TARGET_TABLE = dbutils.widgets.get("target_table")
FULL_SOURCE = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
FULL_TARGET = f"{TARGET_SCHEMA}.{TARGET_TABLE}"
PIPELINE_VER = "1.0.0"

PROCESS_TS = spark.sql("SELECT TIMESTAMPADD(HOUR, -5, GETDATE())").collect()[0][0]

AGE_MIN_DAYS = 365 * 1
AGE_MAX_DAYS = 365 * 120
HEIGHT_MIN_CM = 100
HEIGHT_MAX_CM = 250
WEIGHT_MIN_KG = 10.0
WEIGHT_MAX_KG = 200.0
AP_HI_MIN = 60
AP_HI_MAX = 300
AP_LO_MIN = 40
AP_LO_MAX = 200

KEY = ["id"]

# COMMAND ----------

# DBTITLE 1,Columns
columnas = spark.createDataFrame(
    [
        ("id", "id", "INT"),
        ("age", "age", "INT"),
        ("gender", "gender", "INT"),
        ("height", "height_cm", "INT"),
        ("weight", "weight_kg", "DOUBLE"),
        ("ap_hi", "ap_hi", "INT"),
        ("ap_lo", "ap_lo", "INT"),
        ("cholesterol", "cholesterol", "INT"),
        ("gluc", "gluc", "INT"),
        ("smoke", "smoke", "BOOLEAN"),
        ("alco", "alco", "BOOLEAN"),
        ("active", "active", "BOOLEAN"),
        ("cardio", "cardio", "BOOLEAN"),
    ],
    ["originalName", "columnName", "dataType"],
).collect()

# COMMAND ----------

# DBTITLE 1,Metadata
column_comments = {
    "id": "Original patient identifier from the source dataset.",
    "age_years": "Patient age in full years (age / 365.25), rounded to 1 decimal.",
    "age_group_id": "Age group FK: 1=<30, 2=30-44, 3=45-59, 4=60-74, 5=>=75. Join with dim_age_group.",
    "gender": "Coded gender: 1 = female, 2 = male. Nulls imputed with mode.",
    "height_cm": "Height in centimetres. Outliers removed; nulls imputed with median by gender.",
    "weight_kg": "Weight in kilograms. Outliers removed; nulls imputed with median by gender.",
    "bmi": "Body Mass Index = weight_kg / (height_cm / 100)^2, rounded to 2 decimals.",
    "ap_hi": "Systolic blood pressure in mmHg. Outliers removed; nulls imputed with median.",
    "ap_lo": "Diastolic blood pressure in mmHg. Outliers removed; nulls imputed with median.",
    "pulse_pressure": "Pulse pressure = ap_hi - ap_lo.",
    "hypertension": "True when systolic >= 140 OR diastolic >= 90 (ACC/AHA 2017 criteria).",
    "cholesterol": "Cholesterol level: 1=normal, 2=above normal, 3=well above normal. Nulls imputed with mode.",
    "gluc": "Glucose level: 1=normal, 2=above normal, 3=well above normal. Nulls imputed with mode.",
    "smoke": "Smoking status boolean. Nulls imputed as false.",
    "alco": "Alcohol intake boolean. Nulls imputed as false.",
    "active": "Physical activity boolean. Nulls imputed as false.",
    "cardio": "Target — presence (true) or absence (false) of cardiovascular disease. Rows with null are dropped.",
    "is_current": "SCD Type-2 flag — true for the active version of this patient record.",
    "_silverIngestTime": "Timestamp (COT) when this record was written to the silver layer.",
}

table_comment = (
    "Silver layer — Cleaned, imputed and enriched cardiovascular disease data. "
    "Critical nulls (id, age, cardio, ap_hi, ap_lo) cause row removal. "
    "Rows where ap_lo >= ap_hi are removed as physiologically invalid. "
    "Outliers removed: height 100–250 cm, weight 10–200 kg, ap_hi 60–300 mmHg, ap_lo 40–200 mmHg, age 1–120 years. "
    "Secondary nulls imputed — gender/cholesterol/gluc: mode; height/weight: median by gender; ap_hi/ap_lo: global median; smoke/alco/active: false. "
    "Derived columns: "
    "age_years (age/365.25 rounded to 1 decimal), "
    "age_group_id (1=<30, 2=30-44, 3=45-59, 4=60-74, 5=>=75), "
    "bmi (weight_kg / (height_cm/100)^2), "
    "pulse_pressure (ap_hi - ap_lo), "
    "hypertension (ap_hi >= 140 OR ap_lo >= 90, ACC/AHA 2017). "
    "Binary fields (smoke, alco, active, cardio, hypertension) cast to Boolean. "
    "Implements SCD Type-2 on patient id — two-pass merge: pass 1 expires changed records, pass 2 inserts new versions. "
    f"Source: pf1.{FULL_SOURCE}. Pipeline version: {PIPELINE_VER}."
)

table_properties = {
    "data.domain": "health",
    "data.layer": "silver",
    "data.source": FULL_SOURCE,
    "data.owner": "data-engineering",
    "data.pii": "false",
    "data.contains.phi": "true",
    "data.sensitivity": "medium",
    "data.subject": "cardiovascular-disease",
    "pipeline.version": PIPELINE_VER,
    "delta.autoOptimize.optimizeWrite": "true",
    "delta.autoOptimize.autoCompact": "true",
}

# COMMAND ----------

# DBTITLE 1,Read bronze table
try:
    bronze_df = spark.table(FULL_SOURCE)
    bronze_count = bronze_df.count()

    if bronze_count == 0:
        raise Exception(f"Source table {FULL_SOURCE} returned 0 rows.")

except Exception as e:
    raise Exception(f"[Extract] Failed to read {FULL_SOURCE}: {e}")

# COMMAND ----------

# DBTITLE 1,Remove nulls, duplicates and outliers
try:
    clean_df = (
        bronze_df
        .filter(F.col("id").isNotNull())
        .filter(F.col("age").isNotNull())
        .filter(F.col("age").between(AGE_MIN_DAYS, AGE_MAX_DAYS))
        .filter(F.col("height").isNull() | F.col("height").between(HEIGHT_MIN_CM, HEIGHT_MAX_CM))
        .filter(F.col("weight").isNull() | F.col("weight").between(WEIGHT_MIN_KG, WEIGHT_MAX_KG))
        .filter(F.col("ap_hi").between(AP_HI_MIN, AP_HI_MAX))
        .filter(F.col("ap_lo").between(AP_LO_MIN, AP_LO_MAX))
        .filter(F.col("ap_hi").isNotNull())
        .filter(F.col("ap_lo").isNotNull())
        .filter(F.col("ap_lo") < F.col("ap_hi"))
        .filter(F.col("cardio").isNotNull())
        .drop("_ingestTime")
        .dropDuplicates()
    )

except Exception as e:
    raise Exception(f"[Filter] Critical-null / outlier removal failed: {e}")

# COMMAND ----------

# DBTITLE 1,Median and Mode
try:
    stats = clean_df.agg(
        F.percentile_approx("ap_hi", 0.5).alias("median_ap_hi"),
        F.percentile_approx("ap_lo", 0.5).alias("median_ap_lo"),
    ).collect()[0]

    MEDIAN_AP_HI = stats["median_ap_hi"]
    MEDIAN_AP_LO = stats["median_ap_lo"]

    # Height and weight medians by gender (more representative than a global median)
    gender_stats = (
        clean_df.groupBy("gender")
        .agg(
            F.percentile_approx("height", 0.5).alias("median_height"),
            F.percentile_approx("weight", 0.5).alias("median_weight"),
        )
        .collect()
    )
    height_by_gender = {row["gender"]: row["median_height"] for row in gender_stats}
    weight_by_gender = {row["gender"]: row["median_weight"] for row in gender_stats}

    def _mode(df, col):
        return (
            df.filter(F.col(col).isNotNull())
            .groupBy(col)
            .count()
            .orderBy(F.col("count").desc())
            .limit(1)
            .collect()[0][col]
        )

    MODE_GENDER = _mode(clean_df, "gender")
    MODE_CHOLESTEROL = _mode(clean_df, "cholesterol")
    MODE_GLUC = _mode(clean_df, "gluc")

except Exception as e:
    raise Exception(f"[Imputation Stats] Failed to compute imputation values: {e}")

# COMMAND ----------

# DBTITLE 1,Imputed data
try:
    # Gender-keyed median expressions for height and weight
    height_expr = F.col("height")
    weight_expr = F.col("weight")

    for gender_code, h_median in height_by_gender.items():
        w_median = weight_by_gender.get(gender_code)

        height_expr = (
            F.when(
                F.col("height").isNull()
                & (F.coalesce(F.col("gender"), F.lit(MODE_GENDER)) == gender_code),
                F.lit(h_median),
            )
            .otherwise(height_expr)
        )

        weight_expr = (
            F.when(
                F.col("weight").isNull()
                & (F.coalesce(F.col("gender"), F.lit(MODE_GENDER)) == gender_code),
                F.lit(w_median),
            )
            .otherwise(weight_expr)
        )

    imputed_df = (
        clean_df
        # Categorical: impute with mode
        .withColumn("gender", F.coalesce(F.col("gender"), F.lit(MODE_GENDER)))
        .withColumn("cholesterol", F.coalesce(F.col("cholesterol"), F.lit(MODE_CHOLESTEROL)))
        .withColumn("gluc", F.coalesce(F.col("gluc"), F.lit(MODE_GLUC)))
        # Numerical: impute with gender-aware median
        .withColumn("height", height_expr)
        .withColumn("weight", weight_expr)
        # Numerical: impute with global median
        .withColumn("ap_hi", F.coalesce(F.col("ap_hi"), F.lit(MEDIAN_AP_HI)))
        .withColumn("ap_lo", F.coalesce(F.col("ap_lo"), F.lit(MEDIAN_AP_LO)))
        # Self-reported binary fields: assume false when not reported
        .withColumn("smoke", F.coalesce(F.col("smoke"), F.lit(0)))
        .withColumn("alco", F.coalesce(F.col("alco"), F.lit(0)))
        .withColumn("active", F.coalesce(F.col("active"), F.lit(0)))
    )

except Exception as e:
    raise Exception(f"[Imputation] Failed to impute null values: {e}")

# COMMAND ----------

# DBTITLE 1,Column mapping and derived columns
try:
    stage = imputed_df.select(
        [F.col(c.originalName).cast(c.dataType).alias(c.columnName) for c in columnas]
    )

    incoming_df = (
        stage
        .withColumn("age_years", F.round(F.col("age") / 365.25, 1))
        .withColumn(
            "age_group_id",
            F.when(F.col("age_years") < 30, 1)
            .when(F.col("age_years") < 45, 2)
            .when(F.col("age_years") < 60, 3)
            .when(F.col("age_years") < 75, 4)
            .otherwise(5),
        )
        .withColumn("bmi", F.round(F.col("weight_kg") / F.pow(F.col("height_cm") / 100.0, 2), 2))
        .withColumn("pulse_pressure", F.col("ap_hi") - F.col("ap_lo"))
        # Hypertension flag (ACC/AHA criteria: systolic >= 140 OR diastolic >= 90)
        .withColumn("hypertension", ((F.col("ap_hi") >= 140) | (F.col("ap_lo") >= 90)).cast(BooleanType()))
        .drop("age")
        .withColumn("is_current", F.lit(True).cast(BooleanType()))
        .withColumn("_silverIngestTime", F.lit(PROCESS_TS).cast(TimestampType()))
        .select(
            "id",
            "age_years",
            "age_group_id",
            "gender",
            "height_cm",
            "weight_kg",
            "bmi",
            "ap_hi",
            "ap_lo",
            "pulse_pressure",
            "hypertension",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "cardio",
            "is_current",
            "_silverIngestTime",
        )
    )

except Exception as e:
    raise Exception(f"[Transform] Feature engineering failed: {e}")

# COMMAND ----------

# DBTITLE 1,Create target table
try:
    table_schema_ddl = ",\n  ".join([
        f"`{field.name}` {field.dataType.simpleString()}"
        for field in incoming_df.schema.fields
    ])
    spark.sql(f"CREATE TABLE IF NOT EXISTS {FULL_TARGET} ({table_schema_ddl}) USING DELTA")

except Exception as e:
    raise Exception(f"[Bootstrap] Failed to create table {FULL_TARGET}: {e}")

# COMMAND ----------

# DBTITLE 1,Mappings of merge
try:
    silver = DeltaTable.forName(spark, FULL_TARGET)
except Exception as e:
    raise Exception(f"[Merge] Cannot open DeltaTable {FULL_TARGET}: {e}")

mapping = {f"`{c}`": f"s.`{c}`" for c in incoming_df.columns}
mappingFalse = {"is_current": "false"}

update_condition = " OR ".join(
    [
        f"NOT(a.`{field.name}` <=> s.`{field.name}`)"
        for field in incoming_df.drop(*KEY)
        .drop("is_current", "_silverIngestTime")
        .schema.fields
    ]
)

# COMMAND ----------

# DBTITLE 1,Merge conditions
try:
    (
        silver.alias("a")
        .merge(
            incoming_df.alias("s"),
            " AND ".join([f"a.{k} = s.{k}" for k in KEY]) + " AND a.is_current = true",
        )
        .whenMatchedUpdate(condition=update_condition, set=mappingFalse)
        .whenNotMatchedBySourceUpdate(condition="a.is_current = true", set=mappingFalse)
        .execute()
    )

except Exception as e:
    raise Exception(f"[Merge] Pass 1 (expire) failed on {FULL_TARGET}: {e}")

try:
    (
        silver.alias("a")
        .merge(
            incoming_df.alias("s"),
            " AND ".join([f"a.{k} = s.{k}" for k in KEY]) + " AND a.is_current = true",
        )
        .whenNotMatchedInsert(values=mapping)
        .execute()
    )

except Exception as e:
    raise Exception(f"[Merge] Pass 2 (insert) failed on {FULL_TARGET}: {e}")

# COMMAND ----------

# DBTITLE 1,Add metadata to the target
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
    raise Exception(
        f"[Table Comment] Failed to apply table comment on {FULL_TARGET}: {e}"
    )

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
    raise Exception(
        f"[Table Properties] Failed to apply properties on {FULL_TARGET}: {e}"
    )
