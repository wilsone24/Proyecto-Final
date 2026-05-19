# Databricks notebook source
# DBTITLE 1,Catalog
# Set the active Unity Catalog for all subsequent table references.
spark.sql("USE CATALOG `databricks_service_pf`")

# COMMAND ----------

# DBTITLE 1,Libraries
# Spark APIs for transformations, Delta merge, and Python datetime for ingest timestamps.
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, TimestampType
from delta.tables import DeltaTable
from datetime import datetime, timezone

# COMMAND ----------

# DBTITLE 1,Parameters
# Job-level inputs: source bronze table and target silver table.
dbutils.widgets.text("source_schema", "bronze")
dbutils.widgets.text("source_table",  "raw_kaggle__cardio_patients")
dbutils.widgets.text("target_schema", "silver")
dbutils.widgets.text("target_table",  "cardio_patients")

# COMMAND ----------

# DBTITLE 1,Constants
# Validate widgets and derive runtime constants (tables, ingest timestamp, thresholds, SCD config, imputation rules, target schema).
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
PROCESS_TS       = datetime.now(timezone.utc).isoformat()
PIPELINE_VERSION = "1.0.0"

# Outlier filtering thresholds — inclusive ranges
AGE_MIN_DAYS  = 365 * 1
AGE_MAX_DAYS  = 365 * 120
HEIGHT_MIN_CM = 100
HEIGHT_MAX_CM = 250
WEIGHT_MIN_KG = 10.0
WEIGHT_MAX_KG = 200.0
AP_HI_MIN     = 60
AP_HI_MAX     = 300
AP_LO_MIN     = 40
AP_LO_MAX     = 200

# Clinical thresholds — ACC/AHA 2017 hypertension stage 2 criteria
HYPERTENSION_SYSTOLIC_THRESHOLD  = 140  # mmHg
HYPERTENSION_DIASTOLIC_THRESHOLD = 90   # mmHg

# SCD Type-2 conventions
KEY                     = ["id"]
SCD_CURRENT_FLAG        = "is_current"
SCD_INGEST_COLUMN       = "_silverIngestTime"
# Operational metadata columns — must NOT trigger SCD-2 updates on their own.
# If only these change between bronze runs, the patient record is the same.
SCD_NON_TRACKED_COLUMNS = [
    SCD_CURRENT_FLAG,
    SCD_INGEST_COLUMN,
    "_sourceFileName",
    "_sourceFileModificationTime",
]

# Declarative imputation strategy — drives the imputation cell below.
# To change how a column is imputed, move it between buckets here.
IMPUTATION_RULES = {
    "categorical_mode":        ["gender", "cholesterol", "gluc"],
    "numerical_global_median": ["ap_hi", "ap_lo"],
    "numerical_gender_median": ["height", "weight"],
    "binary_false":            ["smoke", "alco", "active"],
}

# Final column order written to silver (matches the schema downstream consumers expect).
SILVER_COLUMNS = [
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
    SCD_CURRENT_FLAG,
    SCD_INGEST_COLUMN,
    # Bronze lineage — carried forward for full traceability (not SCD-tracked)
    "_sourceFileName",
    "_sourceFileModificationTime",
]

# Operational metadata columns added to every silver record.
# To add a new metadata column: add an entry here AND an entry in column_comments.
METADATA_COLUMNS = {
    SCD_CURRENT_FLAG:  F.lit(True).cast(BooleanType()),
    SCD_INGEST_COLUMN: F.lit(PROCESS_TS).cast(TimestampType()),
}

# COMMAND ----------

# DBTITLE 1,Helpers
# Reusable utilities for safe DDL execution, SQL string escaping, mode computation and gender-aware imputation.
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


def compute_mode(df, column: str):
    """Return the most frequent non-null value of a column in a Spark DataFrame.

    Args:
        df: Spark DataFrame to scan.
        column: Name of the column whose mode will be computed.

    Returns:
        The most frequent value (any type). Returns None if the column is empty
        or contains only nulls.
    """
    rows = (
        df.filter(F.col(column).isNotNull())
        .groupBy(column)
        .count()
        .orderBy(F.col("count").desc())
        .limit(1)
        .collect()
    )
    return rows[0][column] if rows else None


def build_gender_aware_imputation(
    column: str,
    gender_medians: dict,
    mode_gender,
):
    """Build a Column expression that imputes nulls in `column` using a gender-keyed median.

    For each null value in `column`, the expression looks at the row's `gender`
    (falling back to `mode_gender` if also null) and substitutes the pre-computed
    median for that gender. Non-null values are passed through unchanged.

    Args:
        column: Name of the column to impute (e.g. "height").
        gender_medians: Mapping from gender code to median value for that column.
        mode_gender: Value used as a fallback when the gender column itself is null.

    Returns:
        A Spark Column expression suitable for `.withColumn(column, ...)`.
    """
    expr = F.col(column)
    for gender_code, median_value in gender_medians.items():
        expr = (
            F.when(
                F.col(column).isNull()
                & (F.coalesce(F.col("gender"), F.lit(mode_gender)) == gender_code),
                F.lit(median_value),
            )
            .otherwise(expr)
        )
    return expr

# COMMAND ----------

# DBTITLE 1,Column mapping
# Source-to-target column mapping (renames + type casts applied during the transform).
# Bronze lineage columns are carried through unchanged for full traceability.
column_mapping = spark.createDataFrame(
    [
        ("id",                          "id",                          "INT"),
        ("age",                         "age",                         "INT"),
        ("gender",                      "gender",                      "INT"),
        ("height",                      "height_cm",                   "INT"),
        ("weight",                      "weight_kg",                   "DOUBLE"),
        ("ap_hi",                       "ap_hi",                       "INT"),
        ("ap_lo",                       "ap_lo",                       "INT"),
        ("cholesterol",                 "cholesterol",                 "INT"),
        ("gluc",                        "gluc",                        "INT"),
        ("smoke",                       "smoke",                       "BOOLEAN"),
        ("alco",                        "alco",                        "BOOLEAN"),
        ("active",                      "active",                      "BOOLEAN"),
        ("cardio",                      "cardio",                      "BOOLEAN"),
        # Bronze lineage — preserved through silver for traceability
        ("_sourceFileName",             "_sourceFileName",             "STRING"),
        ("_sourceFileModificationTime", "_sourceFileModificationTime", "TIMESTAMP"),
    ],
    ["originalName", "columnName", "dataType"],
).collect()

# COMMAND ----------

# DBTITLE 1,Metadata
# Column comments, table comment and Unity Catalog properties applied after the merge.
column_comments = {
    "id":                "Original patient identifier from the source dataset.",
    "age_years":         "Patient age in full years (age / 365.25), rounded to 1 decimal.",
    "age_group_id":      "Age group FK: 1=<30, 2=30-44, 3=45-59, 4=60-74, 5=>=75. Join with dim_age_group.",
    "gender":            "Coded gender: 1 = female, 2 = male. Nulls imputed with mode.",
    "height_cm":         "Height in centimetres. Outliers removed; nulls imputed with median by gender.",
    "weight_kg":         "Weight in kilograms. Outliers removed; nulls imputed with median by gender.",
    "bmi":               "Body Mass Index = weight_kg / (height_cm / 100)^2, rounded to 2 decimals.",
    "ap_hi":             "Systolic blood pressure in mmHg. Outliers removed; nulls imputed with median.",
    "ap_lo":             "Diastolic blood pressure in mmHg. Outliers removed; nulls imputed with median.",
    "pulse_pressure":    "Pulse pressure = ap_hi - ap_lo.",
    "hypertension":      "True when systolic >= 140 OR diastolic >= 90 (ACC/AHA 2017 criteria).",
    "cholesterol":       "Cholesterol level: 1=normal, 2=above normal, 3=well above normal. Nulls imputed with mode.",
    "gluc":              "Glucose level: 1=normal, 2=above normal, 3=well above normal. Nulls imputed with mode.",
    "smoke":             "Smoking status boolean. Nulls imputed as false.",
    "alco":              "Alcohol intake boolean. Nulls imputed as false.",
    "active":            "Physical activity boolean. Nulls imputed as false.",
    "cardio":            "Target — presence (true) or absence (false) of cardiovascular disease. Rows with null are dropped.",
    "is_current":                  "SCD Type-2 flag — true for the active version of this patient record.",
    "_silverIngestTime":           "UTC timestamp when this record was written to the silver layer.",
    "_sourceFileName":             "Source file name from bronze. Preserved for end-to-end lineage and debugging.",
    "_sourceFileModificationTime": "Source file modification timestamp from bronze. Preserved for end-to-end lineage.",
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
    "Bronze lineage (_sourceFileName, _sourceFileModificationTime) is preserved for end-to-end traceability and excluded from SCD-2 change detection. "
    f"Source: {FULL_SOURCE}. Pipeline version: {PIPELINE_VERSION}."
)

table_properties = {
    "data.domain":       "health",
    "data.layer":        "silver",
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

# DBTITLE 1,Read bronze table
# Load source bronze table and abort early if it returned no rows.
try:
    bronze_df    = spark.table(FULL_SOURCE)
    bronze_count = bronze_df.count()

    if bronze_count == 0:
        raise Exception(f"Source table {FULL_SOURCE} returned 0 rows.")

except Exception as e:
    raise Exception(f"[Extract] Failed to read {FULL_SOURCE}: {e}")

# COMMAND ----------

# DBTITLE 1,Remove nulls, duplicates and outliers
# Drop rows with critical nulls, physiologically impossible values and outliers; deduplicate.
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

# DBTITLE 1,Compute imputation statistics (median + mode)
# Pre-compute global medians, gender-keyed medians and modes used by the imputation step.
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

    MODE_GENDER      = compute_mode(clean_df, "gender")
    MODE_CHOLESTEROL = compute_mode(clean_df, "cholesterol")
    MODE_GLUC        = compute_mode(clean_df, "gluc")

    # Lookup tables consumed by the declarative imputation loop below
    mode_values = {
        "gender":      MODE_GENDER,
        "cholesterol": MODE_CHOLESTEROL,
        "gluc":        MODE_GLUC,
    }
    global_medians = {
        "ap_hi": MEDIAN_AP_HI,
        "ap_lo": MEDIAN_AP_LO,
    }
    gender_medians = {
        "height": height_by_gender,
        "weight": weight_by_gender,
    }

except Exception as e:
    raise Exception(f"[Imputation Stats] Failed to compute imputation values: {e}")

# COMMAND ----------

# DBTITLE 1,Impute nulls
# Apply the imputation rules declared in IMPUTATION_RULES.
# Categorical → mode | gender-keyed numerical → median per gender | global numerical → global median | binary → false.
try:
    imputed_df = clean_df

    # Categorical: impute with mode
    for column in IMPUTATION_RULES["categorical_mode"]:
        imputed_df = imputed_df.withColumn(
            column,
            F.coalesce(F.col(column), F.lit(mode_values[column])),
        )

    # Numerical: impute with gender-aware median
    for column in IMPUTATION_RULES["numerical_gender_median"]:
        imputed_df = imputed_df.withColumn(
            column,
            build_gender_aware_imputation(column, gender_medians[column], MODE_GENDER),
        )

    # Numerical: impute with global median
    for column in IMPUTATION_RULES["numerical_global_median"]:
        imputed_df = imputed_df.withColumn(
            column,
            F.coalesce(F.col(column), F.lit(global_medians[column])),
        )

    # Self-reported binary fields: assume false when not reported
    for column in IMPUTATION_RULES["binary_false"]:
        imputed_df = imputed_df.withColumn(
            column,
            F.coalesce(F.col(column), F.lit(0)),
        )

except Exception as e:
    raise Exception(f"[Imputation] Failed to impute null values: {e}")

# COMMAND ----------

# DBTITLE 1,Column mapping and derived columns
# Rename + cast to the silver schema, derive age/BMI/pulse/hypertension and attach SCD metadata.
try:
    stage = imputed_df.select(
        [F.col(c.originalName).cast(c.dataType).alias(c.columnName) for c in column_mapping]
    )

    enriched = (
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
        # Hypertension flag (ACC/AHA 2017 criteria: systolic >= 140 OR diastolic >= 90)
        .withColumn(
            "hypertension",
            (
                (F.col("ap_hi") >= HYPERTENSION_SYSTOLIC_THRESHOLD)
                | (F.col("ap_lo") >= HYPERTENSION_DIASTOLIC_THRESHOLD)
            ).cast(BooleanType()),
        )
        .drop("age")
    )

    # Attach operational metadata columns from the centralized dict
    for column_name, column_expression in METADATA_COLUMNS.items():
        enriched = enriched.withColumn(column_name, column_expression)

    incoming_df = enriched.select(*SILVER_COLUMNS)

except Exception as e:
    raise Exception(f"[Transform] Feature engineering failed: {e}")

# COMMAND ----------

# DBTITLE 1,Create target table if missing
# Bootstrap an empty Delta table with the incoming schema so the first merge has a target.
try:
    table_schema_ddl = ",\n  ".join([
        f"`{field.name}` {field.dataType.simpleString()}"
        for field in incoming_df.schema.fields
    ])
    execute_sql_safely(
        "Bootstrap",
        f"CREATE TABLE IF NOT EXISTS {FULL_TARGET} ({table_schema_ddl}) USING DELTA",
    )

except Exception as e:
    raise Exception(f"[Bootstrap] Failed to create table {FULL_TARGET}: {e}")

# COMMAND ----------

# DBTITLE 1,Build merge mappings
# Build the column-to-column mapping and the change-detection condition used by the SCD-2 merge.
try:
    silver_table = DeltaTable.forName(spark, FULL_TARGET)
except Exception as e:
    raise Exception(f"[Merge] Cannot open DeltaTable {FULL_TARGET}: {e}")

merge_column_mapping = {f"`{c}`": f"s.`{c}`" for c in incoming_df.columns}
expire_mapping       = {SCD_CURRENT_FLAG: "false"}

update_condition = " OR ".join(
    [
        f"NOT(a.`{field.name}` <=> s.`{field.name}`)"
        for field in incoming_df.drop(*KEY)
            .drop(*SCD_NON_TRACKED_COLUMNS)
            .schema.fields
    ]
)

# COMMAND ----------

# DBTITLE 1,SCD Type-2 merge (pass 1 — expire, pass 2 — insert)
# Pass 1: flip is_current=false on rows whose attributes changed OR that disappeared from source.
# Pass 2: insert the new version of every changed/new id.
try:
    (
        silver_table.alias("a")
        .merge(
            incoming_df.alias("s"),
            " AND ".join([f"a.{k} = s.{k}" for k in KEY]) + f" AND a.{SCD_CURRENT_FLAG} = true",
        )
        .whenMatchedUpdate(condition=update_condition, set=expire_mapping)
        .whenNotMatchedBySourceUpdate(condition=f"a.{SCD_CURRENT_FLAG} = true", set=expire_mapping)
        .execute()
    )

except Exception as e:
    raise Exception(f"[Merge] Pass 1 (expire) failed on {FULL_TARGET}: {e}")

try:
    (
        silver_table.alias("a")
        .merge(
            incoming_df.alias("s"),
            " AND ".join([f"a.{k} = s.{k}" for k in KEY]) + f" AND a.{SCD_CURRENT_FLAG} = true",
        )
        .whenNotMatchedInsert(values=merge_column_mapping)
        .execute()
    )

except Exception as e:
    raise Exception(f"[Merge] Pass 2 (insert) failed on {FULL_TARGET}: {e}")

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
