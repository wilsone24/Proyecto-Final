"""
test_silver.py
--------------
Runs the silver transformation logic locally without Databricks.
Uses local_spark.py for a local SparkSession + dbutils mock.

Requirements
    pip install pyspark delta-spark pandas

Usage
    python tests/test_silver.py
    # or, from the repo root:
    python -m tests.test_silver
"""

import pathlib
import sys

# Ensure repo root is on the path so `local_spark` resolves
REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from local_spark import spark, dbutils          # noqa: E402
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, TimestampType, StructType, StructField, IntegerType, DoubleType

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE = REPO_ROOT / "data" / "cardio_train.csv"

# ---------------------------------------------------------------------------
# Constants (mirrors cardioSilver.py)
# ---------------------------------------------------------------------------
AGE_MIN_DAYS   = 365 * 1
AGE_MAX_DAYS   = 365 * 120
HEIGHT_MIN_CM  = 100
HEIGHT_MAX_CM  = 250
WEIGHT_MIN_KG  = 10.0
WEIGHT_MAX_KG  = 200.0
AP_HI_MIN      = 60
AP_HI_MAX      = 300
AP_LO_MIN      = 40
AP_LO_MAX      = 200

SCHEMA = StructType([
    StructField("id",          IntegerType(), True),
    StructField("age",         IntegerType(), True),
    StructField("gender",      IntegerType(), True),
    StructField("height",      IntegerType(), True),
    StructField("weight",      DoubleType(),  True),
    StructField("ap_hi",       IntegerType(), True),
    StructField("ap_lo",       IntegerType(), True),
    StructField("cholesterol", IntegerType(), True),
    StructField("gluc",        IntegerType(), True),
    StructField("smoke",       IntegerType(), True),
    StructField("alco",        IntegerType(), True),
    StructField("active",      IntegerType(), True),
    StructField("cardio",      IntegerType(), True),
])

# ---------------------------------------------------------------------------
# 1. Load raw CSV (replaces spark.table("bronze.cardioBronze"))
# ---------------------------------------------------------------------------
def load_bronze():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}.\n"
            "Place cardio_train.csv in the data/ folder and re-run."
        )
    df = (
        spark.read
        .option("header", True)
        .option("delimiter", ";")
        .schema(SCHEMA)
        .csv(str(DATA_FILE))
    )
    print(f"Bronze rows loaded: {df.count():,}")
    return df

# ---------------------------------------------------------------------------
# 2. Silver transformation (mirrors cardioSilver.py logic)
# ---------------------------------------------------------------------------
def transform_silver(bronze_df):
    # ── Remove nulls, duplicates, outliers ─────────────────────────────────
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
        .dropDuplicates()
    )

    # ── Compute imputation stats ────────────────────────────────────────────
    stats = clean_df.agg(
        F.percentile_approx("ap_hi", 0.5).alias("median_ap_hi"),
        F.percentile_approx("ap_lo", 0.5).alias("median_ap_lo"),
    ).collect()[0]
    MEDIAN_AP_HI = stats["median_ap_hi"]
    MEDIAN_AP_LO = stats["median_ap_lo"]

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
            .groupBy(col).count()
            .orderBy(F.col("count").desc())
            .limit(1).collect()[0][col]
        )

    MODE_GENDER      = _mode(clean_df, "gender")
    MODE_CHOLESTEROL = _mode(clean_df, "cholesterol")
    MODE_GLUC        = _mode(clean_df, "gluc")

    # ── Impute ──────────────────────────────────────────────────────────────
    height_expr = F.col("height")
    weight_expr = F.col("weight")

    for gender_code, h_med in height_by_gender.items():
        w_med = weight_by_gender.get(gender_code)
        height_expr = F.when(
            F.col("height").isNull() &
            (F.coalesce(F.col("gender"), F.lit(MODE_GENDER)) == gender_code),
            F.lit(h_med),
        ).otherwise(height_expr)
        weight_expr = F.when(
            F.col("weight").isNull() &
            (F.coalesce(F.col("gender"), F.lit(MODE_GENDER)) == gender_code),
            F.lit(w_med),
        ).otherwise(weight_expr)

    imputed_df = (
        clean_df
        .withColumn("gender",      F.coalesce(F.col("gender"),      F.lit(MODE_GENDER)))
        .withColumn("cholesterol", F.coalesce(F.col("cholesterol"), F.lit(MODE_CHOLESTEROL)))
        .withColumn("gluc",        F.coalesce(F.col("gluc"),        F.lit(MODE_GLUC)))
        .withColumn("height",      height_expr)
        .withColumn("weight",      weight_expr)
        .withColumn("ap_hi",       F.coalesce(F.col("ap_hi"), F.lit(MEDIAN_AP_HI)))
        .withColumn("ap_lo",       F.coalesce(F.col("ap_lo"), F.lit(MEDIAN_AP_LO)))
        .withColumn("smoke",       F.coalesce(F.col("smoke"),  F.lit(0)))
        .withColumn("alco",        F.coalesce(F.col("alco"),   F.lit(0)))
        .withColumn("active",      F.coalesce(F.col("active"), F.lit(0)))
    )

    # ── Column mapping + derived columns ────────────────────────────────────
    PROCESS_TS = spark.sql("SELECT current_timestamp()").collect()[0][0]

    silver_df = (
        imputed_df
        .withColumnRenamed("height", "height_cm")
        .withColumnRenamed("weight", "weight_kg")
        .withColumn("age_years",   F.round(F.col("age") / 365.25, 1))
        .withColumn(
            "age_group_id",
            F.when(F.col("age_years") < 30, 1)
             .when(F.col("age_years") < 45, 2)
             .when(F.col("age_years") < 60, 3)
             .when(F.col("age_years") < 75, 4)
             .otherwise(5),
        )
        .withColumn("bmi",           F.round(F.col("weight_kg") / F.pow(F.col("height_cm") / 100.0, 2), 2))
        .withColumn("pulse_pressure", F.col("ap_hi") - F.col("ap_lo"))
        .withColumn("hypertension",  ((F.col("ap_hi") >= 140) | (F.col("ap_lo") >= 90)).cast(BooleanType()))
        .withColumn("smoke",         F.col("smoke").cast(BooleanType()))
        .withColumn("alco",          F.col("alco").cast(BooleanType()))
        .withColumn("active",        F.col("active").cast(BooleanType()))
        .withColumn("cardio",        F.col("cardio").cast(BooleanType()))
        .withColumn("is_current",    F.lit(True).cast(BooleanType()))
        .withColumn("_silverIngestTime", F.lit(PROCESS_TS).cast(TimestampType()))
        .drop("age")
        .select(
            "id", "age_years", "age_group_id", "gender",
            "height_cm", "weight_kg", "bmi",
            "ap_hi", "ap_lo", "pulse_pressure", "hypertension",
            "cholesterol", "gluc",
            "smoke", "alco", "active", "cardio",
            "is_current", "_silverIngestTime",
        )
    )
    return silver_df

# ---------------------------------------------------------------------------
# 3. Validate
# ---------------------------------------------------------------------------
def validate(silver_df):
    count = silver_df.count()
    print(f"\nSilver rows: {count:,}")
    print("\nSchema:")
    silver_df.printSchema()
    print("\nSample (5 rows):")
    silver_df.show(5, truncate=False)

    null_counts = silver_df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in silver_df.columns]
    )
    print("\nNull counts per column:")
    null_counts.show(truncate=False)

    assert count > 0, "Silver table is empty!"
    print("\nAll assertions passed.")

# ---------------------------------------------------------------------------
# 4. Optionally persist as Delta locally
# ---------------------------------------------------------------------------
def save_local_delta(silver_df):
    out_path = str(REPO_ROOT / "data" / "silver_delta")
    (
        silver_df.write
        .format("delta")
        .mode("overwrite")
        .save(out_path)
    )
    print(f"\nDelta table written to: {out_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bronze = load_bronze()
    silver = transform_silver(bronze)
    validate(silver)
    save_local_delta(silver)
