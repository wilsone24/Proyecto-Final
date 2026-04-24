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

# DBTITLE 1,Define source and target
sourceSchema = "silver"
sourceTable  = "cardio_silver_train"

schemaName = "gold"
tableName  = "cardio_gold_train"

# COMMAND ----------

# DBTITLE 1,Read from Silver Delta table
df = spark.table(f"{sourceSchema}.{sourceTable}")

print(f"Total rows: {df.count()}")
display(df.limit(5))

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
