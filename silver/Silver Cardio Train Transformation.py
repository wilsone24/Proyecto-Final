# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
spark.sql(f"USE CATALOG `pf1`")

# COMMAND ----------

# DBTITLE 1,Import libraries
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# COMMAND ----------

# DBTITLE 1,Define source and target
sourceSchema = "bronze"
sourceTable  = "cardio_bronze_train"

schemaName = "silver"
tableName  = "cardio_silver_train"

# COMMAND ----------

# DBTITLE 1,Read from Bronze Delta table
df = spark.table(f"{sourceSchema}.{sourceTable}")

# COMMAND ----------

# DBTITLE 1,Exploratory Data Analysis (EDA) - row count and nulls
print(f"Total rows: {df.count()}")
df.printSchema()
display(df.summary())

# COMMAND ----------

# DBTITLE 1,Null counts per column
null_counts = df.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
)
display(null_counts)

# COMMAND ----------

# DBTITLE 1,Impute nulls with column mean (numerical columns)
numerical_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]

for col_name in numerical_cols:
    mean_val = df.select(F.mean(F.col(col_name))).collect()[0][0]
    df = df.withColumn(
        col_name,
        F.when(F.col(col_name).isNull(), F.lit(mean_val)).otherwise(F.col(col_name))
    )

categorical_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]

for col_name in categorical_cols:
    mode_val = (
        df.groupBy(col_name)
          .count()
          .orderBy(F.desc("count"))
          .first()[0]
    )
    df = df.withColumn(
        col_name,
        F.when(F.col(col_name).isNull(), F.lit(mode_val)).otherwise(F.col(col_name))
    )

# COMMAND ----------

# DBTITLE 1,Convert age from days to years
df = df.withColumn("age_years", F.round(F.col("age") / 365.25).cast(IntegerType()))
df = df.drop("age").withColumnRenamed("age_years", "age")

# COMMAND ----------

# DBTITLE 1,Filter physiologically impossible outliers
df = df.filter(
    (F.col("ap_hi").between(60, 250)) &
    (F.col("ap_lo").between(40, 160)) &
    (F.col("ap_hi") > F.col("ap_lo")) &
    (F.col("height").between(100, 250)) &
    (F.col("weight").between(30, 200))
)

print(f"Rows after outlier removal: {df.count()}")

# COMMAND ----------

# DBTITLE 1,Standardize column types
df = (
    df
    .withColumn("weight", F.col("weight").cast(DoubleType()))
    .withColumn("gender",      F.col("gender").cast(IntegerType()))
    .withColumn("cholesterol", F.col("cholesterol").cast(IntegerType()))
    .withColumn("gluc",        F.col("gluc").cast(IntegerType()))
    .withColumn("smoke",       F.col("smoke").cast(IntegerType()))
    .withColumn("alco",        F.col("alco").cast(IntegerType()))
    .withColumn("active",      F.col("active").cast(IntegerType()))
    .withColumn("cardio",      F.col("cardio").cast(IntegerType()))
)

# COMMAND ----------

# DBTITLE 1,Add ingestion metadata
df = (
    df
    .withColumn("ingestion_date", F.current_timestamp())
    .withColumn("source_table", F.lit(f"{sourceSchema}.{sourceTable}"))
)

# COMMAND ----------

# DBTITLE 1,Write to Silver Delta table
(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{schemaName}.{tableName}")
)

#print(f"Silver table '{schemaName}.{tableName}' written successfully.")
#display(spark.table(f"{schemaName}.{tableName}").limit(10))
