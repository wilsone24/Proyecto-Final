# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
spark.sql(f"USE CATALOG `pf1`")

# COMMAND ----------

# DBTITLE 1,ingest data from the volume in the lakehouse
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, DoubleType

# COMMAND ----------

bronze = "/Volumes/pf1/bronze/dataset/cardio_train.csv"

schemaName = "bronze" 

tableName = "cardio_bronze_train"

schema = StructType([
    StructField("id", IntegerType(), nullable=True),
    StructField("age", IntegerType(), nullable=True),
    StructField("gender", IntegerType(), nullable=True),
    StructField("height", IntegerType(), nullable=True),
    StructField("weight", DoubleType(),  nullable=True),
    StructField("ap_hi", IntegerType(), nullable=True),
    StructField("ap_lo", IntegerType(), nullable=True),
    StructField("cholesterol", IntegerType(), nullable=True),
    StructField("gluc", IntegerType(), nullable=True),
    StructField("smoke", IntegerType(), nullable=True),
    StructField("alco", IntegerType(), nullable=True),
    StructField("active", IntegerType(), nullable=True),
    StructField("cardio", IntegerType(), nullable=True),
])

# COMMAND ----------

stage = (
    spark.read
    .option("header", True)
    .option("delimiter", ";")
    .schema(schema)
    .csv(bronze)
)

(
    stage.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f'{schemaName}.{tableName}')
)
