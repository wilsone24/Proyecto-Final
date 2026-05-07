-- Databricks notebook source
-- DBTITLE 1,Catalog
USE CATALOG pf1;

-- COMMAND ----------

-- DBTITLE 1,Create dimglucose View
CREATE OR REPLACE VIEW gold.dimglucose AS
WITH glucose_base AS (
    SELECT 1 AS IdGlucoseType, 'Normal' AS GlucoseTypeDescription
    UNION ALL
    SELECT 2 AS IdGlucoseType, 'Above Normal' AS GlucoseTypeDescription
    UNION ALL
    SELECT 3 AS IdGlucoseType, 'Well Above Normal' AS GlucoseTypeDescription
    UNION ALL
    SELECT 4 AS IdGlucoseType, 'Unknown' AS GlucoseTypeDescription
)
SELECT d.IdGlucoseType, d.GlucoseTypeDescription
FROM glucose_base d
LEFT SEMI JOIN gold.factcardio f
    ON d.IdGlucoseType = f.IdGlucoseType;

-- COMMAND ----------

-- DBTITLE 1,Add view comment
COMMENT ON VIEW gold.dimglucose IS 
'Glucose level dimension filtered by values present in silver.cardiosilver. 
1=Normal, 2=Above Normal, 3=Well Above Normal, 4=Unknown';
