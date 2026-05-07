-- Databricks notebook source
-- DBTITLE 1,Catalog
USE CATALOG pf1;

-- COMMAND ----------

-- DBTITLE 1,Create dimgender View
CREATE OR REPLACE VIEW gold.dimgender AS
WITH gender_base AS (
    SELECT 1 AS IdGender, 'Women' AS GenderDescription
    UNION ALL
    SELECT 2 AS IdGender, 'Men' AS GenderDescription
    UNION ALL
    SELECT 3 AS IdGender, 'Unknown' AS GenderDescription
)
SELECT d.IdGender, d.GenderDescription
FROM gender_base d
LEFT SEMI JOIN gold.factcardio f
    ON d.IdGender = f.IdGender;

-- COMMAND ----------

-- DBTITLE 1,Add view comment
COMMENT ON VIEW gold.dimgender IS 
'Gender dimension filtered by values present in silver.cardiosilver. 
1=Women, 2=Men 3=Unknown';
