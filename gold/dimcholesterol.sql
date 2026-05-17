-- Databricks notebook source
-- DBTITLE 1,Catalog
USE CATALOG databricks_service_pf;

-- COMMAND ----------

-- DBTITLE 1,Create dimcholesterol View
CREATE OR REPLACE VIEW gold.dimcholesterol AS
WITH cholesterol_base AS (
    SELECT 1 AS IdCholesterolType, 'Normal' AS CholesterolTypeDescription
    UNION ALL
    SELECT 2 AS IdCholesterolType, 'Above Normal' AS CholesterolTypeDescription
    UNION ALL
    SELECT 3 AS IdCholesterolType, 'Well Above Normal' AS CholesterolTypeDescription
    UNION ALL
    SELECT 4 AS IdCholesterolType, 'Unknown' AS CholesterolTypeDescription
)
SELECT d.IdCholesterolType, d.CholesterolTypeDescription
FROM cholesterol_base d
LEFT SEMI JOIN gold.factcardio f
    ON d.IdCholesterolType = f.IdCholesterolType;


-- COMMAND ----------

-- DBTITLE 1,Add view comment
COMMENT ON VIEW gold.dimcholesterol IS 
'Cholesterol level dimension filtered by values present in silver.cardiosilver. 
1=Normal, 2=Above Normal, 3=Well Above Normal, 4=Unknown';
