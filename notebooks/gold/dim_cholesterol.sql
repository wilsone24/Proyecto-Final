-- Databricks notebook source
-- DBTITLE 1,Catalog
-- Set the active Unity Catalog for all subsequent table references.
USE CATALOG databricks_service_pf;

-- COMMAND ----------

-- DBTITLE 1,Create dim_cholesterol view
-- Cholesterol level dimension materialised as a view, filtered to codes present in the fact.
CREATE OR REPLACE VIEW gold.dim_cholesterol AS
WITH cholesterol_base AS (
  SELECT
    1 AS IdCholesterolType,
    'Normal' AS CholesterolTypeDescription
  UNION ALL
  SELECT
    2 AS IdCholesterolType,
    'Sobre lo normal' AS CholesterolTypeDescription
  UNION ALL
  SELECT
    3 AS IdCholesterolType,
    'Muy sobre lo normal' AS CholesterolTypeDescription
  UNION ALL
  SELECT
    4 AS IdCholesterolType,
    'Desconocido' AS CholesterolTypeDescription
)
SELECT
  d.IdCholesterolType,
  d.CholesterolTypeDescription
FROM
  cholesterol_base d
    LEFT SEMI JOIN gold.fct_cardio_outcomes f
      ON d.IdCholesterolType = f.IdCholesterolType;

-- COMMAND ----------

-- DBTITLE 1,Add view comment
-- Document the view in Unity Catalog for downstream BI / catalog discovery.
COMMENT ON VIEW gold.dim_cholesterol IS
'Cholesterol level dimension filtered by values present in gold.fct_cardio_outcomes.
1=Normal, 2=Above Normal, 3=Well Above Normal, 4=Unknown.
Source: gold.fct_cardio_outcomes (IdCholesterolType). Derived from silver cholesterol logic.';
