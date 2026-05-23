-- Databricks notebook source
-- DBTITLE 1,Catalog
-- Set the active Unity Catalog for all subsequent table references.
USE CATALOG databricks_service_pf;

-- COMMAND ----------

-- DBTITLE 1,Create dim_glucose view
-- Glucose level dimension materialised as a view, filtered to codes present in the fact.
CREATE OR REPLACE VIEW gold.dim_glucose AS
WITH glucose_base AS (
  SELECT
    1 AS IdGlucoseType,
    'Normal' AS GlucoseTypeDescription
  UNION ALL
  SELECT
    2 AS IdGlucoseType,
    'Sobre lo normal' AS GlucoseTypeDescription
  UNION ALL
  SELECT
    3 AS IdGlucoseType,
    'Muy sobre lo normal' AS GlucoseTypeDescription
  UNION ALL
  SELECT
    4 AS IdGlucoseType,
    'Desconocido' AS GlucoseTypeDescription
)
SELECT
  d.IdGlucoseType,
  d.GlucoseTypeDescription
FROM
  glucose_base d
    LEFT SEMI JOIN gold.fct_cardio_outcomes f
      ON d.IdGlucoseType = f.IdGlucoseType;

-- COMMAND ----------

-- DBTITLE 1,Add view comment
-- Document the view in Unity Catalog for downstream BI / catalog discovery.
COMMENT ON VIEW gold.dim_glucose IS
'Glucose level dimension filtered by values present in gold.fct_cardio_outcomes.
1=Normal, 2=Above Normal, 3=Well Above Normal, 4=Unknown.
Source: gold.fct_cardio_outcomes (IdGlucoseType). Derived from silver gluc logic.';
