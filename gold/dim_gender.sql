-- Databricks notebook source
-- DBTITLE 1,Catalog
-- Set the active Unity Catalog for all subsequent table references.
USE CATALOG databricks_service_pf;

-- COMMAND ----------

-- DBTITLE 1,Create dim_gender view
-- Gender dimension materialised as a view, filtered to codes present in the fact.
CREATE OR REPLACE VIEW gold.dim_gender AS
WITH gender_base AS (
  SELECT
    1 AS IdGender,
    'Mujer' AS GenderDescription
  UNION ALL
  SELECT
    2 AS IdGender,
    'Hombre' AS GenderDescription
  UNION ALL
  SELECT
    3 AS IdGender,
    'Desconocido' AS GenderDescription
)
SELECT
  d.IdGender,
  d.GenderDescription
FROM
  gender_base d
    LEFT SEMI JOIN gold.fct_cardio_outcomes f
      ON d.IdGender = f.IdGender;

-- COMMAND ----------

-- DBTITLE 1,Add view comment
-- Document the view in Unity Catalog for downstream BI / catalog discovery.
COMMENT ON VIEW gold.dim_gender IS
'Gender dimension filtered by values present in gold.fct_cardio_outcomes.
1=Women, 2=Men, 3=Unknown.
Source: gold.fct_cardio_outcomes (IdGender). Derived from silver gender logic.';
