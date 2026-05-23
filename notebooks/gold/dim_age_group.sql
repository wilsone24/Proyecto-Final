-- Databricks notebook source
-- DBTITLE 1,Catalog
-- Set the active Unity Catalog for all subsequent table references.
USE CATALOG databricks_service_pf;

-- COMMAND ----------

-- DBTITLE 1,Create dim_age_group view
-- Age group dimension materialised as a view, filtered to codes present in the fact.
CREATE OR REPLACE VIEW gold.dim_age_group AS
WITH age_group_base AS (
  SELECT
    1 AS IdAgeGroup,
    'Menor de 30' AS AgeGroupDescription,
    0 AS AgeMin,
    29 AS AgeMax
  UNION ALL
  SELECT
    2 AS IdAgeGroup,
    '30 a 44 años' AS AgeGroupDescription,
    30 AS AgeMin,
    44 AS AgeMax
  UNION ALL
  SELECT
    3 AS IdAgeGroup,
    '45 a 59 años' AS AgeGroupDescription,
    45 AS AgeMin,
    59 AS AgeMax
  UNION ALL
  SELECT
    4 AS IdAgeGroup,
    '60 a 74 años' AS AgeGroupDescription,
    60 AS AgeMin,
    74 AS AgeMax
  UNION ALL
  SELECT
    5 AS IdAgeGroup,
    '75 años o más' AS AgeGroupDescription,
    75 AS AgeMin,
    999 AS AgeMax
  UNION ALL
  SELECT
    6 AS IdAgeGroup,
    'Desconocido' AS AgeGroupDescription,
    NULL AS AgeMin,
    NULL AS AgeMax
)
SELECT
  d.IdAgeGroup,
  d.AgeGroupDescription,
  d.AgeMin,
  d.AgeMax
FROM
  age_group_base d
    LEFT SEMI JOIN gold.fct_cardio_outcomes f
      ON d.IdAgeGroup = f.IdAgeGroup;

-- COMMAND ----------

-- DBTITLE 1,Add view comment
-- Document the view in Unity Catalog for downstream BI / catalog discovery.
COMMENT ON VIEW gold.dim_age_group IS
'Age group dimension filtered by values present in gold.fct_cardio_outcomes.
1=<30, 2=30-44, 3=45-59, 4=60-74, 5=>=75, 6=Unknown.
AgeMin and AgeMax are inclusive bounds in full years.
Source: gold.fct_cardio_outcomes (IdAgeGroup). Derived from silver age_group_id logic.';
