-- Databricks notebook source
-- DBTITLE 1,Catalog
USE CATALOG databricks_service_pf;

-- COMMAND ----------

-- DBTITLE 1,Create dimagegroup View
CREATE OR REPLACE VIEW gold.dimagegroup AS
WITH age_group_base AS (
  SELECT
    1 AS IdAgeGroup,
    '<30' AS AgeGroupDescription,
    0 AS AgeMin,
    29 AS AgeMax
  UNION ALL
  SELECT
    2 AS IdAgeGroup,
    '30-44' AS AgeGroupDescription,
    30 AS AgeMin,
    44 AS AgeMax
  UNION ALL
  SELECT
    3 AS IdAgeGroup,
    '45-59' AS AgeGroupDescription,
    45 AS AgeMin,
    59 AS AgeMax
  UNION ALL
  SELECT
    4 AS IdAgeGroup,
    '60-74' AS AgeGroupDescription,
    60 AS AgeMin,
    74 AS AgeMax
  UNION ALL
  SELECT
    5 AS IdAgeGroup,
    '>=75' AS AgeGroupDescription,
    75 AS AgeMin,
    999 AS AgeMax
  UNION ALL
  SELECT
    6 AS IdAgeGroup,
    'Unknown' AS AgeGroupDescription,
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
    LEFT SEMI JOIN gold.factcardio f
      ON d.IdAgeGroup = f.IdAgeGroup;

-- COMMAND ----------

-- DBTITLE 1,Add view comment
COMMENT ON VIEW gold.dimagegroup IS
'Age group dimension filtered by values present in gold.factcardio.
1=<30, 2=30-44, 3=45-59, 4=60-74, 5=>=75, 6=Unknown.
AgeMin and AgeMax are inclusive bounds in full years.
Source: gold.factcardio (IdAgeGroup). Derived from silver age_group_id logic.';
