# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Cardiovascular Disease — Model Serving Endpoint
# MAGIC
# MAGIC Deploys the current `@champion` model version as a Databricks Model Serving REST endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Setup and configuration

# COMMAND ----------

# DBTITLE 1,Catalog
# Set the active Unity Catalog for all subsequent table references.
spark.sql("USE CATALOG `databricks_service_pf`")

# COMMAND ----------

# DBTITLE 1,Libraries
# MLflow for the model registry, Databricks SDK for the serving endpoint, pandas for smoke-test data.
import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
from datetime import timedelta

import mlflow
from mlflow.tracking import MlflowClient

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)
from databricks.sdk.errors import ResourceDoesNotExist

# COMMAND ----------

# DBTITLE 1,Parameters
# Job-level inputs: source feature table, target model, endpoint config.
dbutils.widgets.text("source_schema",          "gold")
dbutils.widgets.text("source_table",           "cardio_features")
dbutils.widgets.text("registered_model_name",  "databricks_service_pf.gold.cardio_classifier")
dbutils.widgets.text("endpoint_name",          "cardio-classifier-endpoint")
dbutils.widgets.text("workload_size",          "Small")
dbutils.widgets.text("scale_to_zero",          "true")

# COMMAND ----------

# DBTITLE 1,Constants
# Validate widgets and derive runtime constants (endpoint config, tags, timeout, alias, served-entity name).
SOURCE_SCHEMA         = dbutils.widgets.get("source_schema")
SOURCE_TABLE          = dbutils.widgets.get("source_table")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name")
ENDPOINT_NAME         = dbutils.widgets.get("endpoint_name")
WORKLOAD_SIZE         = dbutils.widgets.get("workload_size")
SCALE_TO_ZERO         = dbutils.widgets.get("scale_to_zero").lower() == "true"

# Fail fast if any required widget was not provided
if not all([SOURCE_SCHEMA, SOURCE_TABLE, REGISTERED_MODEL_NAME, ENDPOINT_NAME, WORKLOAD_SIZE]):
    raise ValueError(
        f"Missing required widgets: source_schema='{SOURCE_SCHEMA}', "
        f"source_table='{SOURCE_TABLE}', "
        f"registered_model_name='{REGISTERED_MODEL_NAME}', "
        f"endpoint_name='{ENDPOINT_NAME}', workload_size='{WORKLOAD_SIZE}'"
    )

# Model invariants (not environment-specific)
TARGET_COLUMN  = "cardio"
CHAMPION_ALIAS = "champion"

# Features excluded from training — must match ml_train_cardio_classifier.py.
# The smoke test drops these from the payload so the request matches what the
# served model expects.
EXCLUDED_FEATURES = ["hypertension", "pulse_pressure", "age_group_id"]

# Smoke-test configuration
SMOKE_TEST_ROWS = 3

# Derived constants
FULL_SOURCE         = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
PIPELINE_VERSION    = "1.0.0"
SERVED_ENTITY_NAME  = "cardio-champion"
READY_TIMEOUT       = timedelta(minutes=30)  # max wait for endpoint to be READY

# Endpoint tags — keys must be snake_case (Databricks Serving rejects dots/colons/slashes).
ENDPOINT_TAGS = [
    EndpointTag(key="pipeline_version", value=PIPELINE_VERSION),
    EndpointTag(key="data_subject",     value="cardiovascular-disease"),
    EndpointTag(key="data_layer",       value="gold"),
    EndpointTag(key="data_owner",       value="data-engineering"),
    EndpointTag(key="model_name",       value=REGISTERED_MODEL_NAME),
    EndpointTag(key="model_alias",      value=CHAMPION_ALIAS),
]

print(f"Registered model:  {REGISTERED_MODEL_NAME}")
print(f"Champion alias:    @{CHAMPION_ALIAS}")
print(f"Endpoint name:     {ENDPOINT_NAME}")
print(f"Workload size:     {WORKLOAD_SIZE}")
print(f"Scale to zero:     {SCALE_TO_ZERO}")

# COMMAND ----------

# DBTITLE 1,Clients
# Initialise MLflow (Unity Catalog registry) and Databricks SDK clients used throughout the notebook.
try:
    mlflow.set_registry_uri("databricks-uc")
    mlflow_client    = MlflowClient()
    workspace_client = WorkspaceClient()
    workspace_host   = workspace_client.config.host
    print(f"Workspace host: {workspace_host}")

except Exception as e:
    raise Exception(f"[Clients] Failed to initialise clients: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Resolve current champion version

# COMMAND ----------

# DBTITLE 1,Get version pointed to by alias
# Look up which model version currently holds the @champion alias.
try:
    champion_info    = mlflow_client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS,
    )
    champion_version = champion_info.version

    champion_metadata = {
        "version":     champion_version,
        "run_id":      champion_info.run_id,
        "description": (champion_info.description or "")[:200],
        "tags":        dict(champion_info.tags) if champion_info.tags else {},
    }
    print(f"Champion resolved: version {champion_version}")
    print(json.dumps(champion_metadata, indent=2, default=str))

except mlflow.exceptions.RestException as e:
    raise Exception(
        f"[Champion] No @{CHAMPION_ALIAS} alias found on {REGISTERED_MODEL_NAME}. "
        f"Run ml_train_cardio_classifier.py first to register an initial model. Details: {e}"
    )

except Exception as e:
    raise Exception(f"[Champion] Failed to resolve champion alias: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Create or update the serving endpoint

# COMMAND ----------

# DBTITLE 1,Build endpoint config
# Build the ServedEntityInput describing what to serve (model, version, compute size, autoscaling).
try:
    served_entity = ServedEntityInput(
        entity_name=           REGISTERED_MODEL_NAME,
        entity_version=        champion_version,
        workload_size=         WORKLOAD_SIZE,
        scale_to_zero_enabled= SCALE_TO_ZERO,
        name=                  SERVED_ENTITY_NAME,
    )

    endpoint_config = EndpointCoreConfigInput(
        served_entities=[served_entity],
    )

    print(f"Config: entity={REGISTERED_MODEL_NAME} v{champion_version}, "
          f"size={WORKLOAD_SIZE}, scale_to_zero={SCALE_TO_ZERO}")

except Exception as e:
    raise Exception(f"[Config] Failed to build endpoint config: {e}")

# COMMAND ----------

# DBTITLE 1,Check if endpoint exists
# Decide upfront whether we'll CREATE (no endpoint yet) or UPDATE (endpoint already exists).
existing_endpoint = None

try:
    existing_endpoint = workspace_client.serving_endpoints.get(name=ENDPOINT_NAME)
    print(f"Endpoint {ENDPOINT_NAME} already exists.")

    served = existing_endpoint.config.served_entities if existing_endpoint.config else []
    if served:
        current_version = served[0].entity_version
        current_name    = served[0].entity_name
        print(f"  Currently serving: {current_name} v{current_version}")

except ResourceDoesNotExist:
    print(f"Endpoint {ENDPOINT_NAME} does not exist — will be created.")

except Exception as e:
    raise Exception(f"[Endpoint Lookup] Unexpected error checking endpoint: {e}")

# COMMAND ----------

# DBTITLE 1,Create or update
# Idempotent dispatch: CREATE if absent, UPDATE if a new version is needed, UNCHANGED otherwise.
try:
    if existing_endpoint is None:
        print(f"Creating endpoint {ENDPOINT_NAME}... (this can take 5-15 minutes)")
        endpoint = workspace_client.serving_endpoints.create_and_wait(
            name=    ENDPOINT_NAME,
            config=  endpoint_config,
            tags=    ENDPOINT_TAGS,
            timeout= READY_TIMEOUT,
        )
        action = "CREATED"
    else:
        # Detect whether the served version has changed
        served   = existing_endpoint.config.served_entities if existing_endpoint.config else []
        same_ver = (
            len(served) == 1
            and served[0].entity_name        == REGISTERED_MODEL_NAME
            and str(served[0].entity_version) == str(champion_version)
        )

        if same_ver:
            print(f"Endpoint already serving v{champion_version} — no update needed.")
            endpoint = existing_endpoint
            action   = "UNCHANGED"
        else:
            print(f"Updating endpoint to serve v{champion_version}... (this can take 5-15 minutes)")
            endpoint = workspace_client.serving_endpoints.update_config_and_wait(
                name=            ENDPOINT_NAME,
                served_entities= endpoint_config.served_entities,
                timeout=         READY_TIMEOUT,
            )
            action = "UPDATED"

    print(f"Endpoint {action}: {ENDPOINT_NAME}")

except Exception as e:
    raise Exception(f"[Endpoint Create/Update] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Verify endpoint is READY

# COMMAND ----------

# DBTITLE 1,Confirm ready state
# Re-fetch the endpoint and fail loudly if it isn't fully READY before running the smoke test.
try:
    status       = workspace_client.serving_endpoints.get(name=ENDPOINT_NAME)
    ready_state  = status.state.ready         if status.state else None
    update_state = status.state.config_update if status.state else None

    print(f"Ready state:        {ready_state}")
    print(f"Config update:      {update_state}")

    # Extract just the enum value name so we match "READY" exactly,
    # avoiding the substring trap where "READY" is contained in "NOT_READY".
    state_name = str(ready_state).rsplit(".", 1)[-1] if ready_state else ""
    if state_name != "READY":
        raise Exception(
            f"Endpoint {ENDPOINT_NAME} is not READY (state={ready_state}). "
            f"Check the Serving UI for logs."
        )

except Exception as e:
    raise Exception(f"[Endpoint Status] Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Smoke test

# COMMAND ----------

# DBTITLE 1,Load sample rows
# Pull a handful of rows; drop the target and the excluded features so the payload matches the model schema.
try:
    sample_pdf = (
        spark.table(FULL_SOURCE)
             .limit(SMOKE_TEST_ROWS)
             .toPandas()
    )
    if TARGET_COLUMN in sample_pdf.columns:
        actual_labels = sample_pdf[TARGET_COLUMN].astype(int).tolist()
        sample_pdf    = sample_pdf.drop(columns=[TARGET_COLUMN])
    else:
        actual_labels = None

    # Drop features the model was trained without (would cause schema mismatch).
    excluded_present = [c for c in EXCLUDED_FEATURES if c in sample_pdf.columns]
    if excluded_present:
        sample_pdf = sample_pdf.drop(columns=excluded_present)
        print(f"Dropped excluded features from payload: {excluded_present}")

    print(f"Loaded {len(sample_pdf)} rows for smoke test")
    display(sample_pdf)

except Exception as e:
    raise Exception(f"[Smoke Test Load] Failed to load sample rows: {e}")

# COMMAND ----------

# DBTITLE 1,Query the endpoint
# Send the sample rows to the live endpoint and compare predictions against actual labels.
try:
    sample_records = sample_pdf.to_dict(orient="records")
    response = workspace_client.serving_endpoints.query(
        name=              ENDPOINT_NAME,
        dataframe_records= sample_records,
    )

    predictions = response.predictions
    print("Endpoint response:")
    print(json.dumps(predictions, indent=2, default=str))

    predictions_df = pd.DataFrame(predictions)
    if actual_labels is not None:
        predictions_df["actual"] = actual_labels
    display(predictions_df)

except Exception as e:
    raise Exception(f"[Smoke Test Query] Endpoint query failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Invocation instructions

# COMMAND ----------

# DBTITLE 1,Endpoint URL and snippets
# Build copy-paste curl / Python / SDK snippets for external consumers of the endpoint.
try:
    invocation_url = f"{workspace_host}/serving-endpoints/{ENDPOINT_NAME}/invocations"

    example_record = sample_records[0] if sample_records else {}
    body_example   = {"dataframe_records": [example_record]}

    curl_snippet = (
        f"curl -X POST '{invocation_url}' \\\n"
        f"     -H 'Authorization: Bearer <DATABRICKS_TOKEN>' \\\n"
        f"     -H 'Content-Type: application/json' \\\n"
        f"     -d '{json.dumps(body_example)}'"
    )

    python_snippet = (
        "import requests, os\n"
        f"url     = '{invocation_url}'\n"
        "headers = {\n"
        "    'Authorization': f\"Bearer {os.environ['DATABRICKS_TOKEN']}\",\n"
        "    'Content-Type':  'application/json',\n"
        "}\n"
        f"body = {json.dumps(body_example, indent=4)}\n"
        "response = requests.post(url, headers=headers, json=body)\n"
        "print(response.json())"
    )

    sdk_snippet = (
        "from databricks.sdk import WorkspaceClient\n"
        "w = WorkspaceClient()\n"
        f"response = w.serving_endpoints.query(\n"
        f"    name='{ENDPOINT_NAME}',\n"
        f"    dataframe_records=[{example_record}],\n"
        f")\n"
        "print(response.predictions)"
    )

    print("=" * 80)
    print(f"ENDPOINT URL:")
    print(f"  {invocation_url}")
    print()
    print("=" * 80)
    print("CURL")
    print("=" * 80)
    print(curl_snippet)
    print()
    print("=" * 80)
    print("PYTHON (requests)")
    print("=" * 80)
    print(python_snippet)
    print()
    print("=" * 80)
    print("PYTHON (Databricks SDK — recommended inside the workspace)")
    print("=" * 80)
    print(sdk_snippet)

except Exception as e:
    raise Exception(f"[Instructions] Failed to build snippets: {e}")

# COMMAND ----------

# DBTITLE 1,Serving summary
# Final summary block so the run is easy to interpret from the notebook output.
print("=" * 70)
print("SERVING ENDPOINT SUMMARY")
print("=" * 70)
print(f"Endpoint name:        {ENDPOINT_NAME}")
print(f"Action:               {action}")
print(f"Serving model:        {REGISTERED_MODEL_NAME} v{champion_version}")
print(f"Alias tracked:        @{CHAMPION_ALIAS}")
print(f"Workload size:        {WORKLOAD_SIZE}")
print(f"Scale to zero:        {SCALE_TO_ZERO}")
print(f"Invocation URL:       {invocation_url}")
print("=" * 70)
print()
print("NEXT STEPS:")
print(f"  - Open Catalog Explorer → {REGISTERED_MODEL_NAME} → Serving tab to monitor.")
print(f"  - When ml_promote_cardio_classifier.py promotes a new @champion, re-run this notebook to update.")
print(f"  - For demo: copy the curl snippet above and run it from any terminal.")
