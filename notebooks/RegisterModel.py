# Databricks notebook source
# MAGIC %pip install databricks-agents databricks-sdk openai>=1.3.5 mlflow --upgrade --pre
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import os

config = mlflow.models.ModelConfig(development_config="../config.yaml")

CATALOG = config.get("catalog")
SCHEMA = config.get("schema")
MODEL_NAME = config.get("model_name")
MLFLOW_EXPERIMENT_NAME = config.get("mlflow_experiment_name")

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksServingEndpoint
)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Workspace/Users/eric.golinko@databricks.com/{MLFLOW_EXPERIMENT_NAME}")

serving_endpoint = config.get("llm_endpoint_name")

# Log the model to MLflow
model_path = os.path.join(os.getcwd(), "pyfunc_model")

if not os.path.exists(model_path):
    model_path = os.path.join(os.getcwd(), "pyfunc_model.py")

with mlflow.start_run(run_name=f"log_mlflow_dais2025"):
    resources = [
        DatabricksServingEndpoint(endpoint_name=serving_endpoint),
    ]
    logged_chain_info = mlflow.pyfunc.log_model(
        python_model=os.path.join(
            os.getcwd(), model_path
        ),  # Chain code file e.g., /path/to/the/chain.py
        model_config=os.path.join(os.getcwd(), "../config.yaml"),  # Chain configuration
        artifact_path="chain",  # Required by MLflow
        input_example=config.get(
            "input_example"
        ),  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        registered_model_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
        resources=resources,
        pip_requirements=[
            "databricks-agents",
            "mlflow",
            "openai>=1.35.3",
            "databricks-sdk",
        ],
    )

# COMMAND ----------

# Test the chain locally
chain = mlflow.pyfunc.load_model(logged_chain_info.model_uri)
chain.predict(config.get("input_example"))

# COMMAND ----------


