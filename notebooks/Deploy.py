# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch databricks-agents==1.8.2 mlflow==3.6.0 backoff uv
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.dropdown(
    name="catalog",
    defaultValue="dev",
    choices=[
        "dev", "staging", "prod",
    ],
)

# COMMAND ----------

import mlflow

config = mlflow.models.ModelConfig(development_config="../config.yaml")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = config.get("schema")
MODEL_NAME = config.get("model_name")

# COMMAND ----------

from utils import get_latest_model_version
from databricks import agents

model_version = get_latest_model_version(f"{CATALOG}.{SCHEMA}.{MODEL_NAME}")

agents.deploy(
    model_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
    model_version=model_version,
)

# COMMAND ----------


