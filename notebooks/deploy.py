# Databricks notebook source
# MAGIC %pip install databricks-agents databricks-sdk openai>=1.3.5 mlflow --upgrade --pre
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.dropdown(
    name="catalog",
    defaultValue="test",
    choices=[
        "test", "dev", "staging", "prod",
    ],
)

# COMMAND ----------

import mlflow

config = mlflow.models.ModelConfig(development_config="../config.yaml")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = config.get("schema")
MODEL_NAME = config.get("model_name")

# COMMAND ----------

reviewer_instructions = f"""## Welcome to the Review App 

### Purpose: 

Feedback 

### Examples:

Examples

### Instructions: 

Instructions

"""

# COMMAND ----------

import os

os.environ["DATABRICKS_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get("mlflow_dais2025", "pat")

env_vars = {
    env_var: os.environ[env_var]
    for env_var in [
        "DATABRICKS_TOKEN",
        "DATABRICKS_HOST",
    ]
}

# COMMAND ----------

from utils import get_latest_model_version
from databricks import agents

model_version = get_latest_model_version(f"{CATALOG}.{SCHEMA}.{MODEL_NAME}")

agents.deploy(
    model_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
    model_version=model_version,
    environment_vars=env_vars,
)

agents.set_review_instructions(
    f"{CATALOG}.{SCHEMA}.{MODEL_NAME}", reviewer_instructions
)

# COMMAND ----------


