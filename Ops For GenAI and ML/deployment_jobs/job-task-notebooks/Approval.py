# Databricks notebook source
# MAGIC %md
# MAGIC This notebook should only be run in a Databricks Job, as part of MLflow 3.0 Deployment Jobs.

# COMMAND ----------

# MAGIC %pip install mlflow --upgrade --pre
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("model_name", "dev.ops.classic_ml_model")
dbutils.widgets.text("model_version", "")
dbutils.widgets.text("approval_tag_name", "Approval_Check")

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
tag_name = dbutils.widgets.get("approval_tag_name")
tags = client.get_model_version(model_name, model_version).tags
if not any(tag.lower() == tag_name.lower() for tag in tags.keys()):
  raise Exception("Model version not approved for deployment")
else:
  if tags.get(tag_name).lower() == "approved":
    print("Model version approved for deployment")
  else:
    raise Exception("Model version not approved for deployment")
