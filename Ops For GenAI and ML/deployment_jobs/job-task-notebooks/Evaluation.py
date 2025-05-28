# Databricks notebook source
# MAGIC %md
# MAGIC This notebook should only be run in a Databricks Job, as part of MLflow 3.0 Deployment Jobs.

# COMMAND ----------

# MAGIC %pip install mlflow --upgrade --pre
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("model_name", "eric_golinko.ops.classic_ml_model")
dbutils.widgets.text("model_version", "1")

# COMMAND ----------

import pandas as pd
from sklearn.datasets import load_iris

def sample_iris_data():
  iris = load_iris()
  iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
  iris_df['quality'] = (iris.target == 2).astype(int)
  return iris_df

# COMMAND ----------

def get_latest_model_version(model_name: str) -> int:
    """Gets the latest version of a model.

    Args:
     model_name: The name of the model to get the latest version for
    Returns:
     The latest version of the model
    """

    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# TODO: add evaluation dataset and target here
data = sample_iris_data()
target = "quality"
# TODO: add model type here (e.g. "regressor", "databricks-agent", etc.)
model_type = "regressor"

model_name = dbutils.widgets.get("model_name")
model_version = get_latest_model_version(model_name=model_name)
model_uri = "models:/" + model_name + "/" + str(model_version)

with mlflow.start_run(run_name="evaluation") as run:
  mlflow.evaluate(
    model=model_uri,
    data=data,
    targets=target,
    model_type=model_type
  )

print(model_uri)

# COMMAND ----------



# COMMAND ----------


