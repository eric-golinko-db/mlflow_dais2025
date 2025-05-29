# Databricks notebook source
# the purpose of running eval this way is to evaluate a model before we register it

# COMMAND ----------

# MAGIC %pip install databricks-agents databricks-sdk openai>=1.3.5 mlflow --upgrade --pre
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./pyfunc_model

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.genai.scorers import scorer

evals = [
  {
    "inputs": {"messages": [{"role": "user", "content": "Good morning"}]},
  }, {
    "inputs": {"inputs": ["Good afternoon"]},
  }, {
    "inputs": {"inputs": [1, 2, 3, 4]},
  }
]

@scorer
def check_valid_format(inputs, outputs=None, expectations=None, traces=None):
  # Check that the request contains a top-level key called "inputs" with a value of a list
  return "inputs" in inputs and isinstance(inputs.get("inputs"), list)


with mlflow.start_run(run_name="check_format"):
  eval_results = mlflow.genai.evaluate(
      data=pd.DataFrame.from_records(evals),
      scorers=[check_valid_format],
  )
eval_results.result_df

# COMMAND ----------


