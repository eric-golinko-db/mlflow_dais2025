# Databricks notebook source
# the purpose of running eval this way is to evaluate a model before we register it

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

pred_agent = LLMChain()

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client
from mlflow.genai.scorers import global_guideline_adherence

# The guidelines below will be used to evaluate any response of the agent.
global_guidelines = ["If the request is unrelated to Databricks, the response must should be a rejection of the request",
    "If the request is related to Databricks, the response must should be concise",
    "If the request is related to Databricks and question about API, the response must have code",
    "The response must be professional."]

eval_set = [{
  "inputs": [{"role": "user", "content": "What is the difference between reduceByKey and groupByKey in Databricks Spark?"}]
}, {
  "inputs": [{"role": "user", "content": "What is the weather today?"}],
}]


# Define a very simple system-prompt agent.
@mlflow.trace(span_type="AGENT")
def llama3_agent(messages):
  SYSTEM_PROMPT = """
    You are a chatbot that answers questions about Databricks.
    For requests unrelated to Databricks, reject the request.
  """
  return get_deploy_client("databricks").predict(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    inputs={"messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages]}
  )


with mlflow.start_run(run_name="global_guidelines") as run:
  mlflow.genai.evaluate(
    data=eval_set,
    # predict_fn=llama3_agent,
    predict_fn=pred_agent.predict(context=None),
    scorers=[global_guideline_adherence(guidelines=global_guidelines)],
  )

# COMMAND ----------


