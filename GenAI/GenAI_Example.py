# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow 3 GenAI Example
# MAGIC
# MAGIC In this example, we will create an agent and then evaluate its performance. First, we will define the agent and log it to MLflow.

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow --pre langchain-community databricks-agents databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

# Define the chain
chat_model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.1,
    max_tokens=2000,
)
prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a chatbot that can answer questions about Databricks."),
    ("user", "{messages}"),
  ]
)
chain = prompt | chat_model

# Log the chain with MLflow, specifying its parameters
# As a new feature, the LoggedModel entity is linked to its name and params
model_info = mlflow.langchain.log_model(
  lc_model=chain,
  artifact_path="basic_chain",
  params={
    "temperature": 0.1,
    "max_tokens": 2000,
    "prompt_template": str(prompt)
  },
  model_type="agent",
  input_example={"messages": "What is MLflow?"},
)

# Inspect the LoggedModel and its properties
logged_model = mlflow.get_logged_model(model_info.model_id)
print(logged_model.model_id, logged_model.params)
# m-123802d4ba324f4d8baa456eb8b5c061, {'max_tokens': '2000', 'prompt_template': "input_variables=['messages'] messages=[SystemMessagePromptTemplate(...), HumanMessagePromptTemplate(...)]", 'temperature': '0.1'}


# COMMAND ----------

# MAGIC %md
# MAGIC Then, we will interactively query the chain in a notebook to make sure that it’s viable enough for further testing. These traces can be viewed in UI, under the Traces tab of the model details page.

# COMMAND ----------

# Enable autologging so that interactive traces from the chain are automatically linked to its LoggedModel
mlflow.langchain.autolog()
loaded_chain = mlflow.langchain.load_model(f"models:/{logged_model.model_id}")
chain_inputs = [
  {"messages": "What is MLflow?"},
  {"messages": "What is Unity Catalog?"},
  {"messages": "What are user-defined functions (UDFs)?"}
]

for chain_input in chain_inputs:
  loaded_chain.invoke(chain_input)

# COMMAND ----------

# MAGIC %md
# MAGIC Assuming the chain is viable, we will run an evaluation job against the agent to determine whether it’s good enough for further QA by subject matter experts. We can construct a dataset from the agent’s responses on various inputs.
# MAGIC
# MAGIC This example will produce one MLflow Run (evaluation_run), one MLflow Logged Model (basic_chain), and traces from the interactive query and evaluation. We can then see all evaluation metrics for the agent.
# MAGIC

# COMMAND ----------

# Prepare the eval dataset in a pandas DataFrame
import pandas as pd
eval_df = pd.DataFrame(
  {
    "request": [
      "What is MLflow Tracking and how does it work?",
      "What is Unity Catalog?",
      "What are user-defined functions (UDFs)?"
    ],
    "expected_response": [
      """MLflow Tracking is a key component of the MLflow platform designed to record and manage machine learning experiments. It enables data scientists and engineers to log parameters, code versions, metrics, and artifacts in a systematic way, facilitating experiment tracking and reproducibility.\n\nHow It Works:\n\nAt the heart of MLflow Tracking is the concept of a run, which is an execution of a machine learning code. Each run can log the following:\n\nParameters: Input variables or hyperparameters used in the model (e.g., learning rate, number of trees). Metrics: Quantitative measures to evaluate the model's performance (e.g., accuracy, loss). Artifacts: Output files like models, datasets, or images generated during the run. Source Code: The version of the code or Git commit hash used. These logs are stored in a tracking server, which can be set up locally or on a remote server. The tracking server uses a backend storage (like a database or file system) to keep a record of all runs and their associated data.\n\n Users interact with MLflow Tracking through its APIs available in multiple languages (Python, R, Java, etc.). By invoking these APIs in the code, you can start and end runs, and log data as the experiment progresses. Additionally, MLflow offers autologging capabilities for popular machine learning libraries, automatically capturing relevant parameters and metrics without manual code changes.\n\nThe logged data can be visualized using the MLflow UI, a web-based interface that displays all experiments and runs. This UI allows you to compare runs side-by-side, filter results, and analyze performance metrics over time. It aids in identifying the best models and understanding the impact of different parameters.\n\nBy providing a structured way to record experiments, MLflow Tracking enhances collaboration among team members, ensures transparency, and makes it easier to reproduce results. It integrates seamlessly with other MLflow components like Projects and Model Registry, offering a comprehensive solution for managing the machine learning lifecycle.""",
      """Unity Catalog is a feature in Databricks that allows you to create a centralized inventory of your data assets, such as tables, views, and functions, and share them across different teams and projects. It enables easy discovery, collaboration, and reuse of data assets within your organization.\n\nWith Unity Catalog, you can:\n\n1. Create a single source of truth for your data assets: Unity Catalog acts as a central repository of all your data assets, making it easier to find and access the data you need.\n2. Improve collaboration: By providing a shared inventory of data assets, Unity Catalog enables data scientists, engineers, and other stakeholders to collaborate more effectively.\n3. Foster reuse of data assets: Unity Catalog encourages the reuse of existing data assets, reducing the need to create new assets from scratch and improving overall efficiency.\n4. Enhance data governance: Unity Catalog provides a clear view of data assets, enabling better data governance and compliance.\n\nUnity Catalog is particularly useful in large organizations where data is scattered across different teams, projects, and environments. It helps create a unified view of data assets, making it easier to work with data across different teams and projects.""",
      """User-defined functions (UDFs) in the context of Databricks and Apache Spark are custom functions that you can create to perform specific tasks on your data. These functions are written in a programming language such as Python, Java, Scala, or SQL, and can be used to extend the built-in functionality of Spark.\n\nUDFs can be used to perform complex data transformations, data cleaning, or to apply custom business logic to your data. Once defined, UDFs can be invoked in SQL queries or in DataFrame transformations, allowing you to reuse your custom logic across multiple queries and applications.\n\nTo use UDFs in Databricks, you first need to define them in a supported programming language, and then register them with the SparkSession. Once registered, UDFs can be used in SQL queries or DataFrame transformations like any other built-in function.\n\nHere\'s an example of how to define and register a UDF in Python:\n\n```python\nfrom pyspark.sql.functions import udf\nfrom pyspark.sql.types import IntegerType\n\n# Define the UDF function\ndef multiply_by_two(value):\n    return value * 2\n\n# Register the UDF with the SparkSession\nmultiply_udf = udf(multiply_by_two, IntegerType())\n\n# Use the UDF in a DataFrame transformation\ndata = spark.range(10)\nresult = data.withColumn("multiplied", multiply_udf(data.id))\nresult.show()\n```\n\nIn this example, we define a UDF called `multiply_by_two` that multiplies a given value by two. We then register this UDF with the SparkSession using the `udf` function, and use it in a DataFrame transformation to multiply the `id` column of a DataFrame by two."""
    ],
  }
)

# COMMAND ----------

# Start a run to represent the evaluation job
with mlflow.start_run() as evaluation_run:
  eval_dataset: mlflow.entities.Dataset = mlflow.data.from_pandas(
    df=eval_df,
    name="eval_dataset",
  )
  # Run the agent evaluation 
  result = mlflow.evaluate(
    model=f"models:/{logged_model.model_id}",
    data=eval_dataset,
    model_type="databricks-agent"
  )
  # Log evaluation metrics and associate with agent
  mlflow.log_metrics(
    metrics=result.metrics,
    dataset=eval_dataset,
    # Specify the ID of the agent logged above
    model_id=logged_model.model_id
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Now register the model to UC. You can also see the model ID, parameters, and metrics in the UC Model Version page

# COMMAND ----------

CATALOG = "eric_golinko"
SCHEMA = "genai"
MODEL_NAME = "genai_model"

mlflow.register_model(logged_model.model_uri, name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}")

# COMMAND ----------


