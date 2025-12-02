# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch databricks-agents==1.8.2 mlflow==3.6.0 backoff uv
# MAGIC %restart_python

# COMMAND ----------

import mlflow

config = mlflow.models.ModelConfig(development_config="../config.yml")

# COMMAND ----------

dbutils.widgets.dropdown(
    name="catalog",
    defaultValue="dev",
    choices=[
        "dev",
        "staging",
        "prod",
    ],
)

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = config.get("schema")
model_name = f'{config.get("model_name")}'

UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import warnings
# MAGIC from typing import Any, Generator
# MAGIC
# MAGIC import backoff
# MAGIC import mlflow
# MAGIC import openai
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from databricks.vector_search.reranker import DatabricksReranker
# MAGIC
# MAGIC
# MAGIC THRESHOLD = 0.1
# MAGIC
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC SYSTEM_PROMPT = """
# MAGIC You are a helpful assistant that provides brief, clear responses.
# MAGIC """
# MAGIC
# MAGIC
# MAGIC class SimpleChatAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     Simple chat agent that calls an LLM using the Databricks OpenAI client API.
# MAGIC
# MAGIC     You can replace this with your own agent.
# MAGIC     The decorators @mlflow.trace tell MLflow Tracing to track calls to the agent.
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self):
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         self.client = self.workspace_client.serving_endpoints.get_open_ai_client()
# MAGIC         self.llm_endpoint = LLM_ENDPOINT_NAME
# MAGIC         self.SYSTEM_PROMPT = SYSTEM_PROMPT
# MAGIC     
# MAGIC
# MAGIC     @backoff.on_exception(backoff.expo, openai.RateLimitError)
# MAGIC     @mlflow.trace(span_type=SpanType.LLM)
# MAGIC     def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
# MAGIC         with warnings.catch_warnings():
# MAGIC             warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
# MAGIC             for chunk in self.client.chat.completions.create(
# MAGIC                 model=self.llm_endpoint,
# MAGIC                 messages=to_chat_completions_input(messages),
# MAGIC                 stream=True,
# MAGIC             ):
# MAGIC                 yield chunk.to_dict()
# MAGIC
# MAGIC     def get_context(self, query: str):
# MAGIC         docs = self.vector_retriever.query_segment(query)
# MAGIC         context = [doc["segmentDescription"] for doc in docs]
# MAGIC         return " ".join(context)
# MAGIC
# MAGIC     # With autologging, you do not need @mlflow.trace here, but you can add it to override the span type.
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     # With autologging, you do not need @mlflow.trace here, but you can add it to override the span type.
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
# MAGIC             i.model_dump() for i in request.input
# MAGIC         ]
# MAGIC         yield from output_to_responses_items_stream(chunks=self.call_llm(messages))
# MAGIC
# MAGIC
# MAGIC mlflow.openai.autolog()
# MAGIC AGENT = SimpleChatAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

import mlflow
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]

with mlflow.start_run(run_name=f"{catalog}_mflow_dais2025_model"):
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        pip_requirements=[
            "databricks-agents==1.8.2",
            "mlflow==3.6.0",
        ],
        resources=resources,
        model_type="agent",
        input_example={"input": [{"role": "user", "content": "What is 5+5?"}]}
        )

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri=f"models:/{logged_agent_info.model_id}")
model = model.predict({"input": [{"role": "user", "content": "What is 5+5?"}]})


# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Before you deploy the agent, you must register the agent to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# register the model to Unity Catalog
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_model.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md Now you can view the model version and all centralized performance data on the model version page in Unity Catalog. You can also get the same information using the API as shown in the following cell.

# COMMAND ----------

# Get the model version
from mlflow import MlflowClient

client = MlflowClient()
model_version = client.get_model_version(
    name=UC_MODEL_NAME, version=uc_registered_model_info.version
)
print(model_version)

# COMMAND ----------


