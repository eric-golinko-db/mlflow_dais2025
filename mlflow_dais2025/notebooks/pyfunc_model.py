# Databricks notebook source
# MAGIC %pip install --upgrade mlflow --pre databricks-agents==0.21.2 databricks-sdk==0.52.0  openai>=1.3.5
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import os
from typing import Any, Dict, List, Optional, Union

import mlflow
import pandas as pd
import yaml
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import (
    ChatCompletionResponse,
    ChatChoice,
    TokenUsageStats,
    ChatMessage,
)
from openai import OpenAI
from databricks.sdk import WorkspaceClient
import uuid

# COMMAND ----------

class LLMChain(ChatModel):
    """Entry point for RAG agent.

    Args:
        config: A .yaml file containing the configurations for the agent
    """

    def __init__(self):
        self.config = mlflow.models.ModelConfig(development_config="../config.yaml")
            
    @property
    def openai_client(self):
        w = WorkspaceClient()
        return w.serving_endpoints.get_open_ai_client()

    @mlflow.trace(span_type="PARSER")
    def extract_user_query_string(
        self, chat_messages_array: List[Union[Dict[str, str], ChatMessage]]
    ) -> str:
        """Extracts user dialogue from string
        Args:
            chat_messages_array: A list of dictionaries containing the chat history and the role of the user.
        Returns:
            A string containing the user's query
        Raises:
            ValueError: If the last message is not a string or a ChatMessage
        """

        last_message = chat_messages_array[-1]
        if isinstance(last_message, dict):
            return last_message["content"]
        elif isinstance(last_message, ChatMessage):
            return last_message.content
        else:
            raise ValueError(f"Unexpected message type: {type(last_message)}")

    @mlflow.trace(span_type="PARSER")
    def extract_chat_history(
        self, chat_messages_array: List[Union[Dict[str, str], ChatMessage]]
    ) -> List[Dict[str, str]]:

        """Extracts chat historty from chat message.
        Args:
            chat_messages_array: A list of dictionaries containing the chat history and the role of the user.
        Returns:
            A list of dictionaries containing the chat history and the role of the user.
        Raises:
            ValueError: If the last message is not a string or a ChatMessage

        """
        history = chat_messages_array[:-1]
        formatted_history = []
        for message in history:
            if isinstance(message, dict):
                formatted_history.append(message)
            elif isinstance(message, ChatMessage):
                formatted_history.append(
                    {"role": message.role, "content": message.content}
                )
            else:
                raise ValueError(f"Unexpected message type: {type(message)}")
        return formatted_history

    @mlflow.trace(span_type="LLM")
    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Rewrites the query based on the chat history.

        Args:
            query: The query to be rewritten
            chat_history: The chat history to be used for rewriting the query
        Returns:
            The rewritten query
        """

        if not chat_history:
            return query

        query_rewrite_template = """Assess the given chat history below by the following criteria: ensure that the chat history is related to the current question, if the history is related then we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. If the history is not related only focus on the current question. The query should be in natural language. Answer with only the query. Do not add explanation.

        Chat history: {chat_history}

        Question: {question}"""

        response = self.openai_client.chat.completions.create(
            model=self.config.get("llm_endpoint_name"),
            messages=[
                {"role": "system", "content": query_rewrite_template},
                {
                    "role": "user",
                    "content": f"Chat history: {json.dumps(chat_history)}\n\nQuestion: {query}",
                },
            ],
            **self.config.get("llm_config")["llm_parameters"],
        )
        return response.choices[0].message.content.strip()

    @mlflow.trace(span_type="PARSER")
    def format_chat_history_for_prompt(self, chat_messages_array):
        """
        Args:
            chat_messages_array: A list of dictionaries containing the chat history and the role of the user.
        Returns:
            A list of dictionaries containing the chat history and the role of the user.
        """

        history = self.extract_chat_history(chat_messages_array)
        formatted_chat_history = []
        if len(history) > 0:
            for chat_message in history:
                if chat_message["role"] == "user":
                    formatted_chat_history.append(
                        {"role": "user", "content": chat_message["content"]}
                    )
                elif chat_message["role"] == "assistant":
                    formatted_chat_history.append(
                        {"role": "assistant", "content": chat_message["content"]}
                    )
        return formatted_chat_history

    @mlflow.trace(span_type="LLM")
    def generate_answer(
        self, query: str, formatted_chat_history: List[Dict[str, str]]
    ) -> Any:
        """Generates an answer to the query based on the context and the chat history.
        Args:
            query: The query to be answered
            context: The context to be used for answering the query
            formatted_chat_history: A list of dictionaries containing the chat history and the role of the user.
        Returns:
            The answer generated by the LLM
        """

        messages = [
            {
                "role": "system",
                "content": self.config.get("llm_config")["system_prompt"],
            },
        ]
        messages.extend(formatted_chat_history)
        messages.append({"role": "user", "content": query})

        response = self.openai_client.chat.completions.create(
            model=self.config.get("llm_endpoint_name"),
            messages=messages,
            **self.config.get("llm_config")["llm_parameters"],
        )

        return response

    
    def convert_chat_response(
        self, response: Any, user_name: str
    ) -> ChatCompletionResponse:
        """Reformulates the chat reponse to a consistent format for output.

        Args:
            response: The response from the LLM.
            retrieved_docs: The retrieved documents.
            user_name: The name of the user.
        Returns:
            A ChatCompletionResponse object
        """

        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            model=response.model,
            choices=[
                ChatChoice(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                        "name": user_name,
                    },
                    finish_reason="stop",
                )
            ],
            usage=TokenUsageStats(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    @mlflow.trace(name="chain", span_type="CHAIN")
    def run_chain(
        self, messages: List[Dict[str, str]], user_name: str
    ) -> ChatCompletionResponse:
        """Runs the RAG chain.
        Args:
            messages (List[Dict[str, str]]): The messages from the chat.
            user_name (str): The name of the user.
        Returns:
            ChatCompletionResponse: The response from the RAG chain.
        """

        user_query = self.extract_user_query_string(messages)
        chat_history = self.extract_chat_history(messages)

        rewritten_query = self.rewrite_query(user_query, chat_history)
        formatted_chat_history = self.format_chat_history_for_prompt(messages)

        response = self.generate_answer(user_query, formatted_chat_history)

        return self.convert_chat_response(response, user_name)

    
    @mlflow.trace(name="pyfunc.predict", span_type="CHAIN")
    def predict(
        self,
        context: mlflow.pyfunc.model.PythonModelContext,
        messages: List[Dict[str, str]],
        params: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletionResponse:
        """Given the input and context of the model, supplies the output.
        Args:
            context (mlflow.pyfunc.model.PythonModelContext): The context of the model.
            messages (List[Dict[str, str]]): The messages from the chat.
            params (Optional[Dict[str, Any]])
        Returns:
            ChatCompletionResponse: The response from the RAG chain.
        """

        if isinstance(messages[-1], dict):
            if messages[-1].get("name") is None:
                user_name = "placeholder@email.com"
            else:
                user_name = messages[-1]["name"]
        elif isinstance(messages[-1], ChatMessage):
            user_name = (
                messages[-1].name
                if messages[-1].name != None
                else "placeholder@email.com"
            )
        else:
            user_name = "placeholder@email.com"

        return self.run_chain(messages, user_name)


# Tell MLflow logging where to find your chain.
# `mlflow.models.set_model(model=...)` function specifies the chain to use for evaluation and deployment.
# This is required to log this chain to MLflow with `mlflow.pyfunc.log_model(...)`.
mlflow.models.set_model(model=LLMChain())

# COMMAND ----------

# lm = LLMChain()

# COMMAND ----------

# r = lm.predict(
#     context=None,
#     messages=[
#         {
#             "role": "user",
#             "content": "How many years has Databricks been a company?",
#         }
#     ],
# )

# COMMAND ----------


