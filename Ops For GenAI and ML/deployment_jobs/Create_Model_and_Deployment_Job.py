# Databricks notebook source
# MAGIC %md
# MAGIC Running this notebook will automatically create a Registered Model in Unity Catalog, as well as a Databricks Job templated as a MLflow 3.0 Deployment Job.

# COMMAND ----------

# MAGIC %pip install mlflow  --upgrade --pre
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# TODO: Update these values as necessary

CATALOG = "eric_golinko"
SCHEMA = "ops"

model_name = f"{CATALOG}.{SCHEMA}.classic_ml_model"
job_name = "deployment_job"

# TODO: Create notebooks for each task and populate the notebook path here
evaluation_notebook_path = "/Workspace/Users/eric.golinko@databricks.com/What's New in MLflow 3.0/Ops For GenAI and ML/deployment_jobs/job-task-notebooks/Evaluation"
approval_notebook_path = "/Workspace/Users/eric.golinko@databricks.com/What's New in MLflow 3.0/Ops For GenAI and ML/deployment_jobs/job-task-notebooks/Approval"
deployment_notebook_path = "/Workspace/Users/eric.golinko@databricks.com/What's New in MLflow 3.0/Ops For GenAI and ML/deployment_jobs/job-task-notebooks/Deployment"

# COMMAND ----------

# Create job with necessary configuration to connect to model as deployment job
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
job_settings = jobs.JobSettings(
    name=job_name,
    tasks=[
        jobs.Task(
            task_key="Evaluation",
            notebook_task=jobs.NotebookTask(notebook_path=evaluation_notebook_path),
            # disable_auto_optimization=True,
            max_retries=0,
        ),
        jobs.Task(
            task_key="Approval_Check",
            notebook_task=jobs.NotebookTask(
                notebook_path=approval_notebook_path,
                base_parameters={"approval_tag_name": "{{task.name}}"},
            ),
            depends_on=[jobs.TaskDependency(task_key="Evaluation")],
            # disable_auto_optimization=True,
            max_retries=0,
        ),
        jobs.Task(
            task_key="Deployment",
            notebook_task=jobs.NotebookTask(notebook_path=deployment_notebook_path),
            depends_on=[jobs.TaskDependency(task_key="Approval_Check")],
            # disable_auto_optimization=True,
            max_retries=0,
        ),
    ],
    parameters=[
        jobs.JobParameter(name="model_name", default=model_name),
        jobs.JobParameter(name="model_version", default=""),
    ],
    queue=jobs.QueueSettings(enabled=True),
    max_concurrent_runs=1,
)

created_job = w.jobs.create(**job_settings.__dict__)
job_id = created_job.job_id

# COMMAND ----------

# Create registered model w/ linked deployment job
from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
try:
    client.create_registered_model(model_name, deployment_job_id=job_id)
except Exception:
    # print(traceback.format_exc())
    print("good job")

# COMMAND ----------


