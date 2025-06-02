from mlflow import MlflowClient


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