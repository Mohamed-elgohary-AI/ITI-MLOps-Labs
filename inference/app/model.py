import os
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "xgb-classifier"

def load_production_model():
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    mlflow.set_tracking_uri(
        "https://dagshub.com/mohamedabdelmonemelgohary/ITI-MLOps-Labs.mlflow"
    )

    client = MlflowClient()
    prod_version = client.get_model_version_by_alias(MODEL_NAME, "production")
    model_uri = f"models:/{MODEL_NAME}@production"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded {MODEL_NAME} v{prod_version.version} from production")

    return model