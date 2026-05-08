import numpy as np
import litserve as ls
import pickle
import os
import pandas as pd
from src.deployment.online.inference_request import InferenceRequest
import dagshub
import mlflow

class InferenceAPI(ls.LitAPI):

    def setup(self, device="cpu"):
        dagshub.init(
            repo_name="ITI-MLOps-Labs",
            repo_owner="mohamedabdelmonemelgohary"
        )
        self.model = mlflow.sklearn.load_model("models:/rf-classifier@production")

    def decode_request(self, request):
        try:
            data = request["input"]

            # SINGLE RECORD
            if isinstance(data, dict):
                InferenceRequest(**data)

                x = pd.DataFrame([data])
                return x

            # BATCH REQUEST
            elif isinstance(data, list):
                for item in data:
                    InferenceRequest(**item)

                x = pd.DataFrame(data)
                return x

            else:
                raise ValueError("Invalid input format")

        except Exception as e:
            print("Decode error:", e)
            return None

    def predict(self, x):
        if x is None:
            return None

        return self.model.predict(x)

    def encode_response(self, output):
        if output is None:
            return {
                "message": "error",
                "prediction": None
            }

        return {
            "message": "success",
            "prediction": output.tolist()
        }