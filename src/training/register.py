import dagshub
import mlflow
import hydra
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from src.logger import ExecutorLogger


MODEL_NAMES = ["lr-classifier", "xgb-classifier", "rf-classifier"]


def get_best_model(client: MlflowClient, logger) -> tuple:
    best_model_name = None
    best_version = None
    best_accuracy = 0

    for model_name in MODEL_NAMES:
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.warning(f"No versions found for {model_name}, skipping")
                continue

            for v in versions:
                # Get the base model name e.g. "lr" from "lr-classifier"
                base_name = model_name.replace("-classifier", "")
                
                # Search evaluate runs in the experiment named after the model
                experiment = client.get_experiment_by_name(base_name)
                if experiment is None:
                    logger.warning(f"No experiment found for {base_name}")
                    continue

                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{base_name}-evaluate'",
                    order_by=["start_time DESC"],
                    max_results=1
                )

                if not runs:
                    logger.warning(f"No evaluate run found for {base_name}")
                    continue

                accuracy = runs[0].data.metrics.get("accuracy", 0)
                logger.info(f"{model_name} v{v.version} accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_name
                    best_version = v.version

        except Exception as e:
            logger.warning(f"Could not fetch {model_name}: {e}")

    return best_model_name, best_version, best_accuracy

def register_best_model(client: MlflowClient, model_name: str, version: str, logger) -> None:
    # Remove production alias from all models first
    for name in MODEL_NAMES:
        try:
            client.delete_registered_model_alias(name=name, alias="production")
            logger.info(f"Removed production alias from {name}")
        except Exception:
            pass  # alias didn't exist, that's fine

    # Set production alias on best model
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=version
    )
    logger.info(f"Promoted {model_name} v{version} to Production")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dagshub.init(
        repo_name="ITI-MLOps-Labs",
        repo_owner="mohamedabdelmonemelgohary"
    )

    logger = ExecutorLogger("register")
    client = MlflowClient()

    logger.info("Searching for best model across lr, xgb, rf...")
    best_name, best_version, best_accuracy = get_best_model(client, logger)

    if best_name is None:
        logger.error("No models found. Make sure train stages ran successfully.")
        return

    logger.info(f"Best model: {best_name} v{best_version} with accuracy {best_accuracy:.4f}")
    register_best_model(client, best_name, best_version, logger)

    # Verify
    prod = client.get_model_version_by_alias(best_name, "production")
    logger.info(f"✓ Confirmed: {best_name} v{prod.version} is in Production")
    logger.info("Registration complete. Check DagsHub Models tab.")


if __name__ == "__main__":
    main()