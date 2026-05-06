import json
import os
import pickle

import dagshub
import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from skore import EstimatorReport
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    confusion_matrix,
)

from src.logger import ExecutorLogger
from .process_data import preprocess_data

PROCESSED_DATA_DIR = os.path.join("data", "processed")


def evaluate(X_test, y_test, cfg: DictConfig, logger) -> None:
    save_dir = cfg.model.save_dir
    model_path = os.path.join(cfg.paths.model_path, save_dir, "final_model.pkl")
    report_dir = os.path.join(cfg.paths.reports_path, save_dir)

    # Load model (full pipeline)
    logger.info(f"Loading {cfg.model.name} model")
    with open(model_path, "rb") as pkl:
        final_model = pickle.load(pkl)

    # Load translator
    with open(cfg.paths.translator_path, "rb") as pkl:
        translator = pickle.load(pkl)

    y_test_enc = y_test.apply(lambda x: translator["encoder"][x])

    # Get predictions directly from full pipeline (no preprocessing needed)
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    # Generate skore report
    logger.info("Creating evaluation report")
    final_report = EstimatorReport(final_model, X_test=X_test, y_test=y_test_enc)

    evaluation_report = {
        "model_name": cfg.model.name,
        "estimator_name": final_report.estimator_name_,
        "accuracy": final_report.metrics.accuracy(),
        "precision": final_report.metrics.precision(),
        "recall": final_report.metrics.recall(),
        "balanced_accuracy": balanced_accuracy_score(y_test_enc, y_pred),
        "roc_auc": roc_auc_score(y_test_enc, y_pred_proba),
        "matthews_corrcoef": matthews_corrcoef(y_test_enc, y_pred),
        "cohen_kappa": cohen_kappa_score(y_test_enc, y_pred),
        "log_loss": log_loss(y_test_enc, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test_enc, y_pred).tolist(),
    }

    # Log metrics to MLflow
    mlflow.log_metrics({
        "accuracy": evaluation_report["accuracy"],
        "balanced_accuracy": evaluation_report["balanced_accuracy"],
        "precision_class_0": evaluation_report["precision"][0],
        "precision_class_1": evaluation_report["precision"][1],
        "recall_class_0": evaluation_report["recall"][0],
        "recall_class_1": evaluation_report["recall"][1],
        "roc_auc": evaluation_report["roc_auc"],
        "matthews_corrcoef": evaluation_report["matthews_corrcoef"],
        "cohen_kappa": evaluation_report["cohen_kappa"],
        "log_loss": evaluation_report["log_loss"],
    })

    # Log confusion matrix as artifact
    os.makedirs(report_dir, exist_ok=True)
    cm_path = os.path.join(report_dir, "confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump({"confusion_matrix": evaluation_report["confusion_matrix"]}, f, indent=4)
    mlflow.log_artifact(cm_path)

    # Save and log full report
    logger.info("Saving evaluation report")
    report_path = os.path.join(report_dir, "evaluation_report.json")
    with open(report_path, "w") as js:
        json.dump(evaluation_report, js, indent=4)
    logger.info(f"Report saved to {report_path}")
    mlflow.log_artifact(report_path)


def load_test_data(cfg: DictConfig, logger):
    train_df = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, f"{cfg.data.file_name}-train.parquet")
    )
    test_df = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, f"{cfg.data.file_name}-test.parquet")
    )

    X_train = train_df.drop(cfg.data.target_col, axis=1)
    y_train = train_df[cfg.data.target_col]
    X_test = test_df.drop(cfg.data.target_col, axis=1)
    y_test = test_df[cfg.data.target_col]

    return X_test, y_test


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dagshub.init(
        repo_name="ITI-MLOps-Labs",
        repo_owner="mohamedabdelmonemelgohary"
    )
    mlflow.set_experiment(cfg.model.name)

    with mlflow.start_run(run_name=f"{cfg.model.name}-evaluate"):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        logger = ExecutorLogger("evaluate")
        X_test, y_test = load_test_data(cfg, logger)
        evaluate(X_test, y_test, cfg, logger)


if __name__ == "__main__":
    main()