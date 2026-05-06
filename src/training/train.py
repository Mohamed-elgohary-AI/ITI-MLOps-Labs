import os
import pickle

import dagshub
import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll import scope
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import Pipeline

from src.logger import ExecutorLogger
from .tuning.tuner import bayesian_opt
from .process_data import preprocess_data

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"
PROCESSED_DATA_DIR = os.path.join("data", "processed")


def encode_target_col(cfg: DictConfig, logger):
    train_df = pd.read_parquet(os.path.join(SOURCE, f"{cfg.data.file_name}-train.parquet"))
    test_df = pd.read_parquet(os.path.join(SOURCE, f"{cfg.data.file_name}-test.parquet"))
    X_train, y_train = train_df.drop(cfg.data.target_col, axis=1), train_df[cfg.data.target_col]
    X_test, y_test = test_df.drop(cfg.data.target_col, axis=1), test_df[cfg.data.target_col]
    logger.info("Fitting the encoder/decoder of target variable")
    logger.info(f"Number of classes: {len(y_train.unique())}")
    encoder = {class_: idx for idx, class_ in enumerate(y_train.unique())}
    decoder = {idx: class_ for class_, idx in encoder.items()}
    label_translator = {"encoder": encoder, "decoder": decoder}
    logger.info("encoder/decoder of target created successfully")
    if not os.path.exists(os.path.join(MODEL_PATH, cfg.model.name)):
        os.makedirs(os.path.join(MODEL_PATH, cfg.model.name))
    with open(
        os.path.join(MODEL_PATH, cfg.model.name, "model_target_translator.pkl"),
        "wb",
    ) as pkl:
        pickle.dump(label_translator, pkl)
    logger.info("encoder/decoder of target saved")
    return X_train, y_train, X_test, y_test


def trainer(X, y, preprocessor, cfg: DictConfig, logger) -> None:
    logger.info("Loading encoder/decoder of target variable")
    with open(cfg.paths.translator_path, "rb") as pkl:
        translator = pickle.load(pkl)

    y_train_enc = y.apply(lambda x: translator["encoder"][x])
    model = bayesian_opt(X, y_train_enc, cfg, logger)

    if hasattr(model, "best_score_"):
        mlflow.log_metric("best_score", model.best_score_)
    if hasattr(model, "best_iteration"):
        mlflow.log_metric("best_iteration", model.best_iteration)

    # Combine preprocessor and model into a single pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    save_dir = os.path.join(cfg.paths.model_path, cfg.model.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "final_model.pkl")
    with open(save_path, "wb") as pkl:
        pickle.dump(full_pipeline, pkl)

    mlflow.sklearn.log_model(
        sk_model=full_pipeline,
        artifact_path="model",
        registered_model_name=f"{cfg.model.name}-classifier"
    )
    logger.info(f"Full pipeline saved to {save_path}")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dagshub.init(
        repo_name="ITI-MLOps-Labs",
        repo_owner="mohamedabdelmonemelgohary"
    )
    mlflow.set_experiment(cfg.model.name)

    with mlflow.start_run(run_name=f"{cfg.model.name}-train"):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        logger = ExecutorLogger("training")
        OmegaConf.save(cfg, "config.yaml", resolve=True)
        X_train, y_train, X_test, y_test = encode_target_col(cfg, logger)
        X_processed, X_test_processed, preprocessor = preprocess_data(
            X_train, y_train, X_test, y_test, logger
        )
        trainer(X_processed, y_train, preprocessor, cfg, logger)


if __name__ == "__main__":
    main()