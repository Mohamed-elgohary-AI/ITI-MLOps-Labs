import os
import pickle

from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from .tuning.tuner import bayesian_opt
from .tuning.objectives import objective_lr, objective_xgb

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"
N_FOLDS = 5
MAX_EVALS = 5
SPACE = {
    "random_state": scope.int(hp.quniform("random_state", 2, 80, 1)),
}
SPACE_LR = {
    "C": hp.loguniform("C", np.log(0.001), np.log(100)),
    "max_iter": scope.int(hp.quniform("max_iter", 500, 5000, 100)),
    "tol": hp.loguniform("tol", np.log(1e-6), np.log(1e-2)),
    "penalty": hp.choice("penalty", ["l1", "l2", "elasticnet"]),
    "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),  # only used when penalty=elasticnet
    "random_state": scope.int(hp.quniform("random_state", 1, 100, 1)),
}
SPACE_XGB = {
    "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 50)),
    "max_depth": scope.int(hp.quniform("max_depth", 3, 12, 1)),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1.0),
    "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, 1)),
    "gamma": hp.loguniform("gamma", np.log(0.01), np.log(5)),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(10)),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-4), np.log(10)),
    "random_state": scope.int(hp.quniform("random_state", 1, 100, 1)),
}


def encode_target_col(
    cfg: DictConfig,
    logger,
):
    train_df = pd.read_parquet(os.path.join(SOURCE, f"{cfg.data.file_name}-train.parquet"))
    test_df = pd.read_parquet(os.path.join(SOURCE, f"{cfg.data.file_name}-test.parquet"))
    X_train, y_train = train_df.drop(cfg.data.target_col, axis=1), train_df[cfg.data.target_col]
    X_test, y_test = test_df.drop(cfg.data.target_col, axis=1), test_df[cfg.data.target_col]
    logger.info("Fitting the encoder/decoder of target variable")
    logger.info(f"Number of classes: {len(y_train.unique())}")
    encoder = {class_: idx for idx, class_ in enumerate(y_train.unique())}
    decoder = {idx: class_ for class_, idx in encoder.items()}
    # save the encoder/decoder of target
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


def trainer(X, y, cfg: DictConfig, logger) -> None:
    # Load translator from config path
    logger.info("Loading encoder/decoder of target variable")
    with open(cfg.paths.translator_path, "rb") as pkl:
        translator = pickle.load(pkl)

    y_train_enc = y.apply(lambda x: translator["encoder"][x])

    # Train whichever model is set in config
    model = bayesian_opt(X, y_train_enc, cfg, logger)

    # Save to config-driven path
    save_dir = os.path.join(cfg.paths.model_path, cfg.model.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "final_model.pkl")
    with open(save_path, "wb") as pkl:
        pickle.dump(model, pkl)

    logger.info(f"Model saved to {save_path}")
