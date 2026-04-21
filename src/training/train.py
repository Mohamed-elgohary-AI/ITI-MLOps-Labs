from functools import partial
import os
import pickle
from typing import Any, Dict

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.fake.estimator import FakeEstimator

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
    file_name: str,
    target_col: str,
    model_name: str,
    logger,
):
    train_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-train.parquet"))
    test_df = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-test.parquet"))
    X_train, y_train = train_df.drop(target_col, axis=1), train_df[target_col]
    X_test, y_test = test_df.drop(target_col, axis=1), test_df[target_col]
    logger.info("Fitting the encoder/decoder of target variable")
    logger.info(f"Number of classes: {len(y_train.unique())}")
    encoder = {class_: idx for idx, class_ in enumerate(y_train.unique())}
    decoder = {idx: class_ for class_, idx in encoder.items()}
    # save the encoder/decoder of target
    label_translator = {"encoder": encoder, "decoder": decoder}
    logger.info("encoder/decoder of target created successfully")
    if not os.path.exists(os.path.join(MODEL_PATH, model_name)):
        os.makedirs(os.path.join(MODEL_PATH, model_name))
    with open(
        os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"),
        "wb",
    ) as pkl:
        pickle.dump(label_translator, pkl)
    logger.info("encoder/decoder of target saved")
    return X_train, y_train, X_test, y_test


def objective(params: Dict[str, Any], X, y, n_folds: int = N_FOLDS) -> Dict[str, Any]:
    model = FakeEstimator(**params)
    scores = cross_validate(model, X, y, cv=n_folds, n_jobs=-1, scoring="accuracy")
    score = np.mean(scores["test_score"])
    return {"loss": score, "params": params, "status": STATUS_OK}


def objective_lr(params, X, y):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=params["C"],
                    max_iter=params["max_iter"],
                    tol=params["tol"],
                    solver="saga",
                    penalty=params["penalty"],
                    l1_ratio=params["l1_ratio"] if params["penalty"] == "elasticnet" else None,
                    random_state=params["random_state"],
                ),
            ),
        ]
    )

    score = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1).mean()

    return {"loss": -score, "status": STATUS_OK, "params": params}


def objective_xgb(params, X, y):
    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        colsample_bylevel=params["colsample_bylevel"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=params["random_state"],
        use_label_encoder=False,
        eval_metric="logloss",
    )

    score = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1).mean()

    return {"loss": -score, "status": STATUS_OK, "params": params}


def bayesian_hyperparameter_lr(X, y_train_enc, logger):
    logger.info("Logistc Regression optimization started")

    bayes_trials = Trials()
    fmin_objective = partial(objective_lr, X=X, y=y_train_enc)

    fmin(
        fn=fmin_objective,
        space=SPACE_LR,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=bayes_trials,
    )

    logger.info("optimization completed")

    best_model = bayes_trials.results[np.argmin([r["loss"] for r in bayes_trials.results])]
    params = best_model["params"]

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ]
    )
    final_model.fit(X, y_train_enc)

    return final_model


def bayesian_hyperparameter_xgb(X, y_train_enc, logger):
    logger.info("XGBoost optimization started")

    bayes_trials = Trials()
    fmin_objective = partial(objective_xgb, X=X, y=y_train_enc)

    fmin(
        fn=fmin_objective,
        space=SPACE_XGB,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=bayes_trials,
    )

    logger.info("optimization completed")

    best_model = bayes_trials.results[np.argmin([r["loss"] for r in bayes_trials.results])]
    params = best_model["params"]

    final_model = XGBClassifier(**params, use_label_encoder=False)
    final_model.fit(X, y_train_enc)

    return final_model


def trainer(X, y, model_name_1: str, model_name_2: str, logger) -> None:
    logger.info("Load encoder/decoder of target variable")
    with open(
        os.path.join(MODEL_PATH, "fake", "model_target_translator.pkl"),
        "rb",
    ) as pkl:
        translator = pickle.load(pkl)
    y_train_enc = y.apply(lambda x: translator["encoder"][x])
    lr_model = bayesian_hyperparameter_lr(X, y_train_enc, logger)
    xgb_model = bayesian_hyperparameter_xgb(X, y_train_enc, logger)

    if not os.path.exists(os.path.join(MODEL_PATH, model_name_1)):
        os.makedirs(os.path.join(MODEL_PATH, model_name_1))
    with open(os.path.join(MODEL_PATH, model_name_1, "final_model.pkl"), "wb") as pkl:
        pickle.dump(lr_model, pkl)

    if not os.path.exists(os.path.join(MODEL_PATH, model_name_2)):
        os.makedirs(os.path.join(MODEL_PATH, model_name_2))
    with open(os.path.join(MODEL_PATH, model_name_2, "final_model.pkl"), "wb") as pkl:
        pickle.dump(xgb_model, pkl)
    logger.info("model trained and saved successfully")
