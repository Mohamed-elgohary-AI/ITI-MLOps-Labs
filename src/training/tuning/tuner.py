from functools import partial

import numpy as np

from hydra.utils import instantiate
from hyperopt import Trials, fmin, tpe, hp
from hyperopt.pyll import scope
from .objectives import objective_lr, objective_xgb, objective_rf
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

OBJECTIVES = {
    "lr": objective_lr,
    "xgb": objective_xgb,
    "rf": objective_rf
}

MODEL_BUILDERS = {
    "lr": lambda params: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ]
    ),
    "xgb": lambda params: XGBClassifier(**params),
    "rf": lambda params: RandomForestClassifier(**params)
}

def build_space(space_cfg) -> dict:
    space = {}
    for param, spec in space_cfg.items():
        t = spec.type
        if t == "loguniform":
            space[param] = hp.loguniform(param, float(spec.low), float(spec.high))
        elif t == "uniform":
            space[param] = hp.uniform(param, float(spec.low), float(spec.high))
        elif t == "quniform":
            dist = hp.quniform(param, float(spec.low), float(spec.high), float(spec.step))
            space[param] = scope.int(dist) if spec.get("int", False) else dist
        elif t == "choice":
            space[param] = hp.choice(param, list(spec.options))
        else:
            raise ValueError(f"Unknown space type: {t}")
    return space


def bayesian_opt(X, y_train_enc, cfg: DictConfig, logger):
    model_name = cfg.model.name
    logger.info(f"{model_name} optimization started")

    space = build_space(cfg.model.space)

    bayes_trials = Trials()
    fmin_objective = partial(OBJECTIVES[model_name], X=X, y=y_train_enc)

    fmin(
        fn=fmin_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=cfg.tuning.max_evals,
        trials=bayes_trials,
    )

    logger.info("Optimization completed")

    best = bayes_trials.results[np.argmin([r["loss"] for r in bayes_trials.results])]
    final_model = MODEL_BUILDERS[model_name](best["params"])
    final_model.fit(X, y_train_enc)

    return final_model
