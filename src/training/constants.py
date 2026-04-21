import os

from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"
N_FOLDS = 5
MAX_EVALS = 4
SPACE = {
    "random_state": scope.int(hp.quniform("random_state", 2, 80, 1)),
}
SPACE_LR = {
    "C": hp.loguniform("C", np.log(0.001), np.log(100)),
    "max_iter": scope.int(hp.quniform("max_iter", 100, 1000, 50)),
    "tol": hp.loguniform("tol", np.log(1e-6), np.log(1e-2)),
    "solver": hp.choice("solver", ["lbfgs", "saga", "liblinear"]),
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
