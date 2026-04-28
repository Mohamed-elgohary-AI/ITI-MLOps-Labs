# objectives.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def objective_lr(params, X, y):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ]
    )
    score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
    return {"loss": -score, "status": "ok", "params": params}


def objective_xgb(params, X, y):
    int_params = ["n_estimators", "max_depth", "min_child_weight", "random_state"]
    params = {
        k: int(v) if k in int_params else v
        for k, v in params.items()
    }

    model = XGBClassifier(**params, eval_metric="logloss")
    score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
    return {"loss": -score, "status": "ok", "params": params}


def objective_rf(params, X, y):
    int_params = ["n_estimators", "max_depth", "min_samples_split", "random_state"]
    params = {
        k: int(v) if k in int_params else v
        for k, v in params.items()
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
    return {"loss": -score, "status": "ok", "params": params}