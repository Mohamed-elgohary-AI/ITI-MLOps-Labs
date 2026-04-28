import json
import os
import pickle

from omegaconf import DictConfig
from skore import EstimatorReport


def evaluate(X_test, y_test, cfg: DictConfig, logger) -> None:
    save_dir = cfg.model.save_dir

    model_path = os.path.join(cfg.paths.model_path, save_dir, "final_model.pkl")
    report_dir = os.path.join(cfg.paths.reports_path, save_dir)

    # Load model
    logger.info("Loading model")
    with open(model_path, "rb") as pkl:
        final_model = pickle.load(pkl)

    # Load translator
    with open(cfg.paths.translator_path, "rb") as pkl:
        translator = pickle.load(pkl)

    y_test_enc = y_test.apply(lambda x: translator["encoder"][x])

    # Generate report
    logger.info("Creating evaluation report")
    final_report = EstimatorReport(final_model, X_test=X_test, y_test=y_test_enc)

    evaluation_report = {
        "model_name": cfg.model.name,
        "estimator_name": final_report.estimator_name_,
        "fitting_time": final_report.fit_time_,
        "accuracy": final_report.metrics.accuracy(),
        "precision": final_report.metrics.precision(),
        "recall": final_report.metrics.recall(),
        "prediction_time": final_report.metrics.timings(),
    }

    # Save report
    logger.info("Saving evaluation report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "evaluation_report.json")
    with open(report_path, "w") as js:
        json.dump(evaluation_report, js, indent=4)

    logger.info(f"Report saved to {report_path}")