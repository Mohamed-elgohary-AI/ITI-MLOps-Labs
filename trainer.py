from src.logger import ExecutorLogger
from src.training.download_data import download_iris_data
from src.training.evaluate import evaluate
from src.training.process_data import read_process_data, preprocess_data
from src.training.train import encode_target_col, trainer


def main(logger) -> None:
    logger.info("Training started")
    download_iris_data(logger)
    read_process_data("train", "PassengerId", "Survived", logger)
    X, y, X_test, y_test = encode_target_col("train", "Survived", "fake", logger)
    X_processed, X_test_procesed=preprocess_data(X, y, X_test, y_test, logger)
    trainer(X_processed, y, "lr","xgb", logger)
    evaluate(X_test_procesed, y_test, "lr", logger)
    evaluate(X_test_procesed, y_test, "xgb", logger)
    logger.info("Training finished")


if __name__ == "__main__":
    logger = ExecutorLogger("training")
    main(logger)