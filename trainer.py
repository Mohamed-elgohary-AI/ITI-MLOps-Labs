from src.logger import ExecutorLogger
from src.training.download_data import download_data
from src.training.evaluate import evaluate
from src.training.process_data import read_process_data, preprocess_data
from src.training.train import encode_target_col, trainer
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = ExecutorLogger("training")
    logger.info("Training started")
    download_data(logger)
    read_process_data(cfg, logger)
    X, y, X_test, y_test = encode_target_col(cfg, logger)
    X_processed, X_test_procesed=preprocess_data(X, y, X_test, y_test, logger)
    trainer(X_processed, y, cfg,logger)
    logger.info("Training finished")
    evaluate(X_test_procesed, y_test,cfg,logger)
    logger.info("Evaluation Finished")


if __name__ == "__main__":
    main()