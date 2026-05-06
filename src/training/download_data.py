import os
import shutil

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
import kagglehub

from src.logger import ExecutorLogger


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = ExecutorLogger("download")
    logger.info(f"Downloading {cfg.data.dataset_name} dataset from Kaggle...")
    load_dotenv()

    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN")

    path = kagglehub.competition_download(cfg.data.dataset_name)

    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        csv_files = files

    os.makedirs(cfg.data.raw_data_path, exist_ok=True)

    for csv_file in csv_files:
        source_file = os.path.join(path, csv_file)
        destination = os.path.join(cfg.data.raw_data_path, csv_file)
        shutil.copy(source_file, destination)
        logger.info(f"Copied {csv_file} to {destination}")

    logger.info(f"Dataset downloaded to {cfg.data.raw_data_path}")


if __name__ == "__main__":
    main()