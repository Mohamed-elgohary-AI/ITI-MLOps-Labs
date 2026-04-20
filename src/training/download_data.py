import os
import kagglehub
import shutil
from dotenv import load_dotenv
RAW_DATA_DIR = os.path.join("data", "raw")


def download_iris_data(logger) -> str:
    logger.info("Downloading Titanic dataset from Kaggle...")
    load_dotenv()

    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN")

    path = kagglehub.competition_download("titanic")

    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        csv_files = files
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    for csv_file in csv_files:
        source_file = os.path.join(path, csv_file)
        destination = os.path.join(RAW_DATA_DIR, csv_file)  # keeps original filename
        shutil.copy(source_file, destination)
        logger.info(f"Copied {csv_file} to {destination}")
    
    shutil.copy(source_file, destination)
    print(destination)

    logger.info(f"Titanic dataset downloaded to {RAW_DATA_DIR}")
    return RAW_DATA_DIR
