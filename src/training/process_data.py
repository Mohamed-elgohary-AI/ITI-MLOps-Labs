import os
from typing import Any, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from omegaconf import DictConfig

SOURCE = os.path.join("data", "raw")
DESTINATION = os.path.join("data", "processed")


def read_process_data(
    cfg: DictConfig,
    logger,
) -> None:
    logger.info("Data Processing started")
    df = pd.read_csv(os.path.join(SOURCE, f"{cfg.data.file_name}.csv"))
    df.set_index(cfg.data.id_col, inplace=True)
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df[cfg.data.target_col]
    )
    train_df.to_parquet(os.path.join(DESTINATION, f"{cfg.data.file_name}-train.parquet"), engine="pyarrow")
    test_df.to_parquet(os.path.join(DESTINATION, f"{cfg.data.file_name}-test.parquet"), engine="pyarrow")


def preprocess_data(
    X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, logger
) -> Tuple[Any, Any]:
    logger.info("Data Preprocessing Pipeline Started")
    num_cols = X.select_dtypes(include=["int", "float"]).columns
    cat_cols = X.select_dtypes(include="object").columns

    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="mean"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed
