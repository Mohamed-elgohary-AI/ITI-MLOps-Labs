# setup_pipeline.sh
#!/bin/bash
set -e  # stop if any command fails

echo "Setting up DVC pipeline..."

# Stage 1 — download
uv run dvc stage add --force -n download \
  -o data/raw/train.csv \
  -o data/raw/test.csv \
  -o data/raw/gender_submission.csv \
  -p conf/data/titanic.yaml:dataset_name \
  "uv run python -m src.training.download_data"

# Stage 2 — process
uv run dvc stage add --force -n process \
  -d data/raw/train.csv \
  -o data/processed/train-train.parquet \
  -o data/processed/train-test.parquet \
  -p conf/data/titanic.yaml:file_name \
  -p conf/data/titanic.yaml:id_col \
  -p conf/data/titanic.yaml:target_col \
  -p conf/data/titanic.yaml:test_size \
  -p conf/data/titanic.yaml:random_state \
  "uv run python -m src.training.process_data"

# Stage 3a — train LR
uv run dvc stage add --force -n train_lr \
  -d data/processed/train-train.parquet \
  -d data/processed/train-test.parquet \
  -o models/lr/final_model.pkl \
  -o models/lr/model_target_translator.pkl \
  -p conf/data/titanic.yaml:target_col \
  -p conf/model/lr.yaml:name \
  -p conf/model/lr.yaml:save_dir \
  -p conf/tuning/hyperopt.yaml:max_evals \
  -p conf/tuning/hyperopt.yaml:n_folds \
  -p conf/tuning/hyperopt.yaml:scoring \
  "uv run python -m src.training.train model=lr"

# Stage 3b — train XGB
uv run dvc stage add --force -n train_xgb \
  -d data/processed/train-train.parquet \
  -d data/processed/train-test.parquet \
  -o models/xgb/final_model.pkl \
  -o models/xgb/model_target_translator.pkl \
  -p conf/data/titanic.yaml:target_col \
  -p conf/model/xgb.yaml:name \
  -p conf/model/xgb.yaml:save_dir \
  -p conf/tuning/hyperopt.yaml:max_evals \
  -p conf/tuning/hyperopt.yaml:n_folds \
  -p conf/tuning/hyperopt.yaml:scoring \
  "uv run python -m src.training.train model=xgb"

# Stage 3c — train RF
uv run dvc stage add --force -n train_rf \
  -d data/processed/train-train.parquet \
  -d data/processed/train-test.parquet \
  -o models/rf/final_model.pkl \
  -o models/rf/model_target_translator.pkl \
  -p conf/data/titanic.yaml:target_col \
  -p conf/model/rf.yaml:name \
  -p conf/model/rf.yaml:save_dir \
  -p conf/tuning/hyperopt.yaml:max_evals \
  -p conf/tuning/hyperopt.yaml:n_folds \
  -p conf/tuning/hyperopt.yaml:scoring \
  "uv run python -m src.training.train model=rf"

# Stage 4a — evaluate LR
uv run dvc stage add --force -n evaluate_lr \
  -d models/lr/final_model.pkl \
  -d models/lr/model_target_translator.pkl \
  -d data/processed/train-test.parquet \
  -o reports/lr/evaluation_report.json \
  -p conf/data/titanic.yaml:target_col \
  -p conf/model/lr.yaml:name \
  -p conf/model/lr.yaml:save_dir \
  "uv run python -m src.training.evaluate model=lr"

# Stage 4b — evaluate XGB
uv run dvc stage add --force -n evaluate_xgb \
  -d models/xgb/final_model.pkl \
  -d models/xgb/model_target_translator.pkl \
  -d data/processed/train-test.parquet \
  -o reports/xgb/evaluation_report.json \
  -p conf/data/titanic.yaml:target_col \
  -p conf/model/xgb.yaml:name \
  -p conf/model/xgb.yaml:save_dir \
  "uv run python -m src.training.evaluate model=xgb"

# Stage 4c — evaluate RF
uv run dvc stage add --force -n evaluate_rf \
  -d models/rf/final_model.pkl \
  -d models/rf/model_target_translator.pkl \
  -d data/processed/train-test.parquet \
  -o reports/rf/evaluation_report.json \
  -p conf/data/titanic.yaml:target_col \
  -p conf/model/rf.yaml:name \
  -p conf/model/rf.yaml:save_dir \
  "uv run python -m src.training.evaluate model=rf"

echo "Pipeline setup complete. Run 'uv run dvc repro' to execute."