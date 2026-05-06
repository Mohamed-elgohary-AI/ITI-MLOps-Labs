FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY app/ app/

# Install dependencies into the system python, not a venv
RUN uv pip install --system fastapi uvicorn mlflow dagshub scikit-learn pandas python-multipart xgboost

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]