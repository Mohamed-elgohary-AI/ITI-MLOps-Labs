import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_production_model

app = FastAPI(title="Titanic Survival Predictor")
model = load_production_model()


class PassengerFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


class PredictionResponse(BaseModel):
    survived: int
    probability: float


def preprocess(passenger: dict) -> pd.DataFrame:
    df = pd.DataFrame([passenger])
    df = pd.get_dummies(df, columns=["Sex", "Embarked"])

    # Ensure all expected columns are present
    expected_cols = [
        "Pclass", "Age", "SibSp", "Parch", "Fare",
        "Sex_female", "Sex_male",
        "Embarked_C", "Embarked_Q", "Embarked_S"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    return df[expected_cols]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerFeatures):
    df = preprocess(passenger.model_dump())
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return PredictionResponse(
        survived=int(prediction),
        probability=round(float(probability), 4)
    )