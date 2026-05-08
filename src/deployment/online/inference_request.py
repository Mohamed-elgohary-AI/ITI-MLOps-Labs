from pydantic import BaseModel

class InferenceRequest(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Ticket: str
    Name: str
    Cabin: str