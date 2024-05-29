import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class DiabetesInput(BaseModel):
    HighBP: int
    HighChol: int
    BMI: int
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int


app = FastAPI()

model = mlflow.xgboost.load_model("./app/model")


@app.post("/single_prediction")
def get_prediction(diabetes_input: DiabetesInput):
    X_dict = diabetes_input.model_dump()
    X = pd.DataFrame(X_dict, index=[0])
    y_pred = model.predict(X)

    return {"prediction": int(y_pred[0])}


@app.post("/single_probability_prediction")
def get_proba_prediction(diabetes_input: DiabetesInput):
    X_dict = diabetes_input.model_dump()
    X = pd.DataFrame(X_dict, index=[0])
    y_pred = model.predict_proba(X)
    print(y_pred)

    return {"probabilityPrediction": float(y_pred[0][1])}
