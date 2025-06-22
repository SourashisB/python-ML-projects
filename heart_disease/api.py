from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model, scaler, and feature list
model = joblib.load('logreg_model.joblib')
scaler = joblib.load('scaler.joblib')
model_features = joblib.load('model_features.joblib')

# Define the expected input (edit typing as needed)
class PatientData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

@app.get("/")
def root():
    return {"message": "Heart Disease Logistic Regression API"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to dataframe
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    # One-hot encode, align columns
    input_encoded = pd.get_dummies(input_df)
    # Ensure all columns present as in training
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

    # Normalize continuous features
    features_to_normalize = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_encoded[features_to_normalize] = scaler.transform(input_encoded[features_to_normalize])

    # Predict
    pred = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    return {
        "prediction": int(pred),
        "probability_heart_disease": float(proba)
    }

# To run locally: uvicorn api:app --reload