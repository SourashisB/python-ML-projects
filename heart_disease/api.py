from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model, scaler, feature list, and default values
model = joblib.load('logreg_model.joblib')
scaler = joblib.load('scaler.joblib')
model_features = joblib.load('model_features.joblib')
defaults = joblib.load('defaults.joblib')  # new: see notes below

# Define the expected input with defaults as None (optional)
class PatientData(BaseModel):
    Age: Optional[int] = None
    Sex: Optional[str] = None
    ChestPainType: Optional[str] = None
    RestingBP: Optional[int] = None
    Cholesterol: Optional[int] = None
    FastingBS: Optional[int] = None
    RestingECG: Optional[str] = None
    MaxHR: Optional[int] = None
    ExerciseAngina: Optional[str] = None
    Oldpeak: Optional[float] = None
    ST_Slope: Optional[str] = None

@app.get("/")
def root():
    return {"message": "Heart Disease Logistic Regression API (handles missing values)"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to dataframe
    input_dict = data.dict()
    # Fill missing entries with defaults
    for k, v in defaults.items():
        if input_dict.get(k) is None:
            input_dict[k] = v
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