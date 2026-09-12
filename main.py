from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json

app = FastAPI(title="Job Change Prediction API")

model = joblib.load("knn_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

with open("columns.json") as f:
    cols = json.load(f)
categorical_columns = cols["categorical_columns"]
numerical_columns = cols["numerical_columns"]


class Applicant(BaseModel):
    city: str
    gender: str
    relevent_experience: str
    enrolled_university: str
    education_level: str
    major_discipline: str
    experience: str
    company_size: str
    company_type: str
    last_new_job: str
    city_development_index: float
    training_hours: float


@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/predict")
def predict(applicant: Applicant):
    data = pd.DataFrame([applicant.dict()])

    cat_encoded = pd.DataFrame(
        encoder.transform(data[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns),
    )
    full = pd.concat([cat_encoded, data[numerical_columns]], axis=1)
    full.columns = full.columns.astype(str)
    scaled = scaler.transform(full)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "prediction": int(prediction),
        "label": "Looking for a job change" if prediction == 1 else "Not looking for a job change",
        "probability": float(probability),
    }
