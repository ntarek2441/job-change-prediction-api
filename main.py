from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import re

app = FastAPI(title="Recruitment Screening API")

model = joblib.load("final_model.pkl")
encoder = joblib.load("encoder.pkl")

with open("metadata.json") as f:
    meta = json.load(f)

categorical_columns = meta["categorical_columns"]
numerical_columns = meta["numerical_columns"]
feature_type = meta["feature_type"]
feature_order = meta["feature_order"]
threshold = meta["threshold"]


def sanitize_columns(cols):
    clean = []
    for c in cols:
        c = c.replace("<", "lt").replace(">", "gt")
        c = re.sub(r"[\[\]{}():,]", "_", c)
        clean.append(c)
    return clean


class Candidate(BaseModel):
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
    return {"status": "API is running", "model": meta["best_model_name"]}


@app.post("/predict")
def predict(candidate: Candidate):
    raw = pd.DataFrame([candidate.dict()])

    cat_encoded = pd.DataFrame(
        encoder.transform(raw[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns),
    )

    if feature_type == "fe":
        exp_num = raw["experience"].replace({"<1": 0, ">20": 21, "Unknown": 0}).astype(float)
        num = raw[numerical_columns].copy()
        num["experience_to_training_ratio"] = exp_num / (raw["training_hours"] + 1)
        num["has_relevant_degree"] = (raw["major_discipline"] == "STEM").astype(int)
        full = pd.concat([cat_encoded, num], axis=1)
    else:
        full = pd.concat([cat_encoded, raw[numerical_columns]], axis=1)

    full.columns = full.columns.astype(str)
    if feature_type == "xgb":
        full.columns = sanitize_columns(full.columns)

    full = full.reindex(columns=feature_order, fill_value=0)


    proba_job_change = float(model.predict_proba(full)[0][1])
    proba_no_job_change = 1 - proba_job_change

    will_change_job = proba_job_change >= threshold

    return {
        "will_change_job": bool(will_change_job),
        "label": "Will Change Job" if will_change_job else "Will Not Change Job",
        "job_change_probability": round(proba_job_change, 4),
        "no_job_change_probability": round(proba_no_job_change, 4),
        "model_used": meta["best_model_name"],
    }


