from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.schemas import CustomerInput

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data.csv"
model = joblib.load(BASE_DIR / "models" / "champion.joblib")
dataset = pd.read_csv(DATASET_PATH)
app = FastAPI(title="Telco Churn API", version="1.0")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/dataset")
@app.get("/data")
async def dataset_summary():
    preview_df = dataset.head(5).copy()
    preview_df = preview_df.where(pd.notna(preview_df), None)
    preview = preview_df.to_dict(orient="records")
    return {
        "rows": int(dataset.shape[0]),
        "columns": int(dataset.shape[1]),
        "column_names": list(dataset.columns),
        "preview": preview,
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")

@app.post("/predict")
async def predict(customer: CustomerInput):
    frame = pd.DataFrame([customer.model_dump()])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    risk = "Bajo" if probability < 0.30 else "Medio" if probability < 0.70 else "Alto"
    return {
        "prediction": prediction,
        "churn_probability": probability,
        "risk_level": risk,
        "message": "Cliente con riesgo de abandono" if prediction else "Cliente con bajo riesgo de abandono"
    }
