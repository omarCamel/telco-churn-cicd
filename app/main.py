from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.schemas import CustomerInput

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data.csv"
model = joblib.load(BASE_DIR / "models" / "champion.joblib")
dataset = pd.read_csv(DATASET_PATH).dropna(axis=1, how="all")
app = FastAPI(title="API de diagnóstico de cáncer", version="1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/dataset")
@app.get("/data")
async def dataset_summary():
    cleaned = dataset.dropna(axis=1, how="all").copy()
    preview_df = cleaned.head(5).copy()
    preview_df = preview_df.where(pd.notna(preview_df), None)
    preview = preview_df.to_dict(orient="records")
    return {
        "rows": int(cleaned.shape[0]),
        "columns": int(cleaned.shape[1]),
        "column_names": list(cleaned.columns),
        "preview": preview,
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.post("/predict")
async def predict(customer: CustomerInput):
    frame = pd.DataFrame([customer.model_dump(by_alias=True)])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    risk = "Bajo" if probability < 0.30 else "Medio" if probability < 0.70 else "Alto"
    diagnosis_label = "Maligno" if prediction else "Benigno"
    return {
        "prediction": prediction,
        "diagnosis_probability": probability,
        "risk_level": risk,
        "message": f"Diagnóstico {diagnosis_label}: probabilidad estimada de malignidad {probability:.3f}",
    }
