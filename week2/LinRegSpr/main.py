"""
Hooke's Law Web App — FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import model as ml

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hooke's Law AI",
    description="TensorFlow linear regression demo for Hooke's Law (F = kx)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR    = Path(__file__).parent
STATIC_DIR  = BASE_DIR / "static"
OUTPUT_DIR  = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    epochs:      int   = Field(default=500, ge=50, le=5000)
    new_mass_kg: float = Field(default=15.0, ge=0.1, le=500.0)


class PredictRequest(BaseModel):
    mass_kg: float = Field(..., ge=0.1, le=500.0)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/train")
async def train(req: TrainRequest):
    result = ml.train_model(epochs=req.epochs, new_mass_kg=req.new_mass_kg)
    return {
        "success": True,
        "slope":            round(result.slope, 4),
        "intercept":        round(result.intercept, 4),
        "final_loss":       round(result.final_loss, 6),
        "epochs":           result.epochs,
        "loss_history":     [round(v, 6) for v in result.loss_history[::5]],  # every 5th
        "predicted_length": round(result.predicted_length, 4) if result.predicted_length else None,
        "model_equation":   f"Length = {result.slope:.2f} × Mass + {result.intercept:.2f}",
        "images": {
            "loss_curve":    "/images/loss_curve.png",
            "spring_fitting": "/images/spring_fitting.png",
        },
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    if not ml.is_trained():
        raise HTTPException(status_code=400, detail="Model not trained yet. Call POST /train first.")
    params = ml.get_model_params()
    predicted = ml.predict(req.mass_kg)
    theoretical = 2.0 * req.mass_kg + 10.0
    return {
        "mass_kg":                round(req.mass_kg, 2),
        "predicted_length_cm":    round(predicted, 4),
        "theoretical_length_cm":  round(theoretical, 4),
        "model_equation":         f"Length = {params['slope']:.2f} × Mass + {params['intercept']:.2f}",
        "error_cm":               round(abs(predicted - theoretical), 4),
    }


@app.get("/status")
async def status():
    if not ml.is_trained():
        return {"is_trained": False}
    params = ml.get_model_params()
    result = ml.get_last_result()
    return {
        "is_trained":     True,
        "slope":          round(params["slope"], 4),
        "intercept":      round(params["intercept"], 4),
        "final_loss":     round(result.final_loss, 6) if result else None,
        "model_equation": f"Length = {params['slope']:.2f} × Mass + {params['intercept']:.2f}",
    }


@app.get("/images/{name}")
async def get_image(name: str):
    if not name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG files supported.")
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found. Train the model first.")
    return FileResponse(str(path), media_type="image/png")
