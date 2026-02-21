"""
FastAPI application for financial news sentiment inference.

Endpoints
---------
GET  /health           – liveness probe
POST /predict          – predict sentiment for a single news text
POST /predict/batch    – predict sentiment for a list of news texts
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")

LABEL_NAMES = {
    0: "Bearish",
    1: "Somewhat-Bearish",
    2: "Neutral",
    3: "Somewhat-Bullish",
    4: "Bullish",
}

# Global model handle loaded at startup
_model: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning("Model file not found at %s. /predict endpoints will return 503.", MODEL_PATH)
    yield
    _model = None


app = FastAPI(
    title="Financial Market Sentiment API",
    description="MLOps inference service for financial news sentiment analysis.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Request / Response schemas                                                   #
# --------------------------------------------------------------------------- #


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Financial news text to classify.")


class PredictResponse(BaseModel):
    label_id: int
    label: str
    confidence: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, description="List of news texts to classify.")


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


# --------------------------------------------------------------------------- #
# Helper                                                                       #
# --------------------------------------------------------------------------- #


def _require_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and place the model first.")


def _predict_texts(texts: list[str]) -> list[PredictResponse]:
    _require_model()
    probas = _model.predict_proba(texts).tolist()
    results = []
    for proba in probas:
        label_id = int(max(range(len(proba)), key=lambda i: proba[i]))
        results.append(
            PredictResponse(
                label_id=label_id,
                label=LABEL_NAMES.get(label_id, str(label_id)),
                confidence=round(max(proba), 4),
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Liveness probe – always returns 200 if the service is running."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for a single news text."""
    return _predict_texts([request.text])[0]


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict sentiment for multiple news texts in one call."""
    predictions = _predict_texts(request.texts)
    return BatchPredictResponse(predictions=predictions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
