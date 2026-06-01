import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from torchvision import transforms

from src.model import CatDogCNN


MODEL_PATH = Path(os.getenv("MODEL_PATH", "deployment/cat_dog_cnn.pth"))
CLASS_NAMES = ["cat", "dog"]

REQUEST_COUNT = Counter(
    "catdog_api_requests_total",
    "Total number of HTTP requests received by the model API.",
    ["endpoint", "method", "status"],
)
PREDICTION_COUNT = Counter(
    "catdog_predictions_total",
    "Total number of predictions made by the model.",
    ["predicted_class"],
)
PREDICTION_ERRORS = Counter(
    "catdog_prediction_errors_total",
    "Total number of failed prediction requests.",
    ["error_type"],
)
INFERENCE_LATENCY = Histogram(
    "catdog_inference_latency_seconds",
    "Model inference latency in seconds.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
MODEL_LOADED = Gauge(
    "catdog_model_loaded",
    "Whether the model weights were loaded successfully.",
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogCNN().to(device)
model.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


def load_weights() -> None:
    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        return

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    MODEL_LOADED.set(1)


def prepare_image(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type="invalid_image").inc()
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc

    tensor = preprocess(image).unsqueeze(0)
    return tensor.to(device)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    load_weights()
    yield


app = FastAPI(
    title="Cats vs Dogs MLOps API",
    description="FastAPI inference service for the deployed CatDogCNN model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    model_loaded = MODEL_PATH.exists()
    status = "ok" if model_loaded else "model_missing"
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status="200").inc()
    return {
        "status": status,
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "device": str(device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not MODEL_PATH.exists():
        PREDICTION_ERRORS.labels(error_type="model_missing").inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="503").inc()
        raise HTTPException(status_code=503, detail="No deployed model found.")

    image_bytes = await file.read()
    image_tensor = prepare_image(image_bytes)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    latency = time.perf_counter() - start_time

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    INFERENCE_LATENCY.observe(latency)
    PREDICTION_COUNT.labels(predicted_class=predicted_class).inc()
    REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()

    return {
        "prediction": predicted_class,
        "confidence": round(confidence.item(), 6),
        "latency_seconds": round(latency, 6),
    }


@app.get("/metrics")
def metrics() -> Response:
    REQUEST_COUNT.labels(endpoint="/metrics", method="GET", status="200").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
