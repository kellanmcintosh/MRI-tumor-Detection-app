"""FastAPI app: routing and HTTP concerns only.

Deliberately thin: `/predict` wires `preprocessing` -> `inference` together
and translates their outcomes into HTTP status codes/JSON. All real logic
(image decoding, resizing, model loading, prediction) lives in
`app.preprocessing` and `app.inference` so it stays testable without spinning
up FastAPI. Grad-CAM (`app/gradcam.py`) is intentionally not wired in here
yet — that's a later issue.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.inference import get_model, predict
from app.preprocessing import ImageDecodeError, preprocess_image_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model cache once at process startup, not on the first request.
    await run_in_threadpool(get_model)
    yield


app = FastAPI(title="MRI Tumor Classifier API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile) -> dict:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type!r}. Please upload an image file.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        preprocessed = preprocess_image_bytes(image_bytes)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predicted_class, confidences = await run_in_threadpool(predict, preprocessed)

    return {"predicted_class": predicted_class, "confidences": confidences}
