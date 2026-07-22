"""FastAPI app: routing and HTTP concerns only.

Deliberately thin: `/predict` wires `preprocessing` -> `inference` ->
`gradcam` together and translates their outcomes into HTTP status
codes/JSON. All real logic (image decoding, resizing, model loading,
prediction, heatmap generation) lives in `app.preprocessing`,
`app.inference`, and `app.gradcam` so each stays testable without spinning
up FastAPI. Static file serving (the `static/` frontend) is just a mount --
no logic lives here either.
"""

import base64
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.gradcam import generate_gradcam_overlay
from app.inference import get_model, predict
from app.preprocessing import ImageDecodeError, preprocess_image_bytes

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model cache once at process startup, not on the first request.
    await run_in_threadpool(get_model)
    yield


app = FastAPI(title="MRI Tumor Classifier API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    # StaticFiles sends ETag/Last-Modified but no Cache-Control, so browsers
    # fall back to heuristic caching and can serve a stale HTML/CSS/JS build
    # after a redeploy with no revalidation at all. Force revalidation on
    # every load instead -- cheap (a 304 if unchanged) and guarantees a
    # redeploy is picked up on next visit without a manual hard refresh.
    response = await call_next(request)
    if request.url.path in ("/", "/about") or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/about")
def about() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "about.html"))


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

    model = get_model()
    heatmap_png_bytes = await run_in_threadpool(
        generate_gradcam_overlay, model, preprocessed, image_bytes
    )
    gradcam_overlay = base64.b64encode(heatmap_png_bytes).decode("ascii")

    return {
        "predicted_class": predicted_class,
        "confidences": confidences,
        "gradcam_overlay": gradcam_overlay,
    }
