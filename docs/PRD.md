# PRD: MRI Tumor Detection App — Inference API + Web UI + Deployment

## Problem Statement

The user has a trained (but currently unrecoverable) Keras CNN that classifies brain MRI images into four categories — glioma, meningioma, no tumor, pituitary — from a notebook-based training exercise that hit 93.7% test accuracy. Right now this work only exists as a Jupyter notebook on a personal machine: there's no way for anyone (recruiters, other developers, the user themselves) to actually try the model without cloning the repo and running the notebook locally. For a portfolio piece, an unrunnable notebook doesn't demonstrate anything beyond "I can write Python" — it doesn't show the ability to ship a working, deployed application.

## Solution

Turn the notebook into a real, publicly deployed web application: retrain the model (the original artifact no longer exists on disk), stand up a FastAPI backend that serves predictions from it, build a minimal drag-and-drop frontend, and deploy the whole thing to Google Cloud Run behind a GitHub-repo-as-source-of-truth workflow. The trained model itself is hosted separately on the Hugging Face Hub (not committed to git) and pulled into the Docker image at build time, pinned to a specific revision. The deployed app includes a persistent on-page disclaimer that it is an educational project, not a diagnostic tool, and a Grad-CAM heatmap overlay so predictions are visually explainable rather than a bare label.

**Deployment target note (updated after Phase 3 implementation):** the PRD below was originally written assuming Hugging Face Spaces (Docker SDK) as the deploy target. Mid-implementation, HF changed policy to require a PRO subscription to create a Docker/Gradio SDK Space on a personal account, even on free `cpu-basic` hardware. The user opted not to pay for that, so the deploy target was switched to **Google Cloud Run** instead — same Dockerfile, no backend/frontend code changes, just a different host. Google Cloud Run requires a GCP account with billing enabled (a card on file), but usage stays within its perpetual free tier for this app's traffic level. References below to "Hugging Face Spaces" as the deploy target are superseded by this note; the model-hosting-on-HF-Hub decisions elsewhere in this doc are unaffected.

## User Stories

1. As a portfolio visitor, I want to open a live URL and immediately see a working MRI classifier, so that I don't have to clone a repo or run a notebook to evaluate the developer's work.
2. As a portfolio visitor, I want to drag and drop (or select) an MRI image file, so that I can quickly try the classifier without friction.
3. As a portfolio visitor, I want to see a preview of the image I uploaded before submitting, so that I can confirm I picked the right file.
4. As a portfolio visitor, I want to click an "Analyze" button and see a loading state while inference runs, so that I know the app is working and hasn't frozen.
5. As a portfolio visitor, I want to see the predicted tumor class along with confidence scores for all four classes, so that I understand how certain the model is, not just its top guess.
6. As a portfolio visitor, I want to see a Grad-CAM heatmap overlaid on my uploaded image, so that I can see which region of the scan drove the model's prediction.
7. As a portfolio visitor or anyone who stumbles on this app outside its portfolio context, I want to see a clear, persistent disclaimer that this is not a real diagnostic tool, so that I don't mistake an educational demo for medical advice.
8. As a portfolio visitor uploading a non-image file or a corrupted file, I want a clear error message instead of a crash, so that the app feels solid rather than fragile.
9. As the developer, I want the trained model artifact stored outside of git (on the Hugging Face Hub), so that I don't hit GitHub's 100MB file size limit or bloat the repo's git history.
10. As the developer, I want the Docker build to pull the model from a pinned Hugging Face Hub revision, so that redeploying for unrelated code changes never silently swaps in a different model version.
11. As the developer, I want the inference-time image preprocessing to exactly mirror the training-time preprocessing (TensorFlow's own resize/decode ops, not a reimplementation in PIL), so that the deployed model's real-world accuracy matches what was measured during training/testing.
12. As the developer, I want a `/health` endpoint, so that I (or the hosting platform) can verify the service is up.
13. As the developer, I want the training-only dependencies (opencv-python, pandas, matplotlib, scikit-learn) kept out of the deployed Docker image, so that builds stay fast and the image stays small on a free-tier host.
14. As the developer, I want pushing to the GitHub repo's `main` branch to automatically build and redeploy the Cloud Run service, so that I have one source of truth and never manually push images out-of-band.
15. As the developer, I want a small automated test suite covering preprocessing and the `/predict` endpoint, so that I can catch regressions if I touch preprocessing or retrain the model later.
16. As the developer, I want the Grad-CAM implementation to find the last convolutional layer programmatically rather than by a hardcoded layer name, so that it doesn't silently break if I retrain with a different architecture configuration.
17. As the developer, I want the README (and the Hugging Face model card) to credit the dataset's CC BY 4.0 license — the creator and the underlying Figshare/SARTAJ/Br35H source datasets — so that the project complies with the dataset's attribution requirement.
18. As the developer, I want a documented, reproducible way to re-download the training dataset (via the Kaggle API, not a manual browser download), so that the training step is scriptable and the README's "how to reproduce this" section is accurate.
19. As the developer, I want the model saved without optimizer state (`include_optimizer=False`), so that the uploaded artifact is as small as reasonably possible.
20. As the developer, I want training-only and API-only Python dependencies split into separate requirement files, so that the local training environment and the deployed container don't carry each other's unnecessary bulk.

## Implementation Decisions

**Model retraining (Phase 0)**
- The previously trained model artifact and its dataset no longer exist on disk (the original save path and venv referenced in the notebook are both gone) — the model must be retrained from the existing notebook, architecture unchanged.
- Dataset acquisition is via the Kaggle API (`kaggle datasets download`), not a manual browser download, so the step is scriptable and documentable.
- Confirmed preprocessing from the existing notebook: images resized to 256×256, RGB (3-channel), pixel values rescaled by `1/255`. Class label order is alphabetical directory order: `glioma, meningioma, notumor, pituitary` (indices 0–3), matching `SparseCategoricalCrossentropy` training.
- Model architecture is unchanged from the notebook: 3 Conv2D+MaxPooling blocks (64/64/128 filters) → Flatten → Dense(128, relu) → Dropout(0.5) → Dense(4, softmax).
- Model is saved with `include_optimizer=False` to reduce artifact size (no need for optimizer state in an inference-only deployment).
- Training and API dependencies are split into two requirement files: `requirements-train.txt` (full: `tensorflow`, `opencv-python`, `pandas`, `matplotlib`, `scikit-learn`, `pillow`, `kaggle`) and `requirements-api.txt` (lean: `tensorflow-cpu`, `fastapi`, `uvicorn`, `python-multipart`, `numpy`).

**Model hosting**
- The trained `.keras` model artifact is uploaded to a **public** Hugging Face Hub model repo — not committed to git, and not gated behind auth (no sensitive data involved; simplifies the Docker build to zero credentials).
- The Docker build step downloads the model from the Hub at **build time** (via `huggingface_hub.hf_hub_download`), pinned to a specific commit revision — not runtime, and not tracking `main` — so the running container never depends on network access to HF Hub, and unrelated code deploys never silently change the live model version.
- `.gitignore` needs a `*.keras` entry (the existing `.gitignore` covers `.h5`/`.pb`/`.pth`/etc. but not the `.keras` extension the notebook actually uses) so a locally retrained model file is never accidentally committed.

**Backend (Phase 1) — module breakdown**
- **Preprocessing module**: single function, uploaded image bytes in → normalized `(1, 256, 256, 3)` float array out. Implemented using TensorFlow's own image ops (`tf.io.decode_image`, `tf.image.resize` with `method='bilinear'`, `/255.0`) to exactly mirror the training pipeline (`tf.keras.utils.image_dataset_from_directory`'s internal resize behavior) — explicitly not reimplemented in PIL, to avoid interpolation/EXIF-handling mismatches between training and serving.
- **Inference module**: loads the model once at process startup from the local path the Docker build populated; takes a preprocessed array, returns predicted class label plus a confidence score per class.
- **Grad-CAM module**: locates the last `Conv2D` layer in the model **programmatically** (`isinstance` filter over `model.layers`, not a hardcoded layer name/index), computes gradients of the predicted class with respect to that layer's activations, and produces a heatmap overlaid on the original uploaded image.
- **Model artifact fetch** (build-time only, not a runtime module): a small script invoked during the Docker build that pulls the pinned HF Hub revision to local disk.
- **FastAPI route layer** stays thin/orchestration-only: `/predict` (accepts an uploaded image, runs preprocessing → inference → Grad-CAM, returns JSON with predicted class, per-class confidence, and the heatmap overlay), and `/health`.
- Minimal input validation lives in Phase 1, not deferred to polish: reject non-image content types and catch image-decode failures, returning a clean `400` instead of an unhandled `500`. Additional hardening (file size caps, magic-byte checks, friendlier messaging) is deferred to Phase 4 polish.

**Frontend (Phase 2)**
- Plain static files — `index.html`, `style.css`, `script.js` in a `static/` folder — no build tooling, no framework.
- Persistent on-page disclaimer ("educational project — not a diagnostic tool, do not use for real medical decisions") visible near both the upload area and the results — not a dismissible modal, and not deferred to only living in the README.
- Drag-and-drop/file picker, image preview before submission, an Analyze button, a loading state during inference, and display of predicted class + per-class confidence scores + Grad-CAM heatmap overlay.

**Serving + deployment (Phase 3)**
- FastAPI mounts the `static/` folder via `StaticFiles` for CSS/JS, and serves `index.html` at `/` via a plain route — standard static-site-serving pattern, not inlined HTML strings in Python.
- Deployment target is **Google Cloud Run**. Originally scoped as Hugging Face Spaces (Docker SDK) over Render, because Render's free tier caps RAM at 512MB, a real OOM risk for TensorFlow plus this model. Mid-implementation, HF began requiring a PRO subscription to create a Docker SDK Space on a personal account; rather than pay for that, the target moved to Cloud Run, which runs the same Dockerfile unmodified, has a perpetual free tier with configurable memory well above 512MB, and needs a GCP billing account (card on file) but no recurring charge at this app's traffic level.
- GitHub remains the single source of truth. A GitHub Actions workflow builds the Docker image, pushes it to Artifact Registry, and deploys a new Cloud Run revision on every push to `main` — no manual out-of-band deploys.

**Documentation / licensing (Phase 4)**
- The Kaggle dataset (`masoudnickparvar/brain-tumor-mri-dataset`) is licensed **CC BY 4.0**, and is itself a combination of the Figshare, SARTAJ, and Br35H datasets. The README must credit the creator (Msoud Nickparvar) and link back to the Kaggle dataset page; the same attribution must appear on the Hugging Face Hub model card, since the model is also publicly hosted there.
- README covers the model architecture, dataset provenance/license, and reproduction steps (including the Kaggle API download command).

## Testing Decisions

- Tests should assert observable behavior (input in, output out) rather than internal implementation details — e.g., test that `/predict` returns the correct class for a known-label image, not that it calls a particular internal function.
- **Preprocessing module**: unit test(s) asserting that a known input image produces output of the correct shape, dtype, and value range (0.0–1.0).
- **`/predict` endpoint**: 2–3 integration tests (via FastAPI's `TestClient`) using known-label images pulled from the Kaggle test set (e.g., one glioma image, one notumor image), asserting the response has the correct shape (predicted class + per-class confidence scores present) and that the predicted class matches the known label.
- Grad-CAM is explicitly **not** covered by automated tests in this pass — correctness of a heatmap isn't meaningfully assertable with a lightweight test, and the user chose to keep test scope minimal (preprocessing + `/predict` only) rather than add a Grad-CAM smoke test.
- No prior test suite exists in this repo to follow as prior art (this is a from-scratch project); the lightweight-pytest pattern here (a handful of targeted tests using real fixture images, not mocks) is the baseline to extend if the app grows.

## Out of Scope

- Improving on the existing 93.7% test accuracy or changing the model architecture/hyperparameters — this PRD retrains the existing notebook as-is, it does not re-engineer the model.
- Any real diagnostic use, clinical validation, or regulatory considerations (explicitly disclaimed on-page and in the README).
- User accounts, authentication, rate limiting, or abuse prevention on the deployed endpoint.
- A CI pipeline beyond the GitHub Actions → Cloud Run deploy workflow (e.g., no automated linting/type-checking pipeline is specified here).
- Automated retraining or a model-versioning/experiment-tracking pipeline — this is a one-shot retrain, not a continuous training system.
- File-size limits, magic-byte validation, and other non-minimal input hardening beyond what's needed to avoid unhandled crashes (deferred to Phase 4 polish, and even then not exhaustively specified).
- Mobile-specific responsive design polish beyond basic usability (not explicitly discussed).
- Automated frontend (JS) tests — the lightweight pytest scope covers backend Python modules only.

## Further Notes

- The original model artifact and training venv are unrecoverable; this project starts from a clean retrain. The dataset itself must also be re-downloaded (not present locally).
- The `.gitignore` at the parent `Port_Projects` directory level currently covers common ML file extensions (`.h5`, `.pb`, `.pth`, `.pt`, `.onnx`) but not `.keras`, which is the format this notebook actually uses — this needs a follow-up entry before any local retraining happens, to avoid accidentally committing a large model file.
- Resolved at Phase 3 implementation time: HF Spaces now requires a PRO subscription for Docker SDK Spaces on personal accounts (a mid-2026 policy change), so the deploy target moved to Google Cloud Run instead. See the "Deployment target note" under Solution above.
