# CLAUDE.md

Guidance for Claude Code (or any collaborator) working in this repo.

## What this project is

A brain MRI tumor classifier (glioma / meningioma / notumor / pituitary) built as a portfolio piece: a Keras CNN trained in a notebook, served behind a FastAPI backend with Grad-CAM explainability, with a minimal static frontend, deployed on Google Cloud Run. Full requirements and the reasoning behind every architectural choice live in `PRD.md` — read that first for the "why," including the note on why the deploy target moved off Hugging Face Spaces. This file is the "how to work in this repo" reference.

The training notebook (`mri_classifier.ipynb`) already exists and is validated (93.7% test accuracy). Everything else — API, frontend, deployment — is being built from scratch.

## Repo structure

This is the target layout. Not everything exists yet — build it out per `PRD.md`'s phases.

```
MRI-tumor-Detection-app/
├── .github/
│   └── workflows/
│       └── deploy-cloud-run.yml   # builds the image, pushes to Artifact Registry, deploys to Cloud Run on every push to main
├── .gitignore
├── LICENSE                        # MIT (code license — separate from the CC BY 4.0 dataset license, see below)
├── README.md
├── PRD.md
├── CLAUDE.md
├── Dockerfile                     # builds the deployed image; downloads the pinned model from HF Hub at build time
├── requirements-train.txt         # full deps for local retraining (tensorflow, opencv-python, pandas, matplotlib, scikit-learn, pillow, kaggle)
├── requirements-api.txt           # lean deps for the deployed container (tensorflow-cpu, fastapi, uvicorn, python-multipart, numpy)
├── mri_classifier.ipynb           # training notebook — source of truth for architecture/preprocessing
├── scripts/
│   └── download_model.py          # build-time script: pulls the pinned HF Hub model revision to app/model/
├── app/
│   ├── main.py                    # FastAPI app: /predict, /health, static mount — thin, orchestration only
│   ├── config.py                  # IMG_SIZE, CLASS_NAMES (order matters), HF repo id + pinned revision, model path
│   ├── preprocessing.py           # deep module: raw upload bytes -> normalized (1,256,256,3) array
│   ├── inference.py                # deep module: array -> predicted class + per-class confidence
│   ├── gradcam.py                 # deep module: model + array -> heatmap overlay image
│   └── model/                     # gitignored — populated by scripts/download_model.py during Docker build
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/
│   ├── conftest.py                 # fixtures: TestClient, loaded sample images
│   ├── fixtures/
│   │   └── sample_images/          # a handful of small known-label images, committed to git for test use — NOT the full dataset
│   ├── test_preprocessing.py
│   └── test_predict_endpoint.py
└── data/                           # gitignored — local Kaggle dataset download, training only, never committed
```

**Why `app/` is split this way:** `preprocessing.py`, `inference.py`, and `gradcam.py` are deep modules — simple interfaces (bytes in/array out; array in/prediction out; array in/heatmap out), real complexity hidden inside, and each is independently testable without spinning up FastAPI. `main.py` should stay thin: it wires these together and handles HTTP concerns (validation, status codes), nothing else. If a change touches model math and also touches a route, that's a signal the boundary is in the wrong place.

## Setup & commands

Local training environment (retraining the model):
```
python -m venv venv-train && source venv-train/bin/activate
pip install -r requirements-train.txt
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
jupyter notebook mri_classifier.ipynb
```
Requires a Kaggle API token (`~/.kaggle/kaggle.json`) — see Kaggle account settings.

Local API development:
```
python -m venv venv-api && source venv-api/bin/activate
pip install -r requirements-api.txt
python scripts/download_model.py   # pulls the pinned model revision to app/model/ (needs the HF Hub repo id set in app/config.py)
uvicorn app.main:app --reload
```

Tests:
```
pytest tests/
```

Docker build (mirrors what Cloud Run runs):
```
docker build -t mri-tumor-app .
docker run -p 8000:8000 mri-tumor-app
```

## Architecture decisions that must not get silently violated

These came out of an explicit design review (see `PRD.md`) and are easy to accidentally regress:

- **Preprocessing must use TensorFlow's own ops** (`tf.io.decode_image`, `tf.image.resize(..., method='bilinear')`, `/255.0`), not PIL. The model was trained on images resized via `tf.keras.utils.image_dataset_from_directory`'s internal TF resize — a PIL-based reimplementation uses a different interpolation algorithm and introduces train/serve skew. If you ever touch `app/preprocessing.py`, keep it TF-native.
- **The model file is never committed to git.** It's hosted on a public Hugging Face Hub model repo and pulled into `app/model/` at **Docker build time**, pinned to a specific commit revision (set in `app/config.py`). Don't add code that downloads it at runtime/startup instead — that adds a network dependency and latency to every cold start. Don't remove the revision pin — an unpinned `main` reference means unrelated code pushes can silently change the live model.
- **GitHub is the source of truth**, not any state on Cloud Run. Deploys happen via a GitHub Actions workflow that builds the image, pushes it to Artifact Registry, and deploys a new Cloud Run revision on push to `main` — don't manually `gcloud run deploy` a local build out-of-band, or the two will drift.
- **Grad-CAM finds its target layer programmatically** (`isinstance(layer, tf.keras.layers.Conv2D)`, last match), not by a hardcoded layer name/index. The model was built with a loop that auto-names layers (`conv2d`, `conv2d_1`, ...) — a hardcoded name breaks silently if the architecture is ever retrained with different args.
- **Class label order is fixed:** `glioma, meningioma, notumor, pituitary` (alphabetical directory order, matching how `image_dataset_from_directory` assigned indices during training). This must stay in sync between `app/config.py` and whatever produced the model — if the model is ever retrained from different source folders, re-verify this order from the training run's output, don't assume it.
- **Minimal input validation lives in the `/predict` route itself**, not bolted on later: reject non-image content types and catch decode failures with a clean `400`, so the endpoint never 500s on a bad upload.
- **The on-page disclaimer is part of the frontend, not just the README.** It must be visible near both the upload area and the results — not a one-time dismissible modal.

## Testing conventions

- Test observable behavior (input → output), not internal implementation details.
- `tests/test_preprocessing.py`: given a known fixture image, assert output shape/dtype/value range.
- `tests/test_predict_endpoint.py`: via FastAPI's `TestClient`, hit `/predict` with 2–3 known-label fixture images (one per represented class is plenty), assert the response shape and that the predicted class matches the known label.
- Grad-CAM is intentionally not covered by automated tests — heatmap correctness isn't meaningfully assertable with a lightweight test. Don't add a Grad-CAM test suite without checking with the user first (deliberately kept out of scope).
- Fixture images for tests live in `tests/fixtures/sample_images/` and are committed to git (a handful of small files) — this is distinct from `data/`, which holds the full Kaggle dataset and is gitignored.

## Licensing note

Two separate licenses are in play — don't conflate them:
- **Code**: MIT (see `LICENSE`).
- **Training dataset**: CC BY 4.0 (Kaggle's `masoudnickparvar/brain-tumor-mri-dataset`, itself a combination of the Figshare, SARTAJ, and Br35H datasets). This requires attribution — credit the dataset creator and link the Kaggle page in both `README.md` and the Hugging Face Hub model card. Don't drop this if the README gets rewritten later.

## Deployment target

**Google Cloud Run**, not Hugging Face Spaces or Render. Originally scoped for HF Spaces (Docker SDK) over Render, since Render's free tier's 512MB RAM cap is a real OOM risk for TensorFlow plus this model. Switched to Cloud Run mid-implementation when HF started requiring a PRO subscription to create a Docker SDK Space on a personal account (see `PRD.md`'s "Deployment target note" for the full story). Cloud Run runs the same Dockerfile unmodified, has a perpetual free tier with memory configurable well above 512MB (this app runs with `--memory 2Gi --cpu 2`), and scales to zero when idle. It does require a GCP project with billing enabled (a card on file), though usage stays within the free tier at this app's traffic level.

Live service: `mri-tumor-detection-app` in GCP project `project-de2dd266-f732-4323-a80`, region `us-central1`.

**Gotchas hit during first deploy, worth knowing if you ever redeploy manually or debug CI:**
- `tensorflow-cpu` has no `linux/arm64` wheels for this Python version — if building locally on Apple Silicon, pass `--platform linux/amd64` to `docker build` (Cloud Build's remote builders are x86_64 already, so this only matters for local builds/pushes).
- This GCP project is new enough that the default Compute Engine service account (`<PROJECT_NUMBER>-compute@developer.gserviceaccount.com`, used by both `gcloud run deploy --source` and Cloud Build) had **zero IAM roles** by default — Google stopped auto-granting it Editor on new projects. It needs `roles/storage.objectViewer` (to read the uploaded build source) and `roles/logging.logWriter` (Cloud Build marks the whole build FAILURE if it can't write its own logs, even if the actual `docker build`/push steps succeeded) at minimum.
- If `gcloud run deploy --source .` keeps failing in ways that are hard to diagnose, the more reliable path is: build locally (or in CI) with `docker build`, `docker push` to Artifact Registry directly, then `gcloud run deploy --image <pushed-image>` — this sidesteps Cloud Build's default service account entirely.
