# CLAUDE.md

Guidance for Claude Code (or any collaborator) working in this repo.

## What this project is

A brain MRI tumor classifier (glioma / meningioma / notumor / pituitary) built as a portfolio piece: a Keras CNN trained in a notebook, served behind a FastAPI backend with Grad-CAM explainability, with a minimal static frontend, deployed on Hugging Face Spaces. Full requirements and the reasoning behind every architectural choice live in `PRD.md` — read that first for the "why." This file is the "how to work in this repo" reference.

The training notebook (`mri_classifier.ipynb`) already exists and is validated (93.7% test accuracy). Everything else — API, frontend, deployment — is being built from scratch.

## Repo structure

This is the target layout. Not everything exists yet — build it out per `PRD.md`'s phases.

```
MRI-tumor-Detection-app/
├── .github/
│   └── workflows/
│       └── sync-to-hf-space.yml   # pushes to the HF Space git remote on every push to main
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

Docker build (mirrors what HF Spaces runs):
```
docker build -t mri-tumor-app .
docker run -p 8000:8000 mri-tumor-app
```

## Architecture decisions that must not get silently violated

These came out of an explicit design review (see `PRD.md`) and are easy to accidentally regress:

- **Preprocessing must use TensorFlow's own ops** (`tf.io.decode_image`, `tf.image.resize(..., method='bilinear')`, `/255.0`), not PIL. The model was trained on images resized via `tf.keras.utils.image_dataset_from_directory`'s internal TF resize — a PIL-based reimplementation uses a different interpolation algorithm and introduces train/serve skew. If you ever touch `app/preprocessing.py`, keep it TF-native.
- **The model file is never committed to git.** It's hosted on a public Hugging Face Hub model repo and pulled into `app/model/` at **Docker build time**, pinned to a specific commit revision (set in `app/config.py`). Don't add code that downloads it at runtime/startup instead — that adds a network dependency and latency to every cold start. Don't remove the revision pin — an unpinned `main` reference means unrelated code pushes can silently change the live model.
- **GitHub is the source of truth**, not the Hugging Face Space's git history. Deploys happen via a GitHub Actions workflow pushing to the Space remote on push to `main` — don't manually push to the Space remote out-of-band, or the two will drift.
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

Hugging Face Spaces (Docker SDK), not Render — chosen because Render's free tier's 512MB RAM cap is a real OOM risk for TensorFlow plus this model, while HF Spaces' free CPU tier gives 16GB RAM. If this ever changes, revisit the Dockerfile's base image and the RAM assumption baked into that decision.
