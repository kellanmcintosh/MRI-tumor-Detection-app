# MRI Tumor Classifier

**Live demo:** https://mri-tumor-detection-app-530180677158.us-central1.run.app

A web app that classifies a brain MRI image as glioma, meningioma, pituitary tumor, or no tumor. A Keras CNN runs behind a FastAPI backend that returns per-class confidence scores and a Grad-CAM heatmap, with a static frontend, deployed as a single container on Google Cloud Run.

> Educational project, not a diagnostic tool. Do not use for real medical decisions.

## Run locally

```
python -m venv venv-api && source venv-api/bin/activate
pip install -r requirements-api.txt
python scripts/download_model.py
uvicorn app.main:app --reload
```

Tests: `pytest tests/`

## Dataset and license

The model was trained on the [brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Msoud Nickparvar, licensed CC BY 4.0. It combines the Figshare, SARTAJ, and Br35H datasets.

The code in this repository is MIT licensed (see `LICENSE`).
