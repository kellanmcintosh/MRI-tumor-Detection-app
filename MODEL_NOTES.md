# Trained model — Hugging Face Hub reference

For use in `app/config.py` in a later slice.

- **HF Hub repo id**: `KellanMcintosh/mri-tumor-classifier`
- **Pinned commit revision**: `a35027761d87b4b738d4d8d70efba85266e57f53`
- **File in repo**: `tumor_classification_model.keras`
- **Repo URL**: https://huggingface.co/KellanMcintosh/mri-tumor-classifier

## Training results (this run)

- Training set accuracy: 98.68%
- Validation set accuracy: 93.93%
- Test set accuracy: 86.06%

Test accuracy is below the original notebook's 93.7% baseline — the notebook doesn't fix
a random seed for model weight initialization (only the train/validation split uses a
fixed seed), so run-to-run variance is expected. Accepted as-is for now to unblock the
app/infrastructure work; retraining for a better run is expected later (see GitHub issue #1).

## Reproducing

```
python -m venv venv-train && source venv-train/bin/activate
pip install -r requirements-train.txt
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
jupyter notebook mri_classifier.ipynb
```

Requires `~/.kaggle/kaggle.json` (or `~/.kaggle/access_token` via `kaggle auth login`) and
`hf auth login` before running the final upload cell.
