# Trained model — Hugging Face Hub reference

For use in `app/config.py` in a later slice.

- **HF Hub repo id**: `KellanMcintosh/mri-tumor-classifier`
- **Pinned commit revision**: `058ed5400e81f56dd9045a5f8fe554fadd60d9fc`
- **File in repo**: `tumor_classification_model.keras`
- **Repo URL**: https://huggingface.co/KellanMcintosh/mri-tumor-classifier

## Architecture (this run)

Four Conv2D + MaxPooling blocks (64, 64, 128, 512 filters), with Global Average Pooling
and Global Max Pooling run in parallel off the last block and concatenated into a
Dense(128) layer with 0.3 dropout, ending in a 4-way softmax.

## Training results (this run)

- Training set accuracy: 99.53%
- Validation set accuracy: 96.70%
- Test set accuracy: 92.44%
- Epochs trained: 72

Test accuracy is slightly below the original notebook's 93.7% baseline, closing most of
the gap from the previous retrain (86.06%). The notebook doesn't fix a random seed for
model weight initialization (only the train/validation split uses a fixed seed), so
run-to-run variance is expected. Pinning a seed to close the remaining gap is still
tracked as a follow-up (see GitHub issue #1).

## Reproducing

```
python -m venv venv-train && source venv-train/bin/activate
pip install -r requirements-train.txt
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
jupyter notebook mri_classifier.ipynb
```

Requires `~/.kaggle/kaggle.json` (or `~/.kaggle/access_token` via `kaggle auth login`) and
`hf auth login` before running the final upload cell.
