# Model accuracy investigation (2026-07-24)

Findings from debugging why the deployed model performs poorly on meningioma, and a
separate, more consequential discovery made while verifying a proposed fix: **this
model's measured accuracy depends materially on whether it's served from CPU or
GPU.** Nothing in this doc has been acted on yet (no promotion, no code changes) —
it's a record of what's been found so a future session doesn't have to re-derive it.

## Background: why `crop_to_content` and `RandomErasing` exist

The retrained model (`app/model/tumor_classification_model.keras`, currently pinned
via `app/config.py`'s `HF_REVISION`) added two things the original 93.7%-baseline
notebook didn't have:

- `crop_to_content` (`app/preprocessing.py`) — crops out the black background/border
  and zeroes any remaining background pixels within the crop, applied identically at
  train time (`mri_classifier.ipynb`'s `crop_data()`) and serve time
  (`preprocess_image_bytes`).
- `RandomErasing` (`app/model_layers.py`) — a training-only augmentation layer that
  randomly zeroes a small rectangular region (`area_frac` 0.02–0.15 of the image) per
  training image.

Both were added because Grad-CAM showed the earlier model relying heavily on the
black background rather than brain tissue — a shortcut-learning risk, since the
source dataset merges three datasets with different backgrounds/cropping per class.

## Bug found: the documented accuracy numbers were stale

`models/tumor_classification_model_metrics.json` claims test accuracy 92.44% (train
99.53%, val 96.70%) with meningioma recall 92.5%. **This describes a different model
than the one actually deployed.**

- `scripts/compute_confusion_matrix.py` hardcodes
  `MODEL_PATH = models/tumor_classification_model.keras` — an *older* local training
  run (SHA-256 `2255b19a...`), not `app.config.MODEL_PATH` (the HF-Hub-downloaded,
  pinned-revision file the API actually serves).
- The deployed model (`app/model/tumor_classification_model.keras`) is byte-identical
  (SHA-256 `92e9e768...`) to `models/tumor_classification_model_v2.keras` and
  `models/best_checkpoint_v2_resumed2.keras` — a later training run that the metrics
  JSON was never regenerated against.
- The old (metrics.json) model doesn't even have `RandomErasing` in its architecture,
  confirming it predates the retrain that's actually live.

**Not yet fixed:** `compute_confusion_matrix.py` still points at the wrong file, so
running it again would keep reproducing stale numbers. It should be changed to load
via `app.config.MODEL_PATH`.

## Real accuracy of the deployed model, measured directly (CPU)

Recomputed from scratch against the full 1600-image Kaggle test set, using the real
serving path (`app.preprocessing.preprocess_image_bytes` → `model.predict`), on
`tensorflow-cpu` (the backend `requirements-api.txt` / production actually uses):

- **Train: 94.96%** (4254/4480, same reproducible 80/20 split as the notebook:
  `image_dataset_from_directory(..., validation_split=0.2, seed=123)`)
- **Validation: 92.23%** (1033/1120)
- **Test: 88.06%** (1409/1600)

Confusion matrix (rows = true, cols = predicted; order glioma / meningioma / notumor
/ pituitary):

```
glioma:     [341,   4,  47,   8]   85.25% recall
meningioma: [ 93, 269,  13,  25]   67.25% recall  <- the problem
notumor:    [  0,   0, 400,   0]   100%
pituitary:  [  1,   0,   0, 399]   99.75%
```

Also checked every other checkpoint saved during training
(`best_checkpoint_v2.keras` 78.69%, `best_checkpoint_v2_resumed.keras` 78.44%,
`best_checkpoint_v2_resumed2.keras` / deployed 88.06%) — **the deployed model is
already the best checkpoint from its own training lineage.** Nothing better is
sitting unused.

## Root cause of the meningioma recall collapse

Two hypotheses investigated (read-only: Grad-CAM inspection + per-image analysis of
the 93 meningioma→glioma misclassifications, no retraining):

1. **Crop clipping tumor tissue — ruled out.** Crop bounding-box size and
   zeroed-pixel fraction are statistically indistinguishable between correctly- and
   incorrectly-classified meningioma images (retained-area mean 0.761 vs 0.799;
   zeroed fraction 0.274 vs 0.244).

2. **Location-cue confusion, amplified by `RandomErasing` — supported.**
   Grad-CAM attention centroid on correctly-classified meningioma images sits more
   peripheral/skull-adjacent (distance-from-center 0.337) than the glioma-class
   heatmap on misclassified ones (0.242, more central) — matching the architecture's
   documented weak point: `GlobalAveragePooling2D`/`GlobalMaxPooling2D` can't encode
   *where* a signal is, and meningiomas sit at the brain's periphery while gliomas
   sit deeper (comment already in `mri_classifier.ipynb`'s `build_model()`, predating
   this investigation). A meaningful share of meningioma→glioma errors are
   *confidently* wrong (92–100%), not close ties — consistent with `RandomErasing`
   fully wiping a thin peripheral tumor crescent during training more often than it
   wipes a large central glioma mass (a fixed-size random erasure box has
   proportionally higher odds of fully covering a small/thin object).

Ranked fix candidates identified (before the CPU/GPU issue below undercut the first
experiment): reduce `RandomErasing`'s erasure area or apply-probability; per-class
loss weighting upweighting meningioma/glioma; a larger architectural change adding a
non-pooled spatial feature (higher effort, not attempted).

## Fine-tune experiment: reduced RandomErasing + class weighting

Continued training from `models/best_checkpoint_v2_resumed2.keras` (not a full
retrain) for 6 epochs, early-stopped:

- `RandomErasing.area_frac` reduced from `(0.02, 0.15)` to `(0.01, 0.06)` (direct
  attribute mutation on the loaded layer — no architecture change needed).
- `class_weight={0: 1.3, 1: 1.5, 2: 1.0, 3: 1.0}` (glioma/meningioma upweighted),
  recompiled with `Adam(lr=2e-4)`.
- Saved to `models/experiment_finetune_v1.keras` — **deployed files untouched.**

**As first evaluated (on GPU, Metal, inside the training environment):** looked like
a strong win — 93.00% overall, meningioma recall 67.25% → 95.00%.

**Re-verified on CPU** (the backend that actually matters — see below): **86.94%
overall, meningioma recall 64.50%** — essentially no real improvement, and slightly
worse overall than the currently-deployed model. The apparent fix was an artifact of
which backend evaluated it, not a real change in the underlying weights' behavior
under CPU inference.

**Isolated ablations (RandomErasing-only vs class-weight-only, evaluated correctly on
CPU) have not yet been tried** — the combined experiment conflated both fixes with
the backend artifact, so it's still an open, cheap (~a few minutes) next step before
concluding neither fix helps at all.

## The CPU vs GPU divergence (the bigger finding)

While verifying the fine-tune experiment, found that the *original, unmodified,
currently-deployed model* itself produces materially different results depending on
inference backend — same weights, same input, same preprocessing:

| | CPU (`tensorflow-cpu`, what production serves) | GPU (Metal, Apple Silicon, `tensorflow-metal`) |
|---|---|---|
| Overall | 88.06% | **92.44%** |
| Glioma recall | 85.25% | 76.75% (worse) |
| Meningioma recall | **67.25%** | **93.5%** (much better) |
| Notumor / Pituitary | ~100% both | ~100% both |

**154 of 1600 test images (9.6%) flip predicted class purely based on which backend
runs inference.** Confirmed this is a real, deterministic divergence, not noise or a
bug in this codebase's own code:

- Preprocessing output differs between backends by ≤5.8e-6 (float32 noise, not the
  cause).
- `RandomErasing` is not accidentally active during GPU inference — predictions were
  bit-identical across 5 repeated GPU runs (if the random-erasure layer were firing,
  each run would differ).
- The divergence happens somewhere inside the model's own forward pass — most likely
  in how Metal's TensorFlow plugin implements Conv2D and/or the
  GlobalAveragePooling2D/GlobalMaxPooling2D branch differently from the CPU kernel.
  Not root-caused down to the specific op — flagged as a worthwhile follow-up if
  accuracy work continues.

**Why this matters more than any single training fix:** production deploys via
`tensorflow-cpu` on Google Cloud Run (`requirements-api.txt`, no GPU available on
Cloud Run at all), so **production is permanently stuck serving the worse numbers**
(88.06% overall, 67.25% meningioma recall) regardless of what a GPU-equipped dev
machine shows. This also isn't simply fixable by "switching to GPU serving" —
`tensorflow-metal` is Apple-Silicon-only; a hypothetical CUDA GPU on a different
Cloud Run configuration would be a third, distinct numerical implementation with no
guarantee of reproducing either number measured here.

## Current status / open follow-ups

- No code or deployed files changed as a result of this investigation.
  `models/experiment_finetune_v1.keras` exists as an experimental artifact only —
  not recommended for promotion given the CPU-verified numbers above.
- `scripts/compute_confusion_matrix.py` still points at the wrong (stale) model file
  — needs fixing to use `app.config.MODEL_PATH` so future runs don't reproduce this
  same confusion.
- Untried: isolated RandomErasing-only and class-weight-only fine-tunes, evaluated
  on CPU.
- Untried: root-causing exactly which op diverges between CPU and Metal GPU kernels.
- Pre-existing, still open, related GitHub issues: #14 (fix `clean_images()`
  validation), #15 (audit train/test split for duplicate/near-duplicate leakage),
  #16 (transfer learning to raise the accuracy ceiling).
