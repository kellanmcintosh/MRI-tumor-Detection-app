# Model accuracy investigation (2026-07-24)

Findings from debugging why the deployed model performs poorly on meningioma, a
separate discovery made while verifying a proposed fix (this model's measured
accuracy depends materially on whether it's served from CPU or GPU, root-caused
below to a `tensorflow-metal` graph-mode bug), and finally the actual production
incident that prompted the most urgent part of this work — see "RESOLVED: production
was serving the wrong model file" below, which is the fix that was actually shipped.

## RESOLVED: production was serving the wrong model file (2026-07-24)

**This was the real cause of production giving wrong predictions on essentially
every request** (a user report of "every single prediction wrong," far worse than
anything the CPU/GPU divergence below could explain) — not a numerics or
architecture issue at all, and not related to the CPU/GPU divergence section below.

**Root cause:** `app/config.py`'s `HF_REVISION` was pinned to
`058ed5400e81f56dd9045a5f8fe554fadd60d9fc` on the `KellanMcintosh/mri-tumor-classifier`
Hugging Face Hub repo. That commit's message reads "Retrain: train 99.53% / val
96.70% / test 92.44%..." (the retrained model's real numbers) — but the file it
actually contains, verified by SHA-256, is `2255b19a...`, the **pre-retrain model**
(same file as local `models/tumor_classification_model.keras`), not the retrained
`models/tumor_classification_model_v2.keras` (`92e9e768...`) the commit message
claims. A second commit one second later (`28cd87a0`, same message) has the same
wrong file. The original upload script pushed the wrong local path to the Hub —
twice — while labeling both commits with the new model's metrics. This was never
caught because local testing/investigation (including earlier in this doc) checked
the hash of `app/model/tumor_classification_model.keras` on a dev machine where that
file had, at some point, been manually placed/replaced with the correct model for
local testing — masking the fact that a fresh `scripts/download_model.py` run (what
Docker build time, and therefore Cloud Run, actually does) pulls the wrong file.
The pre-retrain model was also never trained with `crop_to_content` in its input
pipeline (added later, see "Background" below), so feeding it the current
(cropped) preprocessing made results considerably worse than even the pre-retrain
model's own original numbers would suggest.

**How this was found:** re-investigating a user report of catastrophic live
production failure, initially (incorrectly) suspected as an x86-vs-ARM CPU
numerics issue, since `docker build --platform linux/amd64` (matching Cloud Run
exactly) reproduced the bad predictions while every manually-substituted-correct-
model test on the same architecture scored normally. Bisecting *why* those two
things differed (rather than assuming it was the architecture) led to comparing
file hashes between what `scripts/download_model.py` actually fetches (fresh, via
the real pinned revision) versus what happened to already be sitting in
`app/model/` locally — which is when the hash mismatch surfaced. In hindsight the
"x86 collapse" was a coincidence of which code path happened to touch which file,
not a real architecture-specific bug; once the correct model was used, x86 and ARM
CPU gave consistent results (batch size and threading were also ruled out as
factors during this process).

**Fix applied:**
1. Uploaded the correct file (`models/tumor_classification_model_v2.keras`,
   `92e9e768...`) to the Hub as a new commit (`bd4045947697284c629d5f2e5a261609f1bab691`),
   verified by fresh download + hash check.
2. Updated `app/config.py`'s `HF_REVISION` to that commit.
3. Verified end-to-end: rebuilt the actual production Docker image
   (`--platform linux/amd64`, identical to what Cloud Run runs), ran the real
   `app.inference.predict()` path against the full 1600-image Kaggle test set
   inside that container — **88.06% overall accuracy, 85.25%/67.25%/100%/99.75%
   recall (glioma/meningioma/notumor/pituitary)**, exactly matching this doc's
   documented CPU baseline below. Confirms the fix is complete and the earlier
   x86-specific numbers were an artifact of the wrong-file bug, not a real
   architecture effect.

**Not yet done as of this commit:** merging this branch to `main` and letting the
existing GitHub Actions workflow redeploy Cloud Run with the corrected pin (the
Dockerfile/deploy workflow themselves needed no changes — only the revision
string).

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

**Isolated ablations (RandomErasing-only vs class-weight-only) have now been run and
evaluated on CPU — see "Isolated ablations" below. Neither is a real fix.**

## Isolated ablations (2026-07-24)

The combined fine-tune above changed two things at once (`RandomErasing.area_frac`
and `class_weight`), so it couldn't say which change (if either) did anything real.
Ran each in isolation, same setup: continued training from
`models/best_checkpoint_v2_resumed2.keras`, 6 epochs early-stopped on `val_accuracy`,
`Adam(lr=2e-4)`, same train/val split (`image_dataset_from_directory(...,
validation_split=0.2, seed=123)`). Training ran in `venv-train`
(`tensorflow-metal`, for speed); **evaluation was forced onto CPU** by disabling GPU
visibility (`tf.config.set_visible_devices([], "GPU")`) before any other TF op ran,
via `scripts/eval_full_cpu.py`. Cross-checked predictions on 20 test images
(5 per class) against real `tensorflow-cpu` in `venv-api` — bit-identical
probabilities to 4 decimal places, confirming the CPU-forced venv-train numbers are
trustworthy and not another GPU-artifact repeat.

**RandomErasing-only** (`area_frac` (0.02,0.15) → (0.01,0.06), class_weight
untouched) — saved to `models/experiment_finetune_randomerasing_only.keras`:

- Train 95.29%, Val 93.13%, **Test 87.69%** (1403/1600)

```
glioma:     [333,   7,  54,   6]   83.25% recall
meningioma: [ 86, 274,  17,  23]   68.50% recall
notumor:    [  0,   0, 400,   0]   100%
pituitary:  [  4,   0,   0, 396]   99.00%
```

**class-weight-only** (`class_weight={0:1.3,1:1.5,2:1.0,3:1.0}`, RandomErasing
untouched) — saved to `models/experiment_finetune_classweight_only.keras`:

- Train 93.75%, Val 90.54%, **Test 86.00%** (1376/1600)

```
glioma:     [334,   4,  54,   8]   83.50% recall
meningioma: [105, 246,  21,  28]   61.50% recall
notumor:    [  0,   0, 400,   0]   100%
pituitary:  [  4,   0,   0, 396]   99.00%
```

**Verdict, vs. baseline (88.06% overall / 67.25% meningioma recall) and the combined
experiment (86.94% / 64.50%), all CPU:**

- **RandomErasing-only** is roughly a wash: meningioma recall +1.25pp (67.25% →
  68.50%) but overall accuracy −0.37pp (88.06% → 87.69%), driven by glioma recall
  dropping 85.25% → 83.25%. A small, noise-adjacent shuffle between glioma and
  meningioma, not a real fix — but it's the isolated change closest to neutral, and
  the one responsible for whatever small non-harmful signal exists in the combined
  run.
- **class-weight-only actively hurts**: meningioma recall −5.75pp (67.25% →
  61.50%), overall −2.06pp (88.06% → 86.00%). Upweighting meningioma/glioma loss did
  not translate into better meningioma recall on CPU — if anything more meningioma
  images shifted to glioma (93 → 105 meningioma→glioma errors). This is the
  dominant contributor to the combined experiment's CPU regression.
- **Conclusion: neither isolated change is a real fix, and class weighting is the
  worse of the two.** This matches the combined experiment's CPU result (86.94% /
  64.50%, between these two and dragged down mostly by class weighting) and closes
  out the "isolated ablations" open item — no further fine-tuning of this
  architecture along these two axes is worth pursuing. The CPU/GPU divergence
  documented below remains the dominant, unresolved issue.

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
  `models/experiment_finetune_v1.keras`, `models/experiment_finetune_randomerasing_only.keras`,
  and `models/experiment_finetune_classweight_only.keras` exist as experimental
  artifacts only — none recommended for promotion given the CPU-verified numbers
  above.
- **Fixed:** `scripts/compute_confusion_matrix.py` now loads via `app.config.MODEL_PATH`
  (verified on CPU — reproduces 88.06% overall, 67.25% meningioma recall exactly).
- **Done:** isolated RandomErasing-only and class-weight-only fine-tunes, evaluated
  on CPU — see "Isolated ablations" above. Neither is a real fix; class weighting is
  actively harmful. No further work planned along these two axes.
- Untried: root-causing exactly which op diverges between CPU and Metal GPU kernels
  — now the single highest-value remaining lead, since both training-side fixes
  tried so far (combined and isolated) have failed to move CPU numbers.
- Pre-existing, still open, related GitHub issues: #14 (fix `clean_images()`
  validation), #15 (audit train/test split for duplicate/near-duplicate leakage),
  #16 (transfer learning to raise the accuracy ceiling).

## Root-caused (2026-07-24): the CPU/GPU divergence is a tensorflow-metal graph-mode bug that drops a Dense layer's ReLU activation, not a Conv/pooling numerical difference

Follow-up session, `venv-train` (TF 2.18.0, `tensorflow-metal` 1.2.0, Apple M2 Max).
Read-only: no retraining, no changes to `app/`, `models/`, or the deployed model.

**Method:** loaded `app/model/tumor_classification_model.keras` directly, built
sub-models from `model.input` to each layer's output in turn, and ran each
sub-model on identical preprocessed input (real test images, via
`app.preprocessing.preprocess_image_bytes`) under `tf.device('/CPU:0')` and
`tf.device('/GPU:0')`.

**First surprise: layer-wise CPU-vs-GPU bisection found nothing.** Comparing
`sub(x, training=False)` (direct eager call) on CPU vs GPU, at every layer, for
both the 4 fixture images and a real batch of 32 test images (including known
flip cases) — every layer's max abs diff stayed at float32-noise level
(~1e-7 to ~8e-6) all the way to the final softmax output. This flatly
contradicts the aggregate 88.06%/92.44% gap, which meant the divergence isn't
in eager CPU-vs-GPU numerics at all.

**Second step, found the actual mechanism:** the project's own eval scripts
(`compute_confusion_matrix.py`, and the GPU-side numbers quoted earlier in this
doc) call **`model.predict()`**, not a direct eager call. Comparing
`model.predict()` output against a direct eager `model(x, training=False)` call
**on the same device** isolates a completely different axis:

| | CPU: `predict()` vs eager call | GPU: `predict()` vs eager call |
|---|---|---|
| max abs diff (final softmax) | 0.0 (exact) | **0.6957** |

On CPU, `predict()` and eager give identical results. On GPU (Metal),
`model.predict()` diverges hugely from an eager call on the *identical* model,
identical weights, identical input — proving this was never a CPU-vs-GPU
kernel-numerics issue. It's specific to Metal's compiled-graph execution path.

**Bisecting that predict()-vs-eager gap layer by layer (GPU only) pinpoints the
exact culprit:** every layer up through `concatenate` (both Conv2D blocks,
all four MaxPooling2D layers, GlobalAveragePooling2D, GlobalMaxPooling2D) shows
**exactly 0.0** diff between `predict()` and eager call. The divergence appears
abruptly at the first `Dense` layer (`dense`, 1024→128, ReLU activation,
immediately after the GAP/GMP concat):

```
concatenate    max_abs=0.000000e+00
dense          max_abs=3.984632e+01   <- jumps from 0 to ~40
dropout        max_abs=3.984632e+01   (passthrough at inference)
dense_1        max_abs=6.956916e-01   (final softmax, after 4-class squashing)
```

**Confirmed exactly what's wrong:** `model.predict()`'s output at the `dense`
layer, on GPU, contains negative values (e.g. `-0.99, -8.21, -15.10, -37.17, ...`)
even though the layer's activation is ReLU. Manually computing
`relu(concat_output @ kernel + bias)` from the correct (eager) concat output
matches the eager-call `dense` output exactly (0.0 diff) and matches the raw,
*unclipped* linear result (`concat_output @ kernel + bias`, no ReLU at all)
exactly against `predict()`'s GPU output (0.0 diff). **`model.predict()` on the
Metal GPU backend silently skips the ReLU activation on this Dense layer,**
passing the raw pre-activation (including negative values) forward into
`dense_1`'s softmax, corrupting the final class probabilities by up to 0.95 in
absolute probability mass on individual test images — fully explaining the
154/1600 flipped predictions and the 88.06% vs 92.44% gap.

**Confirmed this is graph-mode-general, not a `.predict()`-internals quirk:**
wrapping the same sub-model call in a bare `@tf.function` (no Keras predict
loop involved at all) reproduces the identical 39.85 max-abs diff on GPU.
Deterministic and reproducible across repeated runs (matches the earlier
finding that 5 repeated GPU predict() runs were bit-identical — this is a
consistent bug, not stochastic noise). Confirmed **`model.compile(run_eagerly=True)`
then `.predict()`** matches the eager-call output exactly (0.0 diff) — eager
execution, whether invoked directly or forced via `run_eagerly=True`, always
applies the activation correctly on this backend; only the default
compiled-graph path drops it.

**Not a simple "avoid fused Dense+activation" fix:** rebuilt the same 1024→128
transform as a separate `Dense(activation=None)` layer followed by a standalone
`Activation('relu')` layer (loading the identical weights) and ran it through
`.predict()` on GPU — **same bug reproduces identically** (39.85 max abs diff
vs the correct output). So this isn't specifically about Keras fusing an
activation string into the Dense op's kernel; the ReLU op itself fails to
execute under Metal's compiled-graph path in this model graph, whether it's a
Dense's built-in activation or a separate Activation layer downstream of a
linear Dense. (For contrast, raw `tf.matmul`/`tf.nn.relu` ops outside of any
Keras layer, wrapped in the same kind of `@tf.function` and run on GPU, computed
correctly — so the bug is specific to how the Keras layer/model machinery
executes under Metal's graph mode, not a blanket "ReLU is broken on Metal.")

**Corroborated by a known, unresolved upstream issue:**
[tensorflow/tensorflow#61650](https://github.com/tensorflow/tensorflow/issues/61650),
"Activation function of a Dense hidden layer not getting invoked" (TF 2.13,
macOS 13.4, Apple M2 Max, `tensorflow-macos`+`tensorflow-metal`) — a from-scratch
autoencoder repro where a Dense layer's ReLU never fires. Filed 2023, closed by
the stale-bot after inactivity, never actually fixed. The reporter's own
comment: "When I set eager execution to true the bug does not manifest ... I am
guessing this has something to do with the execution graph optimization" —
matches this investigation's finding exactly. Broader web search also surfaced
reports of the Matmul+BiasAdd+Activation fusion pattern producing 30-80+
magnitude errors under Metal graph mode vs ~1e-5 in eager mode, and a separate
report of ReLU failing to clip negative values under `tensorflow-metal` on
Apple M4, again only in compiled/graph execution, not CPU. This looks like a
long-standing, never-fixed defect in the `tensorflow-metal` pluggable-device
plugin's graph-mode handling of at least this Dense/activation pattern, not
something specific to this project's model.

**Practical takeaways:**

1. **This project's existing discipline is already correct and must not
   regress:** `scripts/eval_full_cpu.py` forces
   `tf.config.set_visible_devices([], "GPU")` before any other TF op runs.
   Every future accuracy/model-selection decision must go through a CPU-only
   evaluation path like that one. This investigation found the *mechanism*
   behind why GPU-side numbers can't be trusted, but the conclusion (CPU is
   the only backend that matters for go/no-go decisions, because it's the only
   backend Cloud Run production ever runs) was already correct and stands
   reinforced, not revised.
2. **If a GPU-accelerated sanity check is ever wanted during development,**
   it must force eager execution (`model.compile(run_eagerly=True)` before
   `.predict()`/`.evaluate()`, or call the model directly as
   `model(x, training=False)` rather than `.predict()`) — the default
   compiled-graph path silently corrupts results on this backend. This is a
   workaround for local dev-loop sanity checks only, not a reason to ever
   promote a GPU-measured metric to a deploy decision.
3. **Training itself was not re-examined here** (out of scope for this
   read-only investigation, and no retraining was done) — `model.fit()`'s
   `train_step` is also `tf.function`-compiled by default, so it's plausible
   the same class of bug affects some part of the training forward/backward
   pass on this Metal backend too. Flagged as an open question, not concluded:
   if training-time GPU behavior is ever suspected of contributing to a model
   quality issue, it would need its own separate investigation (comparing
   training on forced-CPU vs GPU), which was not attempted here.
4. No code, deployed files, or model artifacts were changed. This section is a
   diagnostic record only.

## Independent corroboration (2026-07-24): confirms "RESOLVED: production was serving the wrong model file" above, plus rules out Docker/Rosetta emulation as a contributing factor, and confirms the fix has not yet reached live Cloud Run

This section is an independently-run reproduction of the same incident described in
"RESOLVED: production was serving the wrong model file" above (same root cause, same
fix commit hash) — run without initial knowledge that a fix already existed on branch
`fix/hf-model-revision-pin`, to answer a specific question the project owner raised:
given that the project owner's local repro used `docker build --platform linux/amd64`
on Apple Silicon (Rosetta/QEMU-emulated, not real x86_64), was the ~45% number possibly
just an emulation artifact rather than a real bug? Read-only investigation, no
retraining; no changes made to `app/`, `models/`, or deployed files (the existing fix
commit was found already in place on disk, not applied by this session).

**Reproduced.** Built fresh (`docker build --platform linux/amd64
-t mri-tumor-app-investigation .`), ran it, drove 100 real HTTP `POST /predict`
requests (25/class) at the container's exposed port 8000 (the same route/method
production actually serves): **42.00% accuracy**, glioma recall **0%** (20/25 glioma
images predicted `notumor`), heavy bias toward `notumor`/`meningioma`. Matches the
reported ~45%.

**Ruled out emulation as the cause, several independent ways:**
- Docker Desktop is configured to emulate `linux/amd64` via **Rosetta**, not QEMU
  (confirmed: the Docker Desktop VM process is launched with a `--rosetta` flag, and
  the `oahd` Rosetta translation daemon is running on the host — checked via `ps aux`
  and the VM launch command line; no Rosetta/virtualization keys were present in
  `~/Library/Group Containers/group.com.docker/settings-store.json`, so this is
  Docker Desktop's current default, not a manual override).
- Basic vectorized numeric sanity checks inside the emulated container came back
  clean: a 512×512 `float32` matmul (both raw NumPy and `tf.matmul`) vs. the same
  computation upcast to `float64` gave max abs diffs of ~2.2e-4 and ~1.0e-4 —
  ordinary float32 rounding, not the 30-80+ magnitude corruption signature of AVX
  mis-emulation or the Metal ReLU-skip bug documented above.
- Two direct-Python (non-HTTP) evaluations inside an x86_64-emulated container,
  against the **correct** model file (SHA-256 `92e9e768...`, the one this doc's
  earlier sections call "the deployed model"), both scored **85%** on a 120-image
  sample (30/class) — matching this doc's documented healthy CPU baseline (88.06% on
  the full 1600-image set): one calling `model.predict()` directly on the main
  thread, one via a worker thread (mimicking `app.main`'s
  `run_in_threadpool(predict, ...)`). Identical result either way — rules out a
  threading-related regression too.
- **Decisive test:** in the *same* running, Rosetta-emulated container that had just
  scored 42% over HTTP, replaced only `/code/app/model/tumor_classification_model.keras`
  with the correct-hash (`92e9e768...`) file, restarted the process, and re-ran the
  identical 100-image HTTP `/predict` test: **85.00% accuracy**, glioma recall 80%,
  meningioma 60% — back in line with the documented CPU baseline. Same emulated
  environment, same code, only the model bytes changed. This isolates the cause
  entirely to *which model file gets loaded*, not the environment.

**Root cause independently reconfirmed via the Hugging Face Hub API directly
(bypassing Docker and any local cache entirely)**, matching "RESOLVED" above exactly:
downloading `tumor_classification_model.keras` from revision
`058ed5400e81f56dd9045a5f8fe554fadd60d9fc` via a plain `curl` to `huggingface.co` (no
Docker, no `hf_hub_download`, no local cache) reproduces SHA-256 `2255b19a...` — the
pre-retrain model. The HF Hub repo's own commit history (`GET
/api/models/KellanMcintosh/mri-tumor-classifier/commits/main`) shows this revision's
own commit message ("Retrain: train 99.53% / val 96.70% / test 92.44%...") does not
match its actual file content — corroborating "RESOLVED"'s account that the wrong
local path was pushed to the Hub under a commit message describing the right model.
The later fix commit `bd4045947697284c629d5f2e5a261609f1bab691` ("Fix: upload correct
retrained model...") downloads as SHA-256 `92e9e768...`, byte-identical to
`models/tumor_classification_model_v2.keras` — this is the exact revision now pinned
in `app/config.py` on the current branch (`fix/hf-model-revision-pin`, commit
`ba09111`).

**Confirmed the fix has not yet reached live Cloud Run** (i.e. real users were still
affected at the time of this check). Looked up the live service URL (`gcloud run
services describe mri-tumor-detection-app --region us-central1 --format
'value(status.url)'`, read-only, already-authenticated `gcloud`, no login/config
changes) — `https://mri-tumor-detection-app-p667kknyea-uc.a.run.app`. Sent a single
test image (`Te-gl_1.jpg`, true label glioma) to real production `/predict` over
HTTPS: returned `notumor` with confidence `0.5955949425697327` — **bit-identical to
15 decimal places** to this session's local emulated-build output for the same image
(i.e. still running the broken pre-fix model). Then sent 60 images (15/class) through
real production `/predict`: **41.67% accuracy**, glioma recall 0%, same
notumor/meningioma bias as the local repro. Confirms `fix/hf-model-revision-pin`
(commit `ba09111`) is not yet merged to `main` / deployed — merging and letting the
GitHub Actions workflow redeploy is still an open action item.

**On the emulation question specifically (this session's main task) — ruled out as a
contributing factor, several independent ways:**
- Docker Desktop is configured to emulate `linux/amd64` via **Rosetta**, not QEMU
  (confirmed: the Docker Desktop VM process is launched with a `--rosetta` flag, and
  the `oahd` Rosetta translation daemon is running on the host — checked via `ps aux`
  and the VM launch command line; no Rosetta/virtualization keys were present in
  `~/Library/Group Containers/group.com.docker/settings-store.json`, so this is
  Docker Desktop's current default, not a manual override).
- Basic vectorized numeric sanity checks inside the emulated container came back
  clean: a 512×512 `float32` matmul (both raw NumPy and `tf.matmul`) vs. the same
  computation upcast to `float64` gave max abs diffs of ~2.2e-4 and ~1.0e-4 —
  ordinary float32 rounding, not the 30-80+ magnitude corruption signature of AVX
  mis-emulation or the Metal ReLU-skip bug documented above.
- **Decisive test:** in a freshly built (`docker build --platform linux/amd64`),
  running, Rosetta-emulated container that scored 42.00% over 100 real HTTP
  `POST /predict` requests (25/class; glioma recall 0%, matching the ~45% report),
  replaced only `/code/app/model/tumor_classification_model.keras` inside that
  container with the correct-hash (`92e9e768...`) file, restarted the process, and
  re-ran the identical 100-image HTTP `/predict` test: **85.00% accuracy**, glioma
  recall 80%, meningioma 60% — back in line with this doc's documented CPU baseline
  (88.06% on the full 1600-image set). Same emulated environment, same code, only the
  model bytes changed — isolates the cause entirely to *which model file gets
  loaded*, not the environment, and independently confirms the fix works.
- The real-production HTTP test above (genuine x86_64 hardware, zero emulation)
  reproduced the identical bad behavior as the local emulated build, which is itself
  proof the ~45% number was never an emulation artifact — the same bug is present on
  real hardware.

**Conclusion for the project owner's original question:** the ~45% number was a real
bug, not a Rosetta/QEMU emulation artifact — confirmed independently via direct model
swap (fixes it), matmul/TF sanity checks (clean), and a live production HTTP test
(reproduces the identical bug on real x86_64 Cloud Run hardware). The fix already
exists and is verified correct (`fix/hf-model-revision-pin`, commit `ba09111`) but has
not yet been merged to `main`, so real Cloud Run production was still serving the
broken model as of this check. Next step: merge that branch and let the existing
GitHub Actions workflow redeploy — do not `gcloud run deploy` a manual/local build
out-of-band. A post-deploy smoke test (CI hitting a known-label fixture image against
the freshly deployed revision and asserting the predicted class) would catch this
class of bug automatically in the future.

No code, deployed files, model artifacts, or GCP/Cloud Run infrastructure were
changed by this session. The already-existing fix on `fix/hf-model-revision-pin` was
found in place, not applied by this session. All Docker containers/images created for
this session's reproduction (`mri-tumor-app-investigation` image and its container)
were removed afterward.
