"""Isolated ablation fine-tunes of the deployed model, continuing from
models/best_checkpoint_v2_resumed2.keras (the currently-deployed checkpoint).

See MODEL_ACCURACY_INVESTIGATION.md: a prior fine-tune combined a reduced
RandomErasing area_frac with per-class loss weighting in one run, which
looked like a big win on GPU (Metal) but turned out to be no real
improvement once re-evaluated correctly on CPU. This script re-runs the two
changes in isolation so each can be attributed independently. Training here
runs on whatever backend is available (GPU/Metal, for speed) -- only
evaluation (scripts/eval_full_cpu.py) needs to be forced onto CPU, since
that's what production actually serves and what the saved weights will be
judged against.

Usage (from repo root, venv-train activated):
    python -m scripts.finetune_ablation --ablation randomerase --output models/experiment_ablation_randomerase.keras
    python -m scripts.finetune_ablation --ablation classweight --output models/experiment_ablation_classweight.keras
"""

import argparse
import os

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from app.config import IMG_SIZE
from app.model_layers import RandomErasing  # noqa: F401 -- required for load_model deserialization
from app.preprocessing import crop_to_content

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "Training")
BASE_CHECKPOINT = os.path.join(REPO_ROOT, "models", "best_checkpoint_v2_resumed2.keras")


def load_train_val():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32,
        label_mode="int", validation_split=0.2, subset="training", seed=123,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32,
        label_mode="int", validation_split=0.2, subset="validation", seed=123,
    )
    return train_ds, val_ds


def crop_data(data):
    def crop_batch(images, labels):
        cropped = tf.map_fn(lambda img: crop_to_content(img, IMG_SIZE), images)
        return cropped, labels
    return data.map(crop_batch)


def normalize_data(data):
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    return data.map(lambda x, y: (normalization_layer(x), y))


def find_random_erasing_layer(model):
    for layer in model.layers:
        if isinstance(layer, RandomErasing):
            return layer
    raise ValueError("No RandomErasing layer found in the loaded model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["randomerase", "classweight"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    train_ds, val_ds = load_train_val()
    train_ds = normalize_data(crop_data(train_ds))
    val_ds = normalize_data(crop_data(val_ds))

    model = tf.keras.models.load_model(BASE_CHECKPOINT)

    class_weight = None
    if args.ablation == "randomerase":
        erasing_layer = find_random_erasing_layer(model)
        erasing_layer.area_frac = (0.01, 0.06)
        print(f"RandomErasing.area_frac set to {erasing_layer.area_frac}; class_weight=None")
    else:
        class_weight = {0: 1.3, 1: 1.5, 2: 1.0, 3: 1.0}
        print(f"RandomErasing.area_frac left at default; class_weight={class_weight}")

    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=2e-4),
        metrics=["accuracy"],
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    checkpoint_path = args.output
    earlystopping = tf.keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, monitor="val_accuracy", mode="max"
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max"
    )

    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[earlystopping, checkpoint],
    )

    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
