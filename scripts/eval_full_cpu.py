"""Evaluate a saved model's train/val/test accuracy strictly on CPU.

MODEL_ACCURACY_INVESTIGATION.md found that this model's measured accuracy
depends materially on whether it's served from CPU (tensorflow-cpu, what
production on Cloud Run actually runs) or GPU (tensorflow-metal, only ever
seen on a Mac dev machine). Any fine-tune candidate must be judged on CPU
numbers only -- GPU numbers from this same venv-train environment (which has
tensorflow-metal installed) are not representative of what would ship.

GPU visibility must be disabled before any other TensorFlow op runs, so this
happens at import time, before app.* (which imports tensorflow) is touched.

Reports train/val accuracy (same reproducible 80/20 split as the notebook:
image_dataset_from_directory(..., validation_split=0.2, seed=123)) and test
accuracy + confusion matrix via the real serving path
(app.preprocessing.preprocess_image_bytes -> model.predict), matching the
methodology already used in MODEL_ACCURACY_INVESTIGATION.md so numbers are
directly comparable to the deployed baseline (88.06% overall, 67.25%
meningioma recall).

Usage (from repo root, venv-train activated):
    python -m scripts.eval_full_cpu --model_path models/experiment_ablation_randomerase.keras
"""

import argparse
import os

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
assert tf.config.list_logical_devices("GPU") == [], "Failed to disable GPU visibility"

import numpy as np
from sklearn.metrics import confusion_matrix

from app.config import CLASS_NAMES, IMG_SIZE
from app.model_layers import RandomErasing  # noqa: F401 -- required for load_model deserialization
from app.preprocessing import crop_to_content, preprocess_image_bytes

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(REPO_ROOT, "data", "Training")
TEST_DIR = os.path.join(REPO_ROOT, "data", "Testing")


def crop_data(data):
    def crop_batch(images, labels):
        cropped = tf.map_fn(lambda img: crop_to_content(img, IMG_SIZE), images)
        return cropped, labels
    return data.map(crop_batch)


def normalize_data(data):
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    return data.map(lambda x, y: (normalization_layer(x), y))


def eval_split(model, subset: str) -> float:
    ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32,
        label_mode="int", validation_split=0.2, subset=subset, seed=123,
    )
    ds = normalize_data(crop_data(ds))
    correct = 0
    total = 0
    for images, labels in ds:
        preds = np.argmax(model.predict(images, verbose=0), axis=1)
        correct += int(np.sum(preds == labels.numpy()))
        total += labels.shape[0]
    return correct / total


def load_test_set():
    images = []
    labels = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TEST_DIR, class_name)
        for filename in sorted(os.listdir(class_dir)):
            path = os.path.join(class_dir, filename)
            with open(path, "rb") as f:
                image = preprocess_image_bytes(f.read())
            images.append(image[0])
            labels.append(class_index)
    return tf.stack(images).numpy(), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)

    train_acc = eval_split(model, "training")
    val_acc = eval_split(model, "validation")

    x_test, y_true = load_test_set()
    y_pred = np.argmax(model.predict(x_test, batch_size=32, verbose=0), axis=1)
    test_acc = float(np.mean(y_pred == y_true))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    recalls = cm.diagonal() / cm.sum(axis=1)

    print(f"Model: {args.model_path}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print()
    print("Confusion matrix (rows=true, cols=pred):", CLASS_NAMES)
    print(cm)
    print()
    for name, recall in zip(CLASS_NAMES, recalls):
        print(f"{name} recall: {recall:.4f}")


if __name__ == "__main__":
    main()
