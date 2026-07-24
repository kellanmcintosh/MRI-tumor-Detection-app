"""Compute a confusion matrix for the currently trained model against the
Kaggle test split, and merge it into models/tumor_classification_model_metrics.json.

Reuses app.preprocessing so the matrix reflects the exact same decode/resize
path the deployed API uses -- not a reimplementation that could quietly skew
the numbers. Run this after any retrain to keep the frontend's reported
confusion matrix in sync with the model actually being served.

Usage (from repo root, with requirements-train.txt installed):
    python -m scripts.compute_confusion_matrix
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from app.config import CLASS_NAMES, MODEL_PATH

# Registers the custom RandomErasing layer baked into the model so
# load_model can deserialize it below -- unused directly in this module,
# but the import itself is what triggers the registration (same reason
# app/inference.py imports this).
from app import model_layers  # noqa: F401

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(REPO_ROOT, "data", "Testing")
METRICS_PATH = os.path.join(REPO_ROOT, "models", "tumor_classification_model_metrics.json")


def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    from app.preprocessing import preprocess_image_bytes

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
    model = tf.keras.models.load_model(MODEL_PATH)

    x_test, y_true = load_test_set()
    y_pred = np.argmax(model.predict(x_test, batch_size=32), axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    metrics["confusion_matrix"] = {
        "labels": CLASS_NAMES,
        "matrix": cm.tolist(),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(json.dumps(metrics["confusion_matrix"], indent=2))


if __name__ == "__main__":
    main()
