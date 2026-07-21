"""Preprocessed array -> predicted class label + per-class confidence.

Loads the Keras model once (see `get_model`) and exposes a simple
array-in/prediction-out interface, independent of any HTTP concerns.

The model file is expected to already be present at `app.config.MODEL_PATH`,
placed there by `scripts/download_model.py` at Docker *build* time. This
module never touches the network -- see CLAUDE.md's "Architecture decisions"
section.
"""

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from app.config import CLASS_NAMES, MODEL_PATH


@lru_cache(maxsize=1)
def get_model() -> tf.keras.Model:
    """Load the trained model once and cache it for the process lifetime."""
    return tf.keras.models.load_model(MODEL_PATH)


def predict(preprocessed_image: tf.Tensor) -> Tuple[str, Dict[str, float]]:
    """Run inference on a preprocessed image batch.

    Args:
        preprocessed_image: array/tensor of shape (1, IMG_SIZE, IMG_SIZE, 3),
            as produced by `app.preprocessing.preprocess_image_bytes`.

    Returns:
        A tuple of (predicted_class, confidences) where `predicted_class` is
        the highest-confidence label from `CLASS_NAMES`, and `confidences` is
        a dict mapping every class name to its softmax probability.
    """
    model = get_model()
    raw_predictions = model.predict(preprocessed_image, verbose=0)
    probabilities = np.asarray(raw_predictions[0])

    confidences = {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    }
    predicted_class = CLASS_NAMES[int(np.argmax(probabilities))]

    return predicted_class, confidences
