"""Preprocessed array -> predicted class label + per-class confidence.

Loads the Keras model once (see `get_model`) and exposes a simple
array-in/prediction-out interface, independent of any HTTP concerns.

Model sourcing note (this slice only): the model is pulled from the pinned
Hugging Face Hub revision recorded in `app/config.py` via `hf_hub_download`
the first time `get_model()` is called, and cached under `app/model/`
(gitignored). This is a deliberate, scoped deviation from the target
architecture described in CLAUDE.md, where the model is fetched at Docker
*build* time by `scripts/download_model.py` and the running container never
touches the network. That build pipeline is out of scope for this issue
(tracked separately); this module's runtime download only exists to unblock
local development and testing of the inference path.
"""

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

from app.config import (
    CLASS_NAMES,
    HF_MODEL_FILENAME,
    HF_REPO_ID,
    HF_REVISION,
    MODEL_DIR,
)


def _download_model() -> str:
    """Fetch the pinned model revision from the Hugging Face Hub, if needed.

    `hf_hub_download` caches by revision hash, so repeated calls (e.g. across
    test runs) are cheap no-ops once the file is on disk.
    """
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        revision=HF_REVISION,
        local_dir=MODEL_DIR,
    )


@lru_cache(maxsize=1)
def get_model() -> tf.keras.Model:
    """Load the trained model once and cache it for the process lifetime."""
    model_path = _download_model()
    return tf.keras.models.load_model(model_path)


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
