"""Tests for app.preprocessing: bytes in -> normalized array out.

Asserts observable behavior only (shape, dtype, value range) per CLAUDE.md's
testing conventions, not internal implementation details.
"""

import numpy as np
import pytest

from app.config import IMG_SIZE
from app.preprocessing import ImageDecodeError, preprocess_image_bytes


def test_preprocess_known_image_has_correct_shape_dtype_and_range(sample_image_bytes):
    image_bytes = sample_image_bytes("glioma")

    result = preprocess_image_bytes(image_bytes)
    array = np.asarray(result)

    assert array.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert array.dtype == np.float32
    assert array.min() >= 0.0
    assert array.max() <= 1.0


@pytest.mark.parametrize("class_name", ["glioma", "meningioma", "notumor", "pituitary"])
def test_preprocess_each_fixture_image(sample_image_bytes, class_name):
    """All committed fixture images should decode and preprocess cleanly."""
    array = np.asarray(preprocess_image_bytes(sample_image_bytes(class_name)))

    assert array.shape == (1, IMG_SIZE, IMG_SIZE, 3)
    assert array.dtype == np.float32
    assert 0.0 <= array.min() and array.max() <= 1.0


def test_preprocess_rejects_non_image_bytes():
    with pytest.raises(ImageDecodeError):
        preprocess_image_bytes(b"this is definitely not an image file")
