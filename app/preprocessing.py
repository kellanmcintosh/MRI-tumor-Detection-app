"""Raw upload bytes -> normalized model input array.

Deliberately implemented with TensorFlow's own image ops rather than PIL or
OpenCV. The model was trained on images resized by
`tf.keras.utils.image_dataset_from_directory`, which decodes and resizes
images internally using TF's ops. A PIL-based reimplementation uses a
different resize/interpolation algorithm and would introduce train/serve
skew — see CLAUDE.md's "Architecture decisions" section. Keep this TF-native
if you ever touch it.
"""

import tensorflow as tf

from app.config import IMG_SIZE


class ImageDecodeError(ValueError):
    """Raised when the given bytes cannot be decoded as an image."""


def preprocess_image_bytes(image_bytes: bytes) -> tf.Tensor:
    """Convert raw uploaded image bytes into a normalized model input.

    Args:
        image_bytes: Raw bytes of an uploaded image file (JPEG/PNG/BMP/GIF).

    Returns:
        A float32 tensor of shape (1, IMG_SIZE, IMG_SIZE, 3) with values
        scaled to the range [0.0, 1.0], matching the training-time
        preprocessing pipeline exactly.

    Raises:
        ImageDecodeError: if `image_bytes` cannot be decoded as an image.
    """
    try:
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    except (tf.errors.InvalidArgumentError, ValueError) as exc:
        raise ImageDecodeError(f"Could not decode image: {exc}") from exc

    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method="bilinear")
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image
