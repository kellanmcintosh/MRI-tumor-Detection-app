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


def content_bounding_box_crop(image: tf.Tensor, threshold: float = 10.0) -> tf.Tensor:
    """Crop out the uniform black background surrounding the head/skull,
    then zero out any *remaining* background pixels within that crop.
    Returns the crop at its native (variable) size -- callers that need a
    fixed size should resize the result themselves; `crop_to_content` below
    does that for the model input path.

    Grad-CAM investigation (see MODEL_NOTES.md) showed the model heavily
    keying off the black background rather than brain tissue -- a known
    shortcut-learning risk with this dataset (it merges three source
    datasets with different backgrounds/cropping per class). The bounding
    -box crop alone wasn't enough: a round/oval head inscribed in its own
    bounding rectangle still leaves black triangles in the four corners
    (geometry, not a bug), and Grad-CAM showed the model still exploiting
    exactly those leftover corners on some classes even after cropping.
    Masking every background pixel -- not just the ones in fully-black
    rows/columns -- removes them regardless of position, following the
    actual (irregular) head silhouette rather than a rectangle. This is
    still just the same brightness threshold already used for the crop
    bounds, not a trained segmentation model, so it doesn't introduce the
    "stripping induces a shortcut on the stylized contour" failure mode a
    real skull-stripping algorithm can (the boundary here is exactly
    wherever each scan's real pixel intensities happen to cross the
    threshold, not an algorithm's smoothed/stylized idea of one).

    Must run identically at train and serve time (see module docstring) --
    this is imported directly by the training notebook for that reason.

    Deliberately avoids `tf.cond` with differently-shaped branches (one
    returning the crop, one returning the original image) -- that pattern
    hung indefinitely partway through a real training run on this machine's
    tensorflow-metal (Apple Silicon GPU) backend, inside a
    `tf.data.Dataset.map()`. Computing `tf.reduce_min`/`tf.reduce_max` on an
    empty selection is well-defined in TF (returns the dtype's max/min as an
    identity element, not an error or NaN), so the bounding box can be
    computed unconditionally and just clamped to the full image via
    `tf.where` on the four scalar coordinates -- one straight-line path, no
    branch, no hang.
    """
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    gray = tf.reduce_max(image, axis=-1)
    mask = gray > threshold
    rows = tf.where(tf.reduce_any(mask, axis=1))
    cols = tf.where(tf.reduce_any(mask, axis=0))
    is_empty = tf.logical_or(tf.equal(tf.size(rows), 0), tf.equal(tf.size(cols), 0))

    row_min = tf.where(is_empty, 0, tf.cast(tf.reduce_min(rows), tf.int32))
    row_max = tf.where(is_empty, height - 1, tf.cast(tf.reduce_max(rows), tf.int32))
    col_min = tf.where(is_empty, 0, tf.cast(tf.reduce_min(cols), tf.int32))
    col_max = tf.where(is_empty, width - 1, tf.cast(tf.reduce_max(cols), tf.int32))

    masked = image * tf.cast(mask[..., tf.newaxis], image.dtype)

    return tf.image.crop_to_bounding_box(
        masked, row_min, col_min, row_max - row_min + 1, col_max - col_min + 1
    )


def crop_to_content(image: tf.Tensor, target_size: int, threshold: float = 10.0) -> tf.Tensor:
    """`content_bounding_box_crop` followed by a resize to a fixed
    `target_size` x `target_size` -- the model input path needs a fixed
    size, so this is what `preprocess_image_bytes` and the training
    notebook's data pipeline both call.
    """
    cropped = content_bounding_box_crop(image, threshold)
    return tf.image.resize(cropped, [target_size, target_size], method="bilinear")


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
    image = crop_to_content(image, IMG_SIZE)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image
