"""Model + preprocessed array + original image bytes -> Grad-CAM heatmap overlay.

Deep module: `generate_gradcam_overlay` is the only public entry point. It
hides target-layer discovery, gradient computation, colormap application, and
alpha-blending. `app/main.py` should only ever call this one function.

The target conv layer is found programmatically
(`isinstance(layer, tf.keras.layers.Conv2D)`, last match) -- never a
hardcoded layer name/index. The model was built with a loop that
auto-names layers (`conv2d`, `conv2d_1`, ...), so a hardcoded name would
silently break if the architecture is ever retrained with different args.
See CLAUDE.md's "Architecture decisions" section.
"""

import numpy as np
import tensorflow as tf


class GradCAMError(RuntimeError):
    """Raised when a Grad-CAM heatmap cannot be produced for a given model."""


def _find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Return the last Conv2D layer in the model, in layer-definition order."""
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise GradCAMError("Model has no Conv2D layers; cannot compute Grad-CAM.")
    return conv_layers[-1]


def _compute_heatmap(
    model: tf.keras.Model,
    preprocessed_image: tf.Tensor,
    target_layer: tf.keras.layers.Layer,
) -> np.ndarray:
    """Run a gradient-tracked forward pass and return a normalized [0, 1]
    heatmap over the target conv layer's spatial activations.

    Deliberately replays the model layer-by-layer inside the `GradientTape`
    context (rather than building a separate `tf.keras.Model` from
    `target_layer.output` and `model.output`, the textbook Grad-CAM
    approach) -- that route produced `None` gradients here, likely because a
    freshly-composed Functional model from an existing Sequential's
    intermediate tensors doesn't retrace as part of the same tape-tracked
    call. Manually replaying `model.layers` guarantees the target layer's
    activations and the final predictions come from one continuous,
    tape-tracked graph.
    """
    image_tensor = tf.convert_to_tensor(preprocessed_image)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        activations = image_tensor
        conv_output = None
        for layer in model.layers:
            activations = layer(activations, training=False)
            if layer is target_layer:
                conv_output = activations
        predictions = activations
        predicted_index = tf.argmax(predictions[0])
        class_score = predictions[:, predicted_index]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise GradCAMError("Could not compute gradients for Grad-CAM (grads was None).")

    # Global-average-pool the gradients per channel, then weight each
    # feature map in the target layer's output by how much it mattered to
    # the predicted class score.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    weighted_activations = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(weighted_activations)

    heatmap = tf.maximum(heatmap, 0)  # ReLU: keep only features that increase the class score
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + 1e-8)  # normalize to [0, 1]; epsilon guards an all-zero map
    return heatmap.numpy()


def _apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a [0, 1] heatmap to RGB (0-255) using a closed-form approximation
    of the "jet" colormap (blue -> cyan -> yellow -> red).

    Implemented directly with numpy rather than pulling in matplotlib or
    opencv, since both are heavier training-only dependencies that
    `requirements-api.txt` deliberately keeps out of the deployed image
    (see CLAUDE.md).
    """
    red = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    return np.stack([red, green, blue], axis=-1) * 255.0


def generate_gradcam_overlay(
    model: tf.keras.Model,
    preprocessed_image: tf.Tensor,
    original_image_bytes: bytes,
    alpha: float = 0.4,
) -> bytes:
    """Produce a Grad-CAM heatmap alpha-blended over the original uploaded image.

    Args:
        model: the loaded Keras classification model, as returned by
            `app.inference.get_model`.
        preprocessed_image: the same normalized `(1, IMG_SIZE, IMG_SIZE, 3)`
            array used for inference, as produced by
            `app.preprocessing.preprocess_image_bytes`. Used to run the
            gradient-tracked forward pass through the model.
        original_image_bytes: the raw uploaded image bytes. Used only to
            recover the original resolution and pixel data, so the overlay
            is rendered at the uploaded image's native size rather than the
            downscaled 256x256 model input.
        alpha: blend strength of the heatmap colormap over the original
            image, in `[0, 1]`. Higher values make the heatmap more opaque.

    Returns:
        PNG-encoded bytes of the original image with the Grad-CAM heatmap
        overlaid, at the original image's resolution.

    Raises:
        GradCAMError: if the model has no Conv2D layers, or gradients could
            not be computed for the target layer.
    """
    target_layer = _find_last_conv_layer(model)
    heatmap = _compute_heatmap(model, preprocessed_image, target_layer)

    original_image = tf.io.decode_image(original_image_bytes, channels=3, expand_animations=False)
    height, width = original_image.shape[0], original_image.shape[1]

    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], [height, width], method="bilinear")
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()
    heatmap_rgb = _apply_jet_colormap(heatmap_resized)

    original_rgb = tf.cast(original_image, tf.float32).numpy()
    overlay = original_rgb * (1 - alpha) + heatmap_rgb * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return tf.io.encode_png(overlay).numpy()
