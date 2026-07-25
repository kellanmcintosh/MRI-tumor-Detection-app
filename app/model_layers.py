"""Custom Keras layers baked into the model itself (not the preprocessing
pipeline), so they need to be importable by both the training notebook and
the serving path.

`tf.keras.models.load_model` can't deserialize a custom layer unless the
class is both registered (`register_keras_serializable`) *and* actually
imported in whatever process is loading the model -- `app/inference.py`
imports this module for exactly that reason, even though it never
constructs `RandomErasing` directly itself.
"""

import keras
import tensorflow as tf
from tensorflow.keras import layers


@keras.saving.register_keras_serializable(package="mri_classifier")
class RandomErasing(layers.Layer):
    """Randomly zeroes a rectangular region per image during training so the
    model can't reliably key off any one fixed region -- added after
    Grad-CAM showed heavy reliance on the black background/border rather
    than brain tissue (see docs/MODEL_NOTES.md). No-op at inference
    (training=False), same convention as RandomFlip/RandomRotation, so
    `app/preprocessing.py` and the rest of `app/inference.py` need no
    changes for this.
    """

    def __init__(self, area_frac=(0.02, 0.15), **kwargs):
        super().__init__(**kwargs)
        self.area_frac = area_frac

    def get_config(self):
        config = super().get_config()
        config.update({"area_frac": self.area_frac})
        return config

    def call(self, images, training=False):
        if not training:
            return images

        def erase_one(img):
            height = tf.shape(img)[0]
            width = tf.shape(img)[1]
            channels = tf.shape(img)[2]
            frac = tf.random.uniform([], self.area_frac[0], self.area_frac[1])
            erase_h = tf.maximum(tf.cast(tf.sqrt(frac) * tf.cast(height, tf.float32), tf.int32), 1)
            erase_w = tf.maximum(tf.cast(tf.sqrt(frac) * tf.cast(width, tf.float32), tf.int32), 1)
            top = tf.random.uniform([], 0, height - erase_h + 1, dtype=tf.int32)
            left = tf.random.uniform([], 0, width - erase_w + 1, dtype=tf.int32)
            hole = tf.zeros([erase_h, erase_w, channels], dtype=img.dtype)
            keep_mask = tf.pad(
                hole,
                [[top, height - top - erase_h], [left, width - left - erase_w], [0, 0]],
                constant_values=1,
            )
            return img * keep_mask

        return tf.map_fn(erase_one, images)
