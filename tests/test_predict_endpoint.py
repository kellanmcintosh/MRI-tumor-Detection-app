"""Integration tests for the /predict and /health routes, via FastAPI's
TestClient. Uses real known-label fixture images (Kaggle brain-tumor-mri-dataset
Testing split, one per class) and asserts against the real loaded model —
no mocks, per CLAUDE.md's testing conventions.

Grad-CAM correctness (whether the heatmap highlights the "right" region) is
intentionally not asserted here — see CLAUDE.md's testing conventions. The
`gradcam_overlay` field is only checked for well-formedness: valid base64
that decodes to a PNG image with the original upload's dimensions.
"""

import base64

import tensorflow as tf

from app.config import CLASS_NAMES


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200


def test_predict_rejects_non_image_content_type(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello, this is plain text", "text/plain")},
    )

    assert response.status_code == 400


def test_predict_rejects_corrupted_image(client):
    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"not-actually-jpeg-bytes", "image/jpeg")},
    )

    assert response.status_code == 400


def test_predict_response_shape(client, sample_image_bytes):
    image_bytes = sample_image_bytes("notumor")

    response = client.post(
        "/predict",
        files={"file": ("notumor_01.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()

    assert "predicted_class" in body
    assert "confidences" in body
    assert "gradcam_overlay" in body
    assert body["predicted_class"] in CLASS_NAMES
    assert set(body["confidences"].keys()) == set(CLASS_NAMES)
    for score in body["confidences"].values():
        assert 0.0 <= score <= 1.0


def test_predict_gradcam_overlay_is_well_formed_image(client, sample_image_bytes):
    """The `gradcam_overlay` field must be valid base64 that decodes to a PNG
    image matching the original upload's dimensions. Whether the heatmap
    itself is "correct" is intentionally not asserted — see CLAUDE.md.
    """
    image_bytes = sample_image_bytes("pituitary")

    response = client.post(
        "/predict",
        files={"file": ("pituitary_01.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()

    overlay_bytes = base64.b64decode(body["gradcam_overlay"], validate=True)
    overlay_image = tf.io.decode_png(overlay_bytes, channels=3)
    original_image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)

    assert overlay_image.shape == original_image.shape


def test_predict_matches_known_label_glioma(client, sample_image_bytes):
    image_bytes = sample_image_bytes("glioma")

    response = client.post(
        "/predict",
        files={"file": ("glioma_01.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["predicted_class"] == "glioma"


def test_predict_matches_known_label_meningioma(client, sample_image_bytes):
    image_bytes = sample_image_bytes("meningioma")

    response = client.post(
        "/predict",
        files={"file": ("meningioma_01.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["predicted_class"] == "meningioma"


def test_predict_matches_known_label_notumor(client, sample_image_bytes):
    image_bytes = sample_image_bytes("notumor")

    response = client.post(
        "/predict",
        files={"file": ("notumor_01.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["predicted_class"] == "notumor"


def test_predict_matches_known_label_pituitary(client, sample_image_bytes):
    image_bytes = sample_image_bytes("pituitary")

    response = client.post(
        "/predict",
        files={"file": ("pituitary_01.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["predicted_class"] == "pituitary"
