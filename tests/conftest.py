"""Shared pytest fixtures: a TestClient (which also loads the real model via
the app's startup lifespan) and helpers for reading known-label sample images.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sample_images")

# Known-label fixture images, sourced from the Kaggle brain-tumor-mri-dataset's
# Testing split (see tests/fixtures/sample_images/ docstring in the PR for
# provenance). Filenames map directly to the true class label.
SAMPLE_IMAGES = {
    "glioma": "glioma_01.jpg",
    "meningioma": "meningioma_01.jpg",
    "notumor": "notumor_01.jpg",
    "pituitary": "pituitary_01.jpg",
}


@pytest.fixture(scope="session")
def client():
    """A TestClient with the app's startup lifespan applied, so the real
    model is downloaded (if needed) and loaded once for the whole test
    session rather than per-test.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_image_path():
    """Returns a function mapping a class name to its fixture image path."""

    def _path(class_name: str) -> str:
        return os.path.join(FIXTURES_DIR, SAMPLE_IMAGES[class_name])

    return _path


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image_path):
    """Returns a function mapping a class name to its fixture image bytes."""

    def _bytes(class_name: str) -> bytes:
        with open(sample_image_path(class_name), "rb") as f:
            return f.read()

    return _bytes
