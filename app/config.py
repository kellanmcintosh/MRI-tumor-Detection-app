"""Central configuration: image sizing, class labels, and model location.

Class order is fixed and must never be reordered without re-verifying against
the training run's output — see CLAUDE.md's "Architecture decisions" section.
`image_dataset_from_directory` assigns label indices in alphabetical directory
order, so this list mirrors that: glioma=0, meningioma=1, notumor=2, pituitary=3.
"""

import os

IMG_SIZE = 256

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Hugging Face Hub location of the trained model artifact (see MODEL_NOTES.md).
# Pinned to a specific commit revision on purpose — never track a mutable ref
# like "main" here, or unrelated pushes to the model repo could silently swap
# out the live model.
HF_REPO_ID = "KellanMcintosh/mri-tumor-classifier"
HF_REVISION = "a35027761d87b4b738d4d8d70efba85266e57f53"
HF_MODEL_FILENAME = "tumor_classification_model.keras"

# Local on-disk cache for the downloaded model file. Gitignored (see .gitignore's
# `app/model/` entry) — populated by `scripts/download_model.py` at Docker
# build time; the running container never fetches it over the network.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, HF_MODEL_FILENAME)
