"""Build-time script: pulls the pinned Hugging Face Hub model revision to
`app/model/`. Run during `docker build` (see Dockerfile), never at container
startup/runtime -- see CLAUDE.md's "Architecture decisions" section.
"""

from huggingface_hub import hf_hub_download

from app.config import HF_MODEL_FILENAME, HF_REPO_ID, HF_REVISION, MODEL_DIR


def main() -> None:
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        revision=HF_REVISION,
        local_dir=MODEL_DIR,
    )
    print(f"Downloaded model to {path}")


if __name__ == "__main__":
    main()
