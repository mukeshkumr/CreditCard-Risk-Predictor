"""
This file help to load the articats folder from Hugging Face model repository if it is not present in the local directory."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

REQUIRED_FILES = [
    "preprocessors/preprocessor_0.pkl",
    "preprocessors/preprocessor_1.pkl",
    "preprocessors/preprocessor_2.pkl",
    "preprocessors/preprocessor_3.pkl",
    "preprocessors/preprocessor_4.pkl",
    "native_models/xgb_model_0.ubj",
    "native_models/xgb_model_1.ubj",
    "native_models/xgb_model_2.ubj",
    "native_models/xgb_model_3.ubj",
    "native_models/xgb_model_4.ubj",
    "calibration.json",
    "best_threshold.pkl",
]


def ensure_artifacts() -> Path:
    repo_id = os.getenv(
        "HF_MODEL_REPO",
        "crash12/creditcard-risk-artifacts",
    )

    # Download the repository's artifacts/ subdirectory
    # directly into the local artifacts/ directory.
    print(f"Downloading model artifacts from {repo_id}...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(BASE_DIR),
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=[f"artifacts/{name}" for name in REQUIRED_FILES],
    )

    missing = [
        name
        for name in REQUIRED_FILES
        if not (ARTIFACTS_DIR / name).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Model artifacts are missing after download: "
            + ", ".join(missing)
        )

    print(f"All model artifacts are available in: {ARTIFACTS_DIR}")
    return ARTIFACTS_DIR