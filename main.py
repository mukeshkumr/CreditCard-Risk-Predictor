import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from artifact_loader import ensure_artifacts

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# MODEL STORAGE
# ============================================================

ml_model = {}


# ============================================================
# LOAD MODEL ARTIFACTS AND MODELS
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Preparing model artifacts...")

    # Download missing artifacts from Hugging Face first.
    artifacts_dir = ensure_artifacts()

    print("Loading ML models...")

    # --------------------------------------------------------
    # Load 5 preprocessors
    # --------------------------------------------------------

    preprocessors = []

    for i in range(5):
        path = (
            artifacts_dir
            / "preprocessors"
            / f"preprocessor_{i}.pkl"
        )

        print(f"Loading preprocessor {i}...")

        preprocessor = joblib.load(path)
        preprocessors.append(preprocessor)

    # --------------------------------------------------------
    # Load 5 native XGBoost models
    # --------------------------------------------------------

    xgb_models = []

    for i in range(5):
        path = (
            artifacts_dir
            / "native_models"
            / f"xgb_model_{i}.ubj"
        )

        print(f"Loading XGBoost model {i}...")

        model = xgb.XGBClassifier()
        model.load_model(str(path))

        xgb_models.append(model)

    # --------------------------------------------------------
    # Load sigmoid calibration parameters
    # --------------------------------------------------------

    calibration_path = artifacts_dir / "calibration.json"

    with calibration_path.open("r", encoding="utf-8") as f:
        calibration_params = json.load(f)

    if len(calibration_params) != 5:
        raise ValueError(
            "Expected calibration parameters for all 5 models."
        )

    print("Calibration parameters loaded.")

    # --------------------------------------------------------
    # Load and validate the decision threshold
    # --------------------------------------------------------

    threshold_path = artifacts_dir / "best_threshold.pkl"
    saved_threshold = joblib.load(threshold_path)
    artifact_threshold = float(np.asarray(saved_threshold).item())

    # Allow an explicit override for testing or deployment.
    override = os.getenv("DECISION_THRESHOLD")

    threshold = (
        float(override)
        if override is not None
        else artifact_threshold
    )

    # A threshold of 1.0 is almost never useful for this
    # risk-classification setup. Fail clearly instead of
    # silently classifying a 95.2% probability as Low Risk.
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"Invalid decision threshold: {threshold}. "
            "Set DECISION_THRESHOLD to a validated value "
            "between 0 and 1, or fix best_threshold.pkl."
        )

    print(f"Decision threshold: {threshold:.4f}")

    # --------------------------------------------------------
    # Store everything for prediction requests
    # --------------------------------------------------------

    ml_model["preprocessors"] = preprocessors
    ml_model["xgb_models"] = xgb_models
    ml_model["calibration"] = calibration_params
    ml_model["threshold"] = threshold

    print("All models loaded successfully.")

    yield

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    ml_model.clear()

    print("Models unloaded.")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Card Risk Predictor",
    lifespan=lifespan,
)


# ============================================================
# INPUT DATA
# ============================================================

class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


# ============================================================
# SIGMOID CALIBRATION
# ============================================================

def sigmoid_calibration(probability, a, b):
    """
    Apply the same sigmoid calibration used by
    sklearn's _SigmoidCalibration.

    p = sigmoid(-(a * probability + b))
    """

    value = -(a * probability + b)
    value = np.clip(value, -500, 500)

    return 1.0 / (1.0 + np.exp(-value))


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: LoanApplication):

    input_df = pd.DataFrame([data.model_dump()])

    calibrated_probabilities = []

    # Run all 5 models, preserving the existing pipeline.
    for i in range(5):

        preprocessor = ml_model["preprocessors"][i]
        xgb_model = ml_model["xgb_models"][i]
        calibration = ml_model["calibration"][i]

        # Preprocessing
        transformed_data = preprocessor.transform(input_df)

        # Raw XGBoost probability
        raw_probability = xgb_model.predict_proba(
            transformed_data
        )[0, 1]

        # Sigmoid calibration
        a = calibration["a"]
        b = calibration["b"]

        calibrated_probability = sigmoid_calibration(
            raw_probability,
            a,
            b,
        )

        calibrated_probabilities.append(
            calibrated_probability
        )

    # Average the five calibrated probabilities.
    probability = float(
        np.mean(calibrated_probabilities)
    )

    # Use the threshold loaded from Hugging Face artifacts.
    threshold = ml_model["threshold"]

    prediction = int(probability >= threshold)

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


# ============================================================
# STATIC FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=str(BASE_DIR / "static"),
        html=True,
    ),
    name="static",
)
