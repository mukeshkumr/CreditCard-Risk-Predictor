import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles


# ============================================================
# MODEL STORAGE
# ============================================================

ml_model = {}


# ============================================================
# LOAD MODELS
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading ML models...")

    # --------------------------------------------------------
    # Load 5 preprocessors
    # --------------------------------------------------------

    preprocessors = []

    for i in range(5):

        path = f"artifacts/preprocessors/preprocessor_{i}.pkl"

        print(f"Loading preprocessor {i}...")

        preprocessor = joblib.load(path)

        preprocessors.append(preprocessor)

    # --------------------------------------------------------
    # Load 5 native XGBoost models
    # --------------------------------------------------------

    xgb_models = []

    for i in range(5):

        path = f"artifacts/native_models/xgb_model_{i}.ubj"

        print(f"Loading XGBoost model {i}...")

        model = xgb.XGBClassifier()

        model.load_model(path)

        xgb_models.append(model)

    # --------------------------------------------------------
    # Load calibration parameters
    # --------------------------------------------------------

    with open("artifacts/calibration.json", "r") as f:
        calibration_params = json.load(f)

    print("Calibration parameters loaded.")

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    threshold = joblib.load(
        "artifacts/best_threshold.pkl"
    )

    print("Threshold:", threshold)

    # --------------------------------------------------------
    # Store everything
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
    lifespan=lifespan
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
    Same sigmoid calibration used by
    sklearn's _SigmoidCalibration.

    p = sigmoid(-(a * probability + b))
    """

    value = -(a * probability + b)

    # Prevent overflow in exp()
    value = np.clip(value, -500, 500)

    return 1.0 / (1.0 + np.exp(-value))


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: LoanApplication):

    # --------------------------------------------------------
    # Convert input into DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame([data.model_dump()])

    calibrated_probabilities = []

    # --------------------------------------------------------
    # Run all 5 models
    # --------------------------------------------------------

    for i in range(5):

        preprocessor = ml_model["preprocessors"][i]

        xgb_model = ml_model["xgb_models"][i]

        calibration = ml_model["calibration"][i]

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        transformed_data = preprocessor.transform(input_df)

        # ----------------------------------------------------
        # XGBoost prediction
        # ----------------------------------------------------

        raw_probability = xgb_model.predict_proba(
            transformed_data
        )[0, 1]

        # ----------------------------------------------------
        # Sigmoid calibration
        # ----------------------------------------------------

        a = calibration["a"]
        b = calibration["b"]

        calibrated_probability = sigmoid_calibration(
            raw_probability,
            a,
            b
        )

        calibrated_probabilities.append(
            calibrated_probability
        )

    # --------------------------------------------------------
    # Average predictions from 5 calibrated classifiers
    # --------------------------------------------------------

    probability = float(
        np.mean(calibrated_probabilities)
    )

    threshold = 0.5  # Default threshold

    prediction = int(
        probability >= threshold
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }


# ============================================================
# STATIC FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory="static",
        html=True
    ),
    name="static"
)