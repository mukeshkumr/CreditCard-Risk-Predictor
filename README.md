<div align="center">

# 🏦 Credit Card Risk Predictor

**An end-to-end machine learning system for consumer loan default prediction**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./LICENSE)

*Production-grade loan risk assessment powered by an ensemble of calibrated XGBoost classifiers with a sleek, browser-based underwriting desk.*

</div>

---

## 📋 Table of Contents

- [🏦 Credit Card Risk Predictor](#-credit-card-risk-predictor)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔍 Overview](#-overview)
  - [🏗️ Architecture](#️-architecture)
  - [✨ Features](#-features)
  - [📊 Dataset](#-dataset)
  - [🧠 Model Pipeline](#-model-pipeline)
    - [Key Design Decisions](#key-design-decisions)
  - [📁 Project Structure](#-project-structure)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create \& Activate a Virtual Environment](#2-create--activate-a-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Pull Data \& Model Artifacts via DVC](#4-pull-data--model-artifacts-via-dvc)
    - [5. Configure Environment](#5-configure-environment)
    - [6. Run the Application](#6-run-the-application)
  - [📡 API Reference](#-api-reference)
    - [`POST /predict`](#post-predict)
  - [🖥️ Web Interface — Credit Ledger](#️-web-interface--credit-ledger)
  - [🛠️ Tech Stack](#️-tech-stack)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)

---

## 🔍 Overview

**Credit Card Risk Predictor** is a full-stack machine learning application that assesses the probability of a consumer defaulting on a loan. It combines a rigorous ML training pipeline with a production-ready REST API (FastAPI + Uvicorn) and an elegant, browser-based underwriting interface — *Credit Ledger*.

The system uses a **5-fold cross-validated ensemble** of XGBoost classifiers, each independently calibrated via **Platt scaling (sigmoid calibration)** to produce well-calibrated probability estimates. The final default probability is the mean of all five calibrated predictions.

> **Disclaimer:** Model outputs are statistical estimates for research and educational purposes only. They do not constitute a lending decision or financial advice.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client (Browser)                          │
│                     Credit Ledger UI (HTML/CSS/JS)               │
└────────────────────────────┬─────────────────────────────────────┘
                             │  POST /predict  (JSON)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (main.py)                  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Lifespan: Model Loading                     │    │
│  │  preprocessor_0..4.pkl   xgb_model_0..4.ubj             │    │
│  │  calibration.json        best_threshold.pkl              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────┐    ┌─────────────────┐   ┌─────────────────┐  │
│  │ Preprocessor │───▶│ XGBoost Predict │──▶│ Sigmoid Calibr. │  │
│  │  (x5 folds)  │    │   (x5 folds)   │   │   (x5 folds)    │  │
│  └──────────────┘    └─────────────────┘   └────────┬────────┘  │
│                                                      │           │
│                                               np.mean(x5)        │
│                                                      │           │
│                                              ┌───────▼────────┐  │
│                                              │ Threshold (0.5)│  │
│                                              │ Low / High Risk│  │
│                                              └────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                             │
              Static files mounted at "/"
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Ensemble Inference** | 5-fold ensemble of XGBoost classifiers for robust predictions |
| 📐 **Probability Calibration** | Platt scaling (sigmoid) per model for well-calibrated probabilities |
| ⚡ **Async Startup** | Models loaded once at startup via FastAPI lifespan — zero cold-start per request |
| 🌐 **REST API** | Clean `POST /predict` endpoint with full Pydantic schema validation |
| 🖥️ **Built-in UI** | Browser-based underwriting desk served directly from the API |
| 📊 **Interactive Gauge** | Animated SVG gauge visualising default probability in real-time |
| 🔄 **DVC Integration** | Data and artifact versioning with DVC for full reproducibility |
| 📓 **Research Notebook** | Jupyter notebook covering full EDA, training, and evaluation |

---

## 📊 Dataset

The model is trained on the **Credit Risk Dataset** (`data/credit_risk_dataset.csv`), a public dataset (~32,000 records) that captures consumer loan application features.

| Feature | Type | Description |
|---|---|---|
| `person_age` | Integer | Age of the applicant (years) |
| `person_income` | Float | Annual income |
| `person_home_ownership` | Categorical | `RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| `person_emp_length` | Float | Employment length (years) |
| `loan_intent` | Categorical | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| `loan_grade` | Categorical | Loan grade `A` to `G` |
| `loan_amnt` | Float | Loan amount requested |
| `loan_int_rate` | Float | Interest rate (%) |
| `loan_percent_income` | Float | Loan amount as a ratio of income |
| `cb_person_default_on_file` | Categorical | Prior default on credit bureau: `Y` / `N` |
| `cb_person_cred_hist_length` | Integer | Length of credit history (years) |
| **`loan_status`** | Binary | **Target** — `1` = Default, `0` = Non-default |

> Data and artifacts are tracked with [DVC](https://dvc.org/) and excluded from version control via `.gitignore`. See `data.dvc` and `artifacts.dvc` to pull them.

---

## 🧠 Model Pipeline

The full training workflow is documented in [`notebooks/Credit_Risk.ipynb`](./notebooks/Credit_Risk.ipynb).

```
Raw CSV Data
    |
    v
+-----------------------------------+
| Exploratory Data Analysis (EDA)   |
| - Missing value imputation        |
| - Outlier analysis                |
| - Feature correlation (SHAP)      |
+------------------+----------------+
                   |
                   v
+-----------------------------------+
|   5-Fold Stratified K-Fold CV     |
|                                   |
|  For each fold:                   |
|  +- ColumnTransformer             |
|  |   +- StandardScaler (num)      |
|  |   +- OrdinalEncoder (cat)      |
|  +- XGBClassifier (trained)       |
|  +- Sigmoid Calibration (a, b)    |
+------------------+----------------+
                   |
                   v
+-----------------------------------+
|  artifacts/                       |
|  +- preprocessors/                |
|  |   +- preprocessor_0..4.pkl     |
|  +- native_models/                |
|  |   +- xgb_model_0..4.ubj        |
|  +- calibration.json              |
|  +- best_threshold.pkl            |
+-----------------------------------+
```

### Key Design Decisions

- **Native XGBoost format (`.ubj`)** — models are saved in XGBoost's native binary format for faster load times and full version compatibility.
- **Per-fold calibration** — sigmoid calibration parameters `a` and `b` are fitted independently per fold and stored in `calibration.json`.
- **Threshold optimisation** — the decision threshold is tuned on the validation set and persisted in `best_threshold.pkl` (default: `0.5`).
- **SHAP analysis** — feature importance is analysed with SHAP values inside the notebook.

---

## 📁 Project Structure

```
CreditCard-Risk-Predictor/
|
+-- artifacts/                    # Trained model artefacts (DVC-tracked)
|   +-- preprocessors/
|   |   +-- preprocessor_{0-4}.pkl   # 5 fitted ColumnTransformers
|   +-- native_models/
|   |   +-- xgb_model_{0-4}.ubj      # 5 trained XGBoost models (native format)
|   +-- calibration.json             # Sigmoid calibration params per fold
|   +-- best_threshold.pkl           # Optimised decision threshold
|
+-- data/                         # Dataset (DVC-tracked)
|   +-- credit_risk_dataset.csv      # ~32K consumer loan records
|
+-- notebooks/
|   +-- Credit_Risk.ipynb            # Full EDA, training & evaluation notebook
|
+-- static/                       # Credit Ledger — browser UI
|   +-- index.html                   # Underwriting desk form
|   +-- style.css                    # IBM Plex / Source Serif typography & styles
|   +-- script.js                    # API client, gauge animation, form logic
|
+-- main.py                       # FastAPI application & inference logic
+-- pyproject.toml                # Project metadata & dependencies
+-- requirements.txt              # Pinned dependency list
+-- .env                          # Local environment variables (not committed)
+-- .dvcignore                    # DVC ignore rules
+-- artifacts.dvc                 # DVC pointer for artifacts/
+-- data.dvc                      # DVC pointer for data/
+-- README.md                     # This file
+-- LICENSE                       # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+** — see `.python-version`
- **[uv](https://docs.astral.sh/uv/)** *(recommended)* or `pip`
- **[DVC](https://dvc.org/)** — to pull data and model artifacts

### 1. Clone the Repository

```bash
git clone https://github.com/mukeshkumr/CreditCard-Risk-Predictor.git
cd CreditCard-Risk-Predictor
```

### 2. Create & Activate a Virtual Environment

```bash
# Using uv (recommended)
uv venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Or with standard pip
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Using uv
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 4. Pull Data & Model Artifacts via DVC

```bash
dvc pull
```

> If you don't have DVC configured with a remote, you can train from scratch using the notebook:
> ```bash
> jupyter notebook notebooks/Credit_Risk.ipynb
> ```

### 5. Configure Environment

The `.env` file stores local configuration. Defaults are shown below:

```env
MODEL_PATH=artifacts/credit_risk_model.pkl
THRESHOLD_PATH=artifacts/best_threshold.pkl
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=True
```

### 6. Run the Application

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000` | Credit Ledger underwriting UI |
| `http://127.0.0.1:8000/docs` | Interactive Swagger API docs |
| `http://127.0.0.1:8000/redoc` | ReDoc API documentation |

---

## 📡 API Reference

### `POST /predict`

Submits a loan application and returns a risk assessment.

**Request body** (`application/json`)

```json
{
  "person_age": 30,
  "person_income": 600000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 100000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

**Field Reference**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `person_age` | `int` | 18 – 100 | Applicant age in years |
| `person_income` | `float` | >= 0 | Annual income |
| `person_home_ownership` | `str` | `RENT`, `MORTGAGE`, `OWN`, `OTHER` | Home ownership status |
| `person_emp_length` | `float` | 0 – 60 | Employment length in years |
| `loan_intent` | `str` | See dataset table | Purpose of the loan |
| `loan_grade` | `str` | `A` – `G` | Loan grade assigned |
| `loan_amnt` | `float` | >= 0 | Loan amount |
| `loan_int_rate` | `float` | 0 – 40 | Interest rate (%) |
| `loan_percent_income` | `float` | 0 – 1 | `loan_amnt / person_income` |
| `cb_person_default_on_file` | `str` | `Y`, `N` | Prior default on credit bureau |
| `cb_person_cred_hist_length` | `int` | 0 – 60 | Credit history length in years |

**Response** (`200 OK`)

```json
{
  "default_probability": 0.1342,
  "default_prediction": 0,
  "threshold": 0.5,
  "Result": "Low Risk"
}
```

| Field | Type | Description |
|---|---|---|
| `default_probability` | `float` | Mean calibrated probability of default (0–1) |
| `default_prediction` | `int` | Binary prediction — `0` = Non-default, `1` = Default |
| `threshold` | `float` | Decision threshold used |
| `Result` | `str` | Human-readable verdict: `"Low Risk"` or `"High Risk"` |

---

## 🖥️ Web Interface — Credit Ledger

The **Credit Ledger** is a bespoke underwriting desk served directly from the FastAPI application at `/`.

- **Typography:** IBM Plex Sans · IBM Plex Mono · Source Serif 4
- **Form sections:** Applicant info → Loan request → Credit bureau file
- **Auto-calculation:** `loan_percent_income` is computed live from income and loan amount
- **Result display:** Animated SVG gauge showing default probability with threshold marker
- **Risk stamp:** Animated `LOW RISK` / `HIGH RISK` verdict badge
- **Service status:** Live indicator showing whether the API is reachable

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **Web Framework** | FastAPI 0.115 + Uvicorn |
| **ML Library** | XGBoost 3.4.1, scikit-learn 1.6.1 |
| **Data Processing** | Pandas 3.x, NumPy 2.1 |
| **Serialisation** | joblib (preprocessors), XGBoost native `.ubj` (models) |
| **Explainability** | SHAP 0.52 |
| **Visualisation** | Matplotlib, Seaborn (notebook) |
| **Data Versioning** | DVC |
| **Linting** | Ruff |
| **Package Manager** | uv / pip |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript (ES2020+) |

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** your changes: `git commit -m 'Add some feature'`
4. **Push** to the branch: `git push origin feature/your-feature`
5. **Open** a Pull Request

Please ensure your code passes `ruff check .` before submitting.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---


