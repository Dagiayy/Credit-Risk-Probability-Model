

# Credit Risk Probability Model

## Overview

This project is part of **Bati Bank's initiative** to enable a Buy-Now-Pay-Later (BNPL) credit offering for an eCommerce partner. It predicts customer creditworthiness based on transaction behavior — enabling responsible lending decisions without relying on traditional credit histories.

---

## 🔍 Project Goals

* Build a proxy variable to classify customers as **high-risk or low-risk** based on transactions.
* Engineer predictive features using **RFM metrics** and behavior patterns.
* Train models to output:

  * ✅ A **risk probability score** (likelihood of default)
  * ✅ A **credit score** (on a human-friendly scale)
  * ✅ **Loan amount/duration** recommendations

---

## 📁 Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml        # ✅ CI/CD Pipeline (Task 6)
├── data/                           # Raw + processed data
├── notebooks/
│   └── 1.0-eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── train.py                    # Model training (Task 4 & 5)
│   ├── predict.py                  # Inference script (Task 5)
│   ├── data_processing.py          # Preprocessing utils
│   ├── feature_eng_process.py      # Full feature pipeline
│   └── api/                        # ✅ FastAPI Inference API (Task 6)
│       ├── main.py                 # FastAPI app
│       └── pydantic_models.py      # Input/output schemas
├── tests/
│   └── test_data_processing.py     # Unit tests (Task 5)
├── Dockerfile                      # ✅ Docker container config (Task 6)
├── docker-compose.yml              # ✅ Docker services incl. MLflow + API (Task 6)
├── requirements.txt                # Python deps
├── .gitignore                      # Ignore data/env
└── README.md                       # You’re here!
```

---

## 🔬 Data Processing & Feature Engineering

### 🔎 Exploratory Data Analysis

* \~95,000 transactions
* Handled missing values, outliers, date features
* Dropped constant/redundant features
* Saved result to `data/processed/data_cleaned.csv`

### 🧠 Feature Engineering (`feature_eng_process.py`)

* Aggregates customer-level behavior (total spend, tx count, avg tx)
* Extracts **hour, day, month** from timestamps
* Fills missing values (mode/median)
* Applies **WOE encoding** to categorical features (optbinning)
* Calculates **Information Value (IV)** for feature selection
* Scales numerical values (StandardScaler)
* Outputs: `data/processed/feature_engineered_data.csv`

To run:

```bash
python src/feature_eng_process.py
```

---

## 🤖 Model Training & MLflow Tracking

### 🏗️ Training & Evaluation (`src/train.py`)

* Trains **Logistic Regression** and **Random Forest**
* Uses:

  * Accuracy, Precision, Recall, F1, ROC-AUC
* Integrated with **MLflow**:

  * Logs metrics, artifacts
  * Registers model under `CreditRiskModel`
  * Adds tags for "stage", "promoted\_by"

### 🔁 Sample CLI:

```bash
python src/train.py
```

---

## 🧪 Unit Testing

Unit tests (`test_data_processing.py`) validate:

* Column dropping
* Target splitting
* Input-output formats

To run:

```bash
pytest tests/
```

---

## ⚙️ Inference (`src/predict.py`)

* Loads model from **MLflow Model Registry**
* Takes processed input and returns:

  * `is_high_risk`: True/False
  * `probability`: Risk score

---

## 🚀 Task 6 – Model Deployment with FastAPI + CI/CD

### ✅ FastAPI API (`src/api/main.py`)

* `/predict` endpoint
* Validates input via **Pydantic**
* Loads model from MLflow
* Returns credit risk prediction

### ✅ Pydantic Schemas (`src/api/pydantic_models.py`)

* `PredictionInput`: Defines required model features
* `PredictionOutput`: Defines response schema

### 🔧 Run API Locally

```bash
docker-compose up --build
```

Access docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🧪 GitHub CI/CD Pipeline (`.github/workflows/ci.yml`)

* Runs on push/pull to `main`
* Steps:

  * Install dependencies
  * Run `flake8` linting
  * Run `pytest`

```yaml
# Simplified
on: [push]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: flake8 src/
      - run: pytest tests/
```

---

## 🐳 Docker Setup

### Dockerfile

Builds image for model training and API inference.

### docker-compose.yml

* Starts **MLflow server**
* Starts **credit-risk-api** (FastAPI)
* Mounts volume for model artifacts

---

## 📊 Business Value

* Aligned with **Basel II** for explainable credit models
* Enables **responsible BNPL lending**
* Supports **regulatory transparency** via WOE encoding & scoring

---

## 🛠️ Next Steps

* Integrate with **real-time API scoring**
* Add **auto-retraining** + drift detection
* Use **SHAP/LIME** for black-box explainability
* Expand feature set using **alternative data sources**
* Add **loan recommendation engine**

---

## ✅ Quickstart

```bash
# 1. Clone the repo
git clonehttps://github.com/Dagiayy/Credit-Risk-Probability-Model.git.git
cd credit-risk-model

# 2. Install deps
pip install -r requirements.txt

# 3. Run model training
python src/train.py

# 4. Run API in Docker
docker-compose up --build

# 5. Access
MLflow:         http://localhost:5000
API docs:       http://localhost:8000/docs
```

---

## 📚 References

* [Basel II Accord – BIS](https://www.bis.org)
* [World Bank Credit Scoring](https://worldbank.org)
* [Interpretable Machine Learning (SHAP, WOE)](https://christophm.github.io/interpretable-ml-book/)

---

> 🙏 Thank you for reviewing this project! Contributions, feedback, and improvements are welcome.

---
