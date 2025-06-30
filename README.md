
# Credit Risk Probability Model

## Overview

This project is part of **Bati Bank's** initiative to enable a **Buy-Now-Pay-Later (BNPL)** credit offering for an eCommerce partner.
It aims to predict customer creditworthiness based on transaction behavior, enabling responsible lending decisions without relying on traditional credit histories.

---

## Project Goals

* Build a **proxy variable** to categorize customers into *high-risk* and *low-risk* based on transaction behavior.
* Engineer predictive features using RFM (Recency, Frequency, Monetary) metrics and behavioral patterns.
* Develop models that output:

  * A **risk probability score** indicating likelihood of default.
  * A **credit score** on a human-friendly scale.
  * Recommendations for **optimal loan amount and duration**.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml          # CI/CD pipeline configuration
├── data/                            # Data folder (ignored by Git)
│   ├── raw/                        # Raw data files
│   └── processed/                  # Processed data for modeling
├── notebooks/
│   └── 1.0-eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Feature engineering and data processing scripts
│   ├── feature_eng_process.py      # Feature engineering pipeline including WOE/IV encoding
│   ├── train.py                    # Model training logic (Task 4 & 5)
│   ├── predict.py                  # Inference logic (Task 5)
│   └── api/
│       ├── main.py                 # FastAPI app to serve the model
│       └── pydantic_models.py      # API schemas
├── tests/
│   └── test_data_processing.py    # Unit tests (Task 5)
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker compose for local dev/testing
├── requirements.txt                # Python dependencies
├── .gitignore                     # Ignored files (data, envs)
└── README.md                      # Project documentation
```

---

## Data Processing & Feature Engineering

### Exploratory Data Analysis (EDA)

* Performed on the dataset of \~95,000 transactions.
* Checked distributions, missing values, outliers, and correlations.
* Added outlier flags and dropped constant columns.
* Saved cleaned data to `data/processed/data_cleaned.csv`.

### Feature Engineering Highlights

* **Aggregation by CustomerId:** Generates customer-level summary features such as total spend, average transaction, and transaction count to capture spending behavior.
* **Datetime features:** Extract hour, day, month, and year from transaction timestamps to capture temporal patterns.
* **Missing value imputation:** Fills missing categorical values with the most common category; numeric missing values with the median.
* **Normalization:** Standardizes numeric features to zero mean and unit variance for better model performance.
* **Weight of Evidence (WOE) encoding:**
  Converts categorical variables into numeric features reflecting their relationship with fraud/default risk, calculated using the `optbinning` package. This improves interpretability and model quality over one-hot encoding.
* **Information Value (IV) calculation:**
  Measures the predictive power of each categorical feature to help with feature selection.

---

## Running Feature Engineering

The feature engineering pipeline is implemented in `src/feature_eng_process.py`.

**To run:**

```bash
python src/feature_eng_process.py
```

* Reads cleaned data.
* Applies aggregation, datetime extraction, imputation, WOE encoding, and numeric scaling.
* Outputs processed dataset with engineered features to `data/processed/feature_engineered_data.csv`.
* Prints Information Value (IV) for categorical features, e.g.:

  ```
  Feature: ProductCategory, IV: 1.0636
  Feature: ChannelId, IV: 1.2231
  Feature: ProviderId, IV: 3.3227
  ```

---

## Model Training & Experiment Tracking (Task 4 & 5)

### Model Training & Logging (Task 4)

* Implemented training scripts in `src/train.py` for Logistic Regression and Random Forest models.
* Models are trained on preprocessed features and evaluated using accuracy, precision, recall, F1 score, and ROC-AUC metrics.
* MLflow is integrated to track experiments, log parameters, metrics, and save models.
* Model registration and versioning are handled through MLflow Model Registry, enabling controlled deployment and model lifecycle management.

### Model Inference Script (Task 5)

* Created `src/predict.py` to load the registered model from MLflow Model Registry for prediction.
* The script demonstrates loading model version (or stage), preprocessing input features consistently, and outputting predictions.
* Added error handling for missing model versions or stage.
* Includes sample usage with processed feature CSV file.

### Unit Testing (Task 5)

* Added unit tests under `tests/test_data_processing.py` to validate data preprocessing functions.
* Tests ensure that feature selection, target separation, and handling of non-feature columns are consistent and reliable.
* Facilitates code robustness and maintainability.

---

## Business Context & Model Interpretability

This project complies with **Basel II** requirements for transparent and auditable credit risk models.
Using WOE encoding and logistic regression models enables:

* **Interpretability:** Regulators and business users can understand how features affect risk.
* **Performance:** Captures meaningful patterns in customer transaction behavior.
* **Responsible lending:** Helps Bati Bank extend BNPL credit safely.

---

## Next Steps & Future Work

* Integrate model inference with real-time APIs for automated credit decisions.
* Develop auto-retraining pipelines to keep models up-to-date.
* Incorporate alternative data sources for richer risk insights.
* Expand explainability tools (SHAP/LIME) for black-box models.
* Fine-tune thresholds for risk categories and loan terms.

---

## Prerequisites & Quickstart

* Python 3.12 or later
* Recommended: Docker & Docker Compose (for easy environment setup)
* Install dependencies:

```bash
pip install -r requirements.txt
```

* Run API locally:

```bash
docker-compose up --build
```

Access API documentation: `http://localhost:8000/docs`

---

## References

* [Basel II Accord Summary (BIS)](https://www.bis.org/publ/bcbs128.pdf)
* [Credit Scoring Approaches – World Bank](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
* [How to Build a Credit Scorecard](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
* [Corporate Finance Institute - Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
* [Explainable AI for Credit Models (Interpretable ML Book)](https://christophm.github.io/interpretable-ml-book/)

---

Thank you for reviewing this project!
Contributions and issues are welcome.

---

