
# 🏦 Credit Risk Probability Model

This project is part of **Bati Bank's** initiative to enable a **Buy-Now-Pay-Later (BNPL)** credit offering for an eCommerce partner.  
It aims to predict customer creditworthiness based on behavioral data, enabling responsible lending decisions without relying on traditional credit histories.

---

## 📈 Project Goals

- Build a **proxy variable** to categorize customers into *high-risk* and *low-risk* based on their transaction behavior.
- Engineer predictive features from transaction data using RFM (Recency, Frequency, Monetary) metrics.
- Develop models to output:
  - A **risk probability score** indicating default likelihood.
  - A **credit score** on a human-friendly scale.
  - Recommendations for **optimal loan amount and duration**.

---

## 📂 Project Structure

```

credit-risk-model/
├── .github/workflows/ci.yml       # CI/CD pipeline configuration
├── data/                          # Data folder (ignored by Git)
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data for modeling
├── notebooks/
│   └── 1.0-eda.ipynb             # Exploratory Data Analysis (Task 2)
├── src/
│   ├── **init**.py
│   ├── data\_processing.py        # Feature engineering and data processing scripts
│   ├── train.py                  # Model training logic
│   ├── predict.py                # Inference logic
│   └── api/
│       ├── main.py               # FastAPI app to serve the model
│       └── pydantic\_models.py    # API schemas
├── tests/
│   └── test\_data\_processing.py  # Unit tests
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker compose for local dev/testing
├── requirements.txt              # Python dependencies
├── .gitignore                    # Ignored files (data, envs)
└── README.md                     # Project documentation

````

---

## 🧠 Credit Scoring Business Understanding

### Basel II and Regulatory Context

Bati Bank’s credit risk modeling follows the **Basel II Accord**, which requires transparent, auditable risk models that:

- Use quantifiable risk metrics.
- Are interpretable by regulators and stakeholders.
- Enable informed, explainable lending decisions.

---

### Proxy Variable for Default Risk

Since direct loan default data is unavailable, we define a **proxy variable** reflecting customer risk through transaction behavior patterns (e.g., purchase frequency, returns). This proxy allows supervised learning but must be carefully validated to avoid bias or regulatory issues.

---

### Model Interpretability vs. Performance

| Model Type             | Interpretability         | Predictive Power         | Regulatory Ease         |
|------------------------|-------------------------|-------------------------|------------------------|
| Logistic Regression + WoE | High (easy to explain)   | Moderate                | Easier                 |
| Gradient Boosting (GBM) | Low (black-box)          | High                    | Requires explainability tools (SHAP/LIME) |

Initial models favor interpretability with potential later enhancement.

---

## 🚀 Task 1: Setup & Initial Project Structure

- Established a **modular and maintainable** folder structure.
- Created scripts for **data loading, processing, training, inference**, and **API serving**.
- Configured **Docker** and **CI/CD pipelines** for reproducibility.

---

## 🔍 Task 2: Exploratory Data Analysis (EDA)

Performed extensive EDA using Jupyter Notebook (`notebooks/1.0-eda.ipynb`):

- Dataset shape: **95,662 transactions, 16 features**
- Data types and distributions assessed.
- No missing values detected.
- Significant outliers identified in `Amount`, `Value`, and `PricingStrategy`.
- Added **outlier flag features** to enhance model awareness of extreme transactions.
- Dropped constant column `CountryCode` due to zero variance.
- Correlation matrix and categorical distributions visualized inline.
- Saved cleaned data to `data/processed/data_cleaned.csv`.

---

### 🔧 Recommendations for Feature Engineering (Task 3)

- **Log-transform** skewed numeric variables (`Amount`, `Value`).
- Use **outlier flags** as model features or cap extreme values.
- Extract **time-based features** from `TransactionStartTime` (hour, day, month).
- Apply **one-hot encoding** for categorical variables (`ProductCategory`, `ChannelId`, `ProviderId`).
- Aggregate by `CustomerId` to calculate **RFM metrics** (Recency, Frequency, Monetary).
- Employ **Weight of Evidence (WoE)** and **Information Value (IV)** for feature ranking and selection.

---

## 🧪 Quickstart Guide

### Prerequisites

- Install [Docker](https://docs.docker.com/get-docker/)  
- Python 3.12 (if running locally outside Docker)

### Run the API with Docker Compose

```bash
docker-compose up --build
````

Access API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📚 References & Further Reading

* [Basel II Accord Summary](https://www.bis.org/publ/bcbs128.pdf)
* [Credit Scoring Approaches – World Bank](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
* [How to Build a Credit Scorecard (Towards Data Science)](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
* [CFI Credit Risk Guide](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
* [Explainable AI (SHAP/LIME) for Credit Models](https://christophm.github.io/interpretable-ml-book/)

---

## 🛠 Future Work

* Integrate real-time credit approvals with external APIs.
* Develop auto-retraining and monitoring pipelines.
* Enhance model explainability for compliance.
* Expand model scope with alternative data sources.

---

Thank you for reviewing this project! Feel free to contribute or raise issues.

```

---

