
# 🏦 Credit Risk Probability Model

This project is developed as part of Bati Bank’s initiative to enable a Buy-Now-Pay-Later (BNPL) credit offering for an eCommerce partnership. The objective is to predict the creditworthiness of customers using behavioral data, such as purchase frequency, monetary spend, and recency (RFM), in the absence of traditional credit history.

The system outputs:
- A **risk probability score**
- A **credit score (human-friendly scale)**
- An estimate of the **optimal loan amount and duration**



## 📂 Project Structure

```

credit-risk-model/
├── .github/workflows/ci.yml       # CI/CD pipeline
├── data/                          # Raw and processed data (not tracked by Git)
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb              # EDA and RFM exploration
├── src/
│   ├── **init**.py
│   ├── data\_processing.py         # Feature engineering logic
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Inference script
│   └── api/
│       ├── main.py                # FastAPI app for serving model
│       └── pydantic\_models.py     # Request/response schemas
├── tests/
│   └── test\_data\_processing.py    # Unit tests
├── Dockerfile                     # Docker config
├── docker-compose.yml             # Compose file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignore data and environments
└── README.md                      # Project documentation

````

---

## 🧠 Credit Scoring Business Understanding

### 1. **Basel II and the Need for Interpretability**

The **Basel II Accord** emphasizes three pillars: minimum capital requirements, supervisory review, and market discipline. In the context of credit risk, it mandates that financial institutions implement **internal risk models** that are **transparent**, **auditable**, and based on **quantifiable measures** of borrower risk.

As a result, our model must:
- Be **interpretable** enough for internal audit, regulatory bodies, and business stakeholders.
- Provide **documentation and traceability** for how credit scores are assigned.
- Avoid black-box predictions that can't explain **why** a customer was approved or denied.

This requirement favors models that balance predictive power with explainability.

---

### 2. **Why We Use a Proxy Variable (and the Risks)**

Since we **don’t have a direct “default” label** (e.g., loan repayment history), we must create a **proxy variable** to indicate whether a customer behaves like a “high-risk” or “low-risk” borrower. Examples include:
- Frequent late payments
- Excessive returns
- Abrupt purchase stoppage

**Why it’s necessary**:
- Supervised machine learning requires a target variable.
- Proxy variables allow us to train predictive models when true default data isn’t available.

**Business risks of relying on a proxy**:
- **Label noise**: Our proxy may misclassify customers, reducing model accuracy.
- **Bias**: If the proxy is based on flawed logic or limited data, it can introduce bias.
- **Regulatory challenge**: Regulators may scrutinize models built on proxies if the definition is unclear or unvalidated.

---

### 3. **Model Interpretability vs. Performance Trade-offs**

| Feature                            | Simple Model (e.g., Logistic Regression + WoE) | Complex Model (e.g., Gradient Boosting) |
|------------------------------------|-----------------------------------------------|------------------------------------------|
| Interpretability                   | High (easy to explain to regulators)          | Low (requires SHAP or LIME to explain)   |
| Regulatory approval                | Easier                                        | Harder due to black-box nature           |
| Training speed                     | Fast                                          | Slower                                   |
| Predictive power                   | Moderate                                      | High                                     |
| Auditability                       | Excellent                                     | Requires external tools                  |
| Flexibility with imbalanced data   | Requires preprocessing                        | Handles natively                         |

In **regulated contexts like banking**, we often start with interpretable models to pass compliance checks. Then, depending on business need and risk appetite, we may blend in more powerful models — but only if we include **explainability techniques** and **robust validation**.

---

## 🚀 Quickstart

### Run FastAPI with Docker:
```bash
docker-compose up --build
````

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) to test the API.

---

## 🧪 Future Enhancements

* Integration with real-time credit approval systems
* Auto-retraining pipelines with Airflow
* Explainability module (SHAP/LIME) for regulators
* Integration with payment gateway API

---

## 📚 References

* [Statistica Credit Scoring Research](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
* [HKMA Alternative Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
* [World Bank Credit Scoring Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
* [How to Build a Credit Scorecard](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
* [CFI Credit Risk Guide](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
* [Risk Officer Credit Risk Summary](https://www.risk-officer.com/Credit_Risk.htm)

---

