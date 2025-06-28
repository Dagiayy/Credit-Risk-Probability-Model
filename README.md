# 🧠 Task 2 – Exploratory Data Analysis (EDA)

This task involves performing an in-depth exploration of the dataset provided by the eCommerce platform to uncover patterns, assess data quality, and derive insights that will guide feature engineering and model development for the credit scoring model.

---

## 📁 File Location

- 📄 Raw dataset: `data/raw/data.csv`
- 📔 Notebook: `notebooks/1.0-eda.ipynb`
- 🧠 Source functions: `src/data_processing.py`
- 💾 Processed output: `data/processed/data_cleaned.csv`

---

## 🎯 Objectives

- Understand the structure and contents of the dataset
- Visualize distributions of numerical and categorical features
- Analyze correlations between variables
- Identify and handle missing values and outliers
- Extract actionable insights for feature engineering

---

## 🛠 Steps Performed

1. **Data Overview**
   - Checked shape, column types, and sample records
   - Found 95,662 transactions across 16 features
   - All data types and encodings verified

2. **Summary Statistics**
   - Computed mean, median, std, and percentiles for numeric features
   - Identified skewness and scale differences (especially in `Amount`, `Value`)

3. **Visual Analysis**
   - Plotted histograms of all numeric features
   - Visualized categorical feature distributions with bar plots
   - Generated correlation matrix heatmap
   - Created box plots to observe outliers

4. **Missing Values**
   - Verified no missing values in the dataset

5. **Outlier Detection**
   - Detected significant outliers in `Amount`, `Value`, and `PricingStrategy`

---

## 🔍 Key Insights

| Insight | Description |
|--------|-------------|
| 📉 `FraudResult` is highly imbalanced | Only ~0.2% of samples are fraud (1) — needs special treatment in modeling. |
| 💰 `Amount` and `Value` have extreme outliers and skewed distribution | Suggests log transformation or robust modeling. |
| 🌍 `CountryCode` is constant (`256`) | Can be safely dropped (no variance). |
| 🔄 Behavioral features are needed | Frequency, recency, and spend patterns likely more predictive than raw transaction values. |

---

## 📦 Output

- ✅ Cleaned and structured dataset saved to:  
  `data/processed/data_cleaned.csv`

- ✅ All plots saved and also visualized inline in the notebook.

---

## 🧠 Next Steps

- Engineer customer-level behavioral features (e.g., RFM)
- Encode categorical features
- Prepare training-ready dataset for modeling
