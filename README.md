# 🧠 Task 2 – Exploratory Data Analysis (EDA)

This task focuses on conducting a thorough exploratory data analysis of the transactional dataset provided by the eCommerce platform. The goal is to understand the underlying data characteristics, identify data quality issues, detect patterns and anomalies, and extract actionable insights that will inform subsequent feature engineering and model development phases for the credit scoring model.



## 📁 Project Structure & Files Involved

- **Raw data:** `data/raw/data.csv`  
  The original dataset as provided, containing all transaction records.

- **Exploratory Notebook:** `notebooks/1.0-eda.ipynb`  
  Jupyter notebook containing all exploratory data analysis steps, visualizations, and observations.

- **Source Code:** `src/data_processing.py`  
  Contains reusable functions for loading data, summarizing statistics, visualizing distributions, detecting outliers, and saving processed data.

- **Processed data:** `data/processed/data_cleaned.csv`  
  The cleaned and enriched dataset after handling missing values, outliers, and feature engineering steps such as adding outlier flags.

---

## 🎯 Objectives

- **Data Understanding:**  
  Investigate the dataset structure — number of rows and columns, data types, and sample records.

- **Summary Statistics:**  
  Calculate key statistics (mean, median, standard deviation, min/max, quartiles) for numerical features to understand data distribution and variability.

- **Visualization of Distributions:**  
  - Plot histograms for numerical variables to observe patterns, skewness, and outliers.  
  - Create bar plots for categorical variables to inspect category frequency and diversity.

- **Correlation Analysis:**  
  Compute and visualize correlation coefficients between numerical variables to detect relationships and multicollinearity.

- **Missing Value Identification:**  
  Check for any missing or null values across features to plan imputation strategies or data cleaning.

- **Outlier Detection and Handling:**  
  Use Interquartile Range (IQR) method to identify extreme values in numerical features.  
  Introduce new binary flag features indicating outlier presence per row to help downstream models explicitly capture anomalous data points.

- **Data Cleaning:**  
  Drop constant or non-informative columns such as `CountryCode` which has no variance.

- **Data Saving:**  
  Save the cleaned and enriched dataset for use in feature engineering and model training.

---

## 🔍 Findings & Insights

### 1. Data Overview

- The dataset contains **95,662 transaction records** with **16 features** including IDs, numeric values, categorical labels, timestamps, and the target fraud indicator.
- Data types are a mix of `object` (strings/IDs), `float64`, and `int64`.

### 2. Summary Statistics

- Numerical features like `Amount` and `Value` have large scale and are **heavily skewed** due to a few extremely large transactions.
- The target feature `FraudResult` is **highly imbalanced** with only about 0.2% positive fraud cases, indicating the need for careful model treatment.
- `CountryCode` is constant (all `256`), providing no predictive value and safely dropped.

### 3. Missing Values

- No missing values detected across any columns, simplifying data preprocessing.

### 4. Outlier Detection

- Significant outliers found in `Amount`, `Value`, and `PricingStrategy`.
- Added **outlier flag columns** (`*_outlier_flag`) for these features using the IQR method to mark extreme values.
- These flags are expected to enhance model interpretability and predictive performance by highlighting anomalous transactions.

### 5. Data Cleaning

- `CountryCode` column dropped due to zero variance.

---

## 📈 Visualizations

- Histograms and boxplots of numerical features reveal skewness and outliers.
- Bar charts of categorical variables show distribution imbalance in channels, product categories, and providers.
- Correlation heatmap indicates weak to moderate correlations, suggesting feature engineering is crucial.

---

## 💡 Next Steps

- **Feature Engineering:**  
  Create customer-level behavioral features such as Recency, Frequency, Monetary (RFM) values, and time-based transaction features from `TransactionStartTime`.

- **Handling Imbalance:**  
  Plan for strategies such as oversampling, undersampling, or class-weighted loss functions to handle fraud class imbalance.

- **Model Development:**  
  Use the cleaned dataset with engineered features and outlier flags to train credit risk and fraud detection models.

---

## 🛠 How to Run EDA Notebook

1. Ensure dependencies listed in `requirements.txt` are installed.
2. Place the raw data in `data/raw/data.csv`.
3. Run the notebook `notebooks/1.0-eda.ipynb`.
4. View inline plots or toggle `FAST_MODE` to speed up analysis without plotting.
5. Review printed summary stats, missing value report, and outlier counts.
6. Check the output cleaned data in `data/processed/data_cleaned.csv`.

---


