import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Data
def load_data(path="data/Data/data.csv"):
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Overview
def overview(df):
    print("🔎 Data Overview")
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nFirst 5 Rows:\n", df.head())

# Summary
def summary_statistics(df):
    print("📊 Summary Statistics")
    return df.describe()

# Numerical Distribution
def plot_numerical_distributions(df, output_folder="notebooks/plots/"):
    os.makedirs(output_folder, exist_ok=True)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=50)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{output_folder}{col}_hist.png")
        plt.close()

# Categorical Distribution
def plot_categorical_distributions(df, output_folder="notebooks/plots/"):
    os.makedirs(output_folder, exist_ok=True)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        df[col].value_counts().nlargest(10).plot(kind='bar')
        plt.title(f'Top 10 Categories in {col}')
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_folder}{col}_bar.png")
        plt.close()

# Correlation Matrix
def correlation_matrix(df, output_file="notebooks/plots/correlation_matrix.png"):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Missing Values
def missing_values(df):
    print("🕳 Missing Values:")
    missing = df.isnull().sum()
    return missing[missing > 0]

# Outlier Detection
def detect_outliers(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    outlier_data = {}

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        outlier_data[col] = len(outliers)

    return outlier_data

def plot_boxplots(df, output_folder="notebooks/plots/boxplots/"):
    os.makedirs(output_folder, exist_ok=True)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.savefig(f"{output_folder}{col}_boxplot.png")
        plt.close()

# Save
def save_processed_data(df, path="data/processed/data_cleaned.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved cleaned data to: {path}")
