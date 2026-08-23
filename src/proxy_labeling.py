import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_rfm(df, snapshot_date_col='TransactionStartTime'):
    df[snapshot_date_col] = pd.to_datetime(df[snapshot_date_col])
    snapshot_date = df[snapshot_date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerId').agg({
        snapshot_date_col: lambda x: (snapshot_date - x.max()).days,
        'CustomerId': 'count',
        'Amount': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()

    return rfm

def label_high_risk_customers(rfm_df):
    # Standardize RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Identify the least engaged cluster (low Frequency + low Monetary)
    cluster_scores = rfm_df.groupby('Cluster')[['Frequency', 'Monetary']].mean().sum(axis=1)
    high_risk_cluster = cluster_scores.idxmin()

    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    return rfm_df[['CustomerId', 'is_high_risk']]

if __name__ == "__main__":
    # Load feature-engineered data
    df = pd.read_csv('data/processed/feature_engineered_data.csv')

    # Step 1: Calculate RFM per Customer
    rfm = calculate_rfm(df)

    # Step 2: Assign proxy high-risk labels using clustering
    risk_labels = label_high_risk_customers(rfm)

    # Step 3: Merge risk label into main dataset
    df_labeled = df.merge(risk_labels, on='CustomerId', how='left')

    # Step 4: Save result
    df_labeled.to_csv('data/processed/feature_engineered_labeled.csv', index=False)
    print("✅ Saved labeled dataset to 'data/processed/feature_engineered_labeled.csv'")
