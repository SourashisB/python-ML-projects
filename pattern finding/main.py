# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('bank_transactions_data_2.csv')

# Step 1: Initial Exploration
print("Dataset Overview:")
print(data.info())
print("\nDataset Statistics:")
print(data.describe())

# Step 2: Data Preprocessing
# Fill missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical columns
categorical_columns = ['TransactionType', 'Location', 'DeviceID', 'IP Address', 'MerchantID', 'Channel', 'CustomerOccupation']
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Feature scaling for numerical columns
scaler = StandardScaler()
numerical_columns = [
    'TransactionAmount', 'CustomerAge', 'TransactionDuration', 
    'LoginAttempts', 'AccountBalance'
]
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 3: Feature Engineering
# Create new features
data['TimeSinceLastTransaction'] = (
    pd.to_datetime(data['TransactionDate']) - pd.to_datetime(data['PreviousTransactionDate'])
).dt.total_seconds()
data['TimeSinceLastTransaction'].fillna(0, inplace=True)
data['TimeSinceLastTransaction'] = scaler.fit_transform(data[['TimeSinceLastTransaction']])

# Transaction frequency and average transaction amount
data['TransactionFrequency'] = data.groupby('AccountID')['TransactionAmount'].transform('count')
data['AvgTransactionAmount'] = data.groupby('AccountID')['TransactionAmount'].transform('mean')
data['TransactionFrequency'] = scaler.fit_transform(data[['TransactionFrequency']])
data['AvgTransactionAmount'] = scaler.fit_transform(data[['AvgTransactionAmount']])

# Drop unnecessary columns
data.drop(columns=['TransactionID', 'AccountID', 'TransactionDate', 'PreviousTransactionDate'], inplace=True)

# Step 4: Visual Exploration
# Pairplot to visualize relationships
sns.pairplot(data.sample(500), diag_kind='kde', corner=True)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 5: Anomaly Detection with Isolation Forest
print("\nRefining Anomaly Detection with Isolation Forest...")

# Tune Isolation Forest parameters
iso_forest = IsolationForest(n_estimators=200, contamination=0.05, max_samples='auto', random_state=42)
data['AnomalyScore'] = iso_forest.fit_predict(data)

# Add a binary anomaly flag
data['Anomaly'] = data['AnomalyScore'].apply(lambda x: 1 if x == -1 else 0)

# Display updated anomaly statistics
print(f"Number of anomalies detected: {data['Anomaly'].sum()}")
print(f"Percentage of anomalies: {data['Anomaly'].mean() * 100:.2f}%")

# Step 6: Clustering (DBSCAN and KMeans)
# Dimensionality reduction with PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data.drop(columns=['AnomalyScore', 'Anomaly']))

# DBSCAN clustering
print("\nExploring Clustering with DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=10, metric='euclidean')
data['DBSCAN_Cluster'] = dbscan.fit_predict(data_pca)

# KMeans clustering
print("\nExploring Clustering with KMeans...")
kmeans = KMeans(n_clusters=5, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(data_pca)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['KMeans_Cluster'], palette='viridis', alpha=0.6)
plt.title("KMeans Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", loc="upper right")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['DBSCAN_Cluster'], palette='viridis', alpha=0.6)
plt.title("DBSCAN Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", loc="upper right")
plt.show()

# Evaluate clustering
dbscan_clusters = len(set(data['DBSCAN_Cluster'])) - (1 if -1 in data['DBSCAN_Cluster'] else 0)
print(f"DBSCAN found {dbscan_clusters} clusters (excluding noise).")

kmeans_silhouette = silhouette_score(data_pca, data['KMeans_Cluster'])
print(f"KMeans Silhouette Score: {kmeans_silhouette:.2f}")

# Step 7: Save Results
output_file = "refined_data_with_anomalies.csv"
data.to_csv(output_file, index=False)
print(f"\nRefined data with anomalies and clustering results saved to: {output_file}")

# Step 8: Enhanced Visualization (Optional)
# Visualize anomalies on a pairplot with clusters
sns.pairplot(
    data.sample(500),
    diag_kind='kde',
    corner=True,
    hue='Anomaly',
    palette={0: 'blue', 1: 'red'}
)
plt.title("Anomaly Detection Pairplot")
plt.show()