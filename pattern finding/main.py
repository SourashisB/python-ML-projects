import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('bank_transactions_data_2.csv')

# Datetime conversion
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])

# Feature engineering
df['TimeSinceLastTransaction'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds()

# Drop non-feature columns
df_features = df.drop(['TransactionID', 'AccountID', 'TransactionDate', 'PreviousTransactionDate'], axis=1)

# Label encoding
categorical_cols = ['TransactionType', 'Location', 'DeviceID', 'IP Address', 'MerchantID', 'Channel', 'CustomerOccupation']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_features[col] = le.fit_transform(df_features[col].astype(str))
    le_dict[col] = le

# Fill missing values
df_features.fillna(df_features.median(numeric_only=True), inplace=True)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Train-test split
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_train)

# Predictions
train_preds = iso_forest.predict(X_train)
test_preds = iso_forest.predict(X_test)
test_scores = iso_forest.decision_function(X_test)

# Visualization 1: Anomaly score distribution
plt.figure(figsize=(10,6))
sns.histplot(test_scores, bins=50, kde=True)
plt.title('Distribution of Anomaly Scores on Test Data')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Number of anomalies
anomalies = np.sum(test_preds == -1)
total = len(test_preds)
print(f"Anomalies detected in test set: {anomalies} out of {total} ({anomalies/total*100:.2f}%)")

# Visualization 3: t-SNE (optional)
from sklearn.manifold import TSNE
X_test_2d = TSNE(n_components=2, random_state=42).fit_transform(X_test)
plt.figure(figsize=(10,6))
sns.scatterplot(x=X_test_2d[:,0], y=X_test_2d[:,1], hue=(test_preds==-1), palette={True:'red', False:'blue'}, alpha=0.5)
plt.title('t-SNE Projection of Test Data\nRed=Detected Anomalies, Blue=Normal')
plt.legend(['Anomaly','Normal'])
plt.show()