import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load and preprocess data
df = pd.read_csv('your_data.csv')

# Ensure all categorical columns are strings
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize selected features
features_to_normalize = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[features_to_normalize] = scaler.fit_transform(X_train[features_to_normalize])

# Train logistic regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(logreg, 'logreg_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X_train.columns.tolist(), 'model_features.joblib')

print("Model, scaler, and feature list saved. Ready for deployment.")