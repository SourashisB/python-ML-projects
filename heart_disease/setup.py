import pandas as pd

# Load your dataset (replace with your actual data source)
df = pd.read_csv('heart.csv')

# Convert all object columns to string
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)

# Display data types after conversion
print("Data types after conversion:\n", df.dtypes)

# Correlation matrix (numeric only)
corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation matrix (numeric only):\n", corr_matrix)

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)
corr_matrix_full = df_encoded.corr()
print("\nCorrelation matrix (with encoded categorical features):\n", corr_matrix_full)

# Identify features to normalize (continuous numeric)
features_to_normalize = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
print("\nFeatures suitable for normalization:", features_to_normalize)