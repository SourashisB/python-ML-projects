import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# === CONFIGURATION ===
DATA_PATH = 'heart.csv'
FEATURES_TO_NORMALIZE = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
CATEGORICAL_FEATURES = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
TARGET = 'HeartDisease'

# === 1. LOAD AND PREPROCESS DATA ===
def load_and_prepare_data(path):
    df = pd.read_csv(path)
    # Ensure all categorical columns are strings
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str)
    return df

df = load_and_prepare_data(DATA_PATH)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df_encoded.drop(TARGET, axis=1)
y = df_encoded[TARGET]

# === 2. TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 3. NORMALIZE CONTINUOUS FEATURES ===
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[FEATURES_TO_NORMALIZE] = scaler.fit_transform(X_train[FEATURES_TO_NORMALIZE])

# === 4. TRAIN MODEL ===
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# === 5. SAVE MODEL, SCALER, FEATURE LIST ===
joblib.dump(logreg, 'logreg_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X_train.columns.tolist(), 'model_features.joblib')

# === 6. SAVE DEFAULT VALUES FOR MISSING DATA (for API use) ===
defaults = {}
# Use original (non-encoded) training data for computing defaults
df_train = df.iloc[X_train.index]

# For continuous features: use mean
for col in FEATURES_TO_NORMALIZE:
    defaults[col] = float(df_train[col].mean())
# For categorical features: use mode
for col in CATEGORICAL_FEATURES:
    defaults[col] = df_train[col].mode()[0]
joblib.dump(defaults, 'defaults.joblib')

print("Model, scaler, feature list, and defaults saved. Ready for deployment.")