import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# 1. LOAD DATA
df = pd.read_csv('heart.csv')

# 2. ENCODE CATEGORICAL VARIABLES
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)  # Ensure all are strings

df_encoded = pd.get_dummies(df, drop_first=True)

# 3. SPLIT FEATURES AND TARGET
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. NORMALIZE CONTINUOUS FEATURES FOR KNN AND LOGISTIC REGRESSION
features_to_normalize = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[features_to_normalize] = scaler.fit_transform(X_train[features_to_normalize])
X_test_scaled[features_to_normalize] = scaler.transform(X_test[features_to_normalize])

# 6. K-NEAREST NEIGHBOURS
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Classifier Accuracy: {accuracy_knn:.4f}")

# 7. XGBOOST
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Classifier Accuracy: {accuracy_xgb:.4f}")

# 8. RANDOM FOREST (tree-based)
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Classifier Accuracy: {accuracy_rf:.4f}")

# 9. LOGISTIC REGRESSION (non-tree-based)
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg:.4f}")

# 10. SUMMARY
print("\nComparison of Classifier Accuracies:")
print(f"K-Nearest Neighbors:  {accuracy_knn:.4f}")
print(f"XGBoost:              {accuracy_xgb:.4f}")
print(f"Random Forest:        {accuracy_rf:.4f}")
print(f"Logistic Regression:  {accuracy_logreg:.4f}")

# Optional: Find the best model
accuracies = {
    "K-Nearest Neighbors": accuracy_knn,
    "XGBoost": accuracy_xgb,
    "Random Forest": accuracy_rf,
    "Logistic Regression": accuracy_logreg
}
best_model = max(accuracies, key=accuracies.get)
print(f"\nBest performing classifier: {best_model} (Accuracy: {accuracies[best_model]:.4f})")