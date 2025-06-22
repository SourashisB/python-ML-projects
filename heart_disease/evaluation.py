import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# 1. LOAD DATA
df = pd.read_csv('heart.csv')

# 2. EDA: Feature Distribution
num_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
plt.figure(figsize=(12, 8))
for idx, col in enumerate(num_features):
    plt.subplot(2, 3, idx+1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# 3. ENCODE CATEGORICAL VARIABLES
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)

df_encoded = pd.get_dummies(df, drop_first=True)

# 4. SPLIT DATA
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Show train-test split sizes
plt.figure(figsize=(5, 3))
sns.barplot(x=['Train', 'Test'], y=[len(y_train), len(y_test)], palette='Blues')
plt.ylabel('Samples')
plt.title('Train-Test Split Sizes')
plt.savefig('train_test_split.png')
plt.close()

# 5. FEATURE SCALING VISUALIZATION
features_to_normalize = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[features_to_normalize] = scaler.fit_transform(X_train[features_to_normalize])
X_test_scaled[features_to_normalize] = scaler.transform(X_test[features_to_normalize])

# Visualize before/after scaling for Age
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(X_train['Age'], kde=True, bins=20, color='purple')
plt.title('Age (Original)')
plt.subplot(1, 2, 2)
sns.histplot(X_train_scaled['Age'], kde=True, bins=20, color='green')
plt.title('Age (Scaled)')
plt.tight_layout()
plt.savefig('feature_scaling.png')
plt.close()

# 6. MODEL TRAINING & EVALUATION

results = {}
conf_matrices = {}

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrices['KNN'] = confusion_matrix(y_test, y_pred_knn)
results['KNN'] = accuracy_knn

# XGBoost
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
conf_matrices['XGBoost'] = confusion_matrix(y_test, y_pred_xgb)
results['XGBoost'] = accuracy_xgb

# Random Forest
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrices['Random Forest'] = confusion_matrix(y_test, y_pred_rf)
results['Random Forest'] = accuracy_rf

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
conf_matrices['Logistic Regression'] = confusion_matrix(y_test, y_pred_logreg)
results['Logistic Regression'] = accuracy_logreg

# 7. ACCURACY COMPARISON BARPLOT
plt.figure(figsize=(7, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('model_accuracy_comparison.png')
plt.close()

# 8. CONFUSION MATRICES PLOT
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
model_list = list(conf_matrices.keys())
for i, ax in enumerate(axes.flat):
    cm = conf_matrices[model_list[i]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    ax.set_title(f"{model_list[i]} Confusion Matrix")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('all_confusion_matrices.png')
plt.close()

# 9. PRINT BULLET POINT SUMMARY
print("\n" + "="*40)
print("📊 **Model Development and Evaluation Process Summary:**")
print("="*40)
print("""
- Loaded the heart disease dataset and explored the distribution of key features.
- Applied one-hot encoding to categorical variables.
- Split the data into 80% training and 20% testing sets, visualized the split.
- Applied standard scaling (z-score normalization) to continuous features for KNN and Logistic Regression.
- Trained and evaluated four classification models:
    - K-Nearest Neighbors (KNN) [non-tree]
    - XGBoost Classifier [tree]
    - Random Forest Classifier [tree]
    - Logistic Regression [non-tree]
- Compared model accuracies via a bar plot.
- Visualized confusion matrices for all classifiers to assess prediction quality.
- XGBoost and Random Forest are tree-based models; KNN and Logistic Regression are non-tree-based.
- The process and results are visualized in the generated PNG files for inclusion in your report.
""")
print("="*40)
print("FILES SAVED FOR REPORT:")
print("- feature_distributions.png (Feature histograms)")
print("- train_test_split.png (Train/test sizes)")
print("- feature_scaling.png (Effect of scaling on 'Age')")
print("- model_accuracy_comparison.png (Model accuracy barplot)")
print("- all_confusion_matrices.png (All confusion matrices)")

# Optionally, print results
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} (Accuracy: {results[best_model]:.4f})")