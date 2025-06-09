import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load the data
df = pd.read_csv('personality_dataset.csv')

# 2. Encode 'Stage_fear' and 'Drained_after_socializing'
df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

# 3. Encode the target variable (Personality)
label_encoder = LabelEncoder()
df['Personality'] = label_encoder.fit_transform(df['Personality'])
# Introvert = 0, Extrovert = 1

# 4. Separate features and target
X = df.drop('Personality', axis=1)
y = df['Personality']

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split into 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 8. Predict and evaluate on validation set
y_pred = clf.predict(X_val)
print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_val, y_pred))

# 9. Visualization: Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 10. Visualization: Feature Importances
feature_importances = clf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()