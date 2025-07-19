import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
df = pd.read_csv('cleaned_data.csv')

# 2. CHOOSE FEATURES (update as needed)
features_to_use = [
    'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education',
    'Gender', 'JobRole', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
    'OverTime', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole'
]

# 3. SPLIT BY ATTRITION
df_yes = df[df['Attrition'] == 'Yes'].copy()
df_no = df[df['Attrition'] == 'No'].copy()

# 4. IDENTIFY NUMERICAL FEATURES (for normalization)
numerical_features = [
    'Age', 'DistanceFromHome', 'Education', 'MonthlyIncome', 
    'NumCompaniesWorked', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole'
]

# --- Check normality and normalize only Gaussian-like features ---
# We'll just assume 'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany' are roughly Gaussian for this example
gaussian_features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']

# 5. PREPROCESS FUNCTION

def preprocess(df_part):
    # Separate features
    num = df_part[gaussian_features]
    cat = df_part[list(set(features_to_use) - set(gaussian_features))]
    
    # Standardize numerical (Gaussian) features
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num)
    num_scaled_df = pd.DataFrame(num_scaled, columns=gaussian_features, index=df_part.index)
    
    # One-hot encode categorical features
    cat_cols = cat.select_dtypes(include=['object', 'category']).columns.tolist()
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # <-- changed here
    cat_encoded = enc.fit_transform(cat[cat_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=enc.get_feature_names_out(cat_cols), index=df_part.index)
    
    # Combine
    processed = pd.concat([num_scaled_df, cat_encoded_df], axis=1)
    return processed

# 6. APPLY PREPROCESSING
X_yes = preprocess(df_yes)
X_no = preprocess(df_no)

# 7. CHOOSE NUMBER OF CLUSTERS (k)
# You can use the elbow method, but we'll just use k=3 for this example
k = 3

# 8. CLUSTERING
kmeans_yes = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters_yes = kmeans_yes.fit_predict(X_yes)
df_yes['cluster'] = clusters_yes

kmeans_no = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters_no = kmeans_no.fit_predict(X_no)
df_no['cluster'] = clusters_no

# 9. PATTERN ANALYSIS FUNCTION
def print_cluster_patterns(df, cluster_col, original_df, group_label):
    print(f"\n--- Patterns in Attrition='{group_label}' employees ---")
    for c in sorted(df[cluster_col].unique()):
        group = df[df[cluster_col] == c]
        print(f"\nCluster {c}: n={len(group)}")
        # Show mean for numerical, mode for categorical
        print("Numerical (mean):")
        print(group[gaussian_features].mean())
        print("Categorical (mode):")
        cat_feats = list(set(features_to_use) - set(gaussian_features))
        for feat in cat_feats:
            mode_val = group[feat].mode()
            if not mode_val.empty:
                print(f"  {feat}: {mode_val.iloc[0]}")
        # Optional: Show most frequent JobRole
        print(f"  Most common JobRole: {group['JobRole'].mode().iloc[0]}")
    print("\n")

# 10. SHOW PATTERNS
print_cluster_patterns(df_yes, 'cluster', df, group_label='Yes')
print_cluster_patterns(df_no, 'cluster', df, group_label='No')

# 11. (Optional) Visualize cluster sizes
plt.figure(figsize=(7,4))
sns.countplot(x='cluster', data=df_yes)
plt.title("Cluster sizes for Attrition='Yes'")
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(x='cluster', data=df_no)
plt.title("Cluster sizes for Attrition='No'")
plt.show()