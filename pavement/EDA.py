# eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Settings for better visualization
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)

# 1. Load Data
df = pd.read_csv('Pavement_Dataset.csv')

# 2. Basic Information
print("----- Dataset Shape -----")
print(df.shape)

print("\n----- First 5 Rows -----")
print(df.head())

print("\n----- Data Types -----")
print(df.dtypes)

print("\n----- Missing Values -----")
print(df.isnull().sum())

print("\n----- Summary Statistics (Numeric) -----")
print(df.describe())

print("\n----- Summary Statistics (Categorical) -----")
cat_cols = df.select_dtypes(include='object').columns
print(df[cat_cols].describe())

# 3. Target Distribution
print("\n----- Target Value Counts -----")
print(df['Needs Maintenance'].value_counts())

plt.figure()
sns.countplot(data=df, x='Needs Maintenance')
plt.title('Needs Maintenance Distribution')
plt.show()

# 4. Correlation Matrix
plt.figure()
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 5. Feature Distributions
num_cols = df.select_dtypes(include=[np.number]).columns.drop('Needs Maintenance')
for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

# 6. Boxplots by Target
for col in num_cols:
    plt.figure()
    sns.boxplot(data=df, x='Needs Maintenance', y=col)
    plt.title(f'{col} by Needs Maintenance')
    plt.show()

# 7. Categorical Feature Analysis
for col in cat_cols:
    plt.figure()
    sns.countplot(data=df, x=col, hue='Needs Maintenance')
    plt.title(f'{col} vs Needs Maintenance')
    plt.xticks(rotation=45)
    plt.show()

# 8. Outlier Detection (IQR Method for Numeric Columns)
print("\n----- Outlier Detection -----")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"{col}: {len(outliers)} outliers")

# 9. Pairplot for Numeric Features (sample if too large)
sample_df = df.sample(n=min(500, len(df)), random_state=42)
sns.pairplot(sample_df, hue='Needs Maintenance', diag_kind='kde')
plt.show()

# 10. Correlation with Target
print("\n----- Correlation with Target -----")
for col in num_cols:
    corr_with_target = df[col].corr(df['Needs Maintenance'])
    print(f"{col}: {corr_with_target:.3f}")

# 11. Missing Value Heatmap
plt.figure()
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title('Missing Data Heatmap')
plt.show()

# Done
print("\nEDA Complete.")