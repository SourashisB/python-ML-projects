import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# 1. AGE
print("\n=== Age ===")
print(df['Age'].describe())
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. ATTRITION
print("\n=== Attrition (Value Counts) ===")
print(df['Attrition'].value_counts())
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='Attrition', palette='Set2')
plt.title('Attrition Count')
plt.tight_layout()
plt.show()

# 3. MONTHLY INCOME
print("\n=== MonthlyIncome ===")
print(df['MonthlyIncome'].describe())
plt.figure(figsize=(6,4))
sns.histplot(df['MonthlyIncome'], kde=True, bins=20, color='orange')
plt.title('Distribution of Monthly Income')
plt.xlabel('Monthly Income')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 4. JOB ROLE
print("\n=== JobRole (Value Counts) ===")
print(df['JobRole'].value_counts())
plt.figure(figsize=(10,5))
sns.countplot(data=df, y='JobRole', order=df['JobRole'].value_counts().index, palette='viridis')
plt.title('Count of Employees by Job Role')
plt.tight_layout()
plt.show()

# 5. YEARS AT COMPANY
print("\n=== YearsAtCompany ===")
print(df['YearsAtCompany'].describe())
plt.figure(figsize=(6,4))
sns.histplot(df['YearsAtCompany'], kde=True, bins=20, color='green')
plt.title('Distribution of Years at Company')
plt.xlabel('Years at Company')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 6. OVERTIME
print("\n=== OverTime (Value Counts) ===")
print(df['OverTime'].value_counts())
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='OverTime', palette='Set1')
plt.title('OverTime Count')
plt.tight_layout()
plt.show()

# --- Relationships ---

# Age vs. Attrition
plt.figure(figsize=(6,4))
sns.boxplot(x='Attrition', y='Age', data=df, palette='Set3')
plt.title('Age vs. Attrition')
plt.tight_layout()
plt.show()

# Monthly Income by JobRole
plt.figure(figsize=(10,5))
sns.boxplot(x='JobRole', y='MonthlyIncome', data=df, palette='coolwarm')
plt.xticks(rotation=45)
plt.title('Monthly Income by Job Role')
plt.tight_layout()
plt.show()

# Years at Company vs. OverTime
plt.figure(figsize=(6,4))
sns.boxplot(x='OverTime', y='YearsAtCompany', data=df, palette='Set2')
plt.title('Years at Company by OverTime')
plt.tight_layout()
plt.show()