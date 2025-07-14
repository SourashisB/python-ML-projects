import pandas as pd
import numpy as np

# 1. Load the Excel file
df = pd.read_excel('Cleaned_Employee_Data.xlsx')

# 2. Specify the column types
string_cols = [
    'Attrition', 'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime'
]

# Columns that are obviously numeric (int/float)
numeric_cols = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
    'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

# 3. Clean and convert columns

# Strip strings and convert to string type
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Convert numeric columns to integers (if possible), otherwise floats
for col in numeric_cols:
    if col in df.columns:
        # If there are NaNs, convert to float, else int
        if df[col].isnull().any():
            df[col] = pd.to_numeric(df[col], errors='coerce')  # will be float if NaNs present
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)

# 4. Handle missing values (optional, depends on your needs)
# For now, let's just keep as NaN. You can fill/drop as needed:
# df = df.dropna()  # or
# df = df.fillna(0)  # or
# df = df.fillna('Unknown')  # for string columns

# 5. Save to CSV
df.to_csv('cleaned_data.csv', index=False)

print("Data cleaned and saved as 'cleaned_data.csv'.")