import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('coffee_shop_revenue.csv')

# Exploratory Data Analysis
print("Data Summary:")
print(df.describe())

# Correlation analysis
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Coffee Shop Data')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Multi-objective optimization function
def calculate_composite_score(customers, revenue, marketing):
    # Normalize the values to 0-1 range for each metric
    customers_norm = (customers - customers.min()) / (customers.max() - customers.min())
    revenue_norm = (revenue - revenue.min()) / (revenue.max() - revenue.min())
    # Invert marketing so lower is better
    marketing_norm = 1 - ((marketing - marketing.min()) / (marketing.max() - marketing.min()))
    
    # Create a composite score - equal weighting for simplicity
    composite_score = (customers_norm + revenue_norm + marketing_norm) / 3
    return composite_score

# Add composite score to dataframe
df['Composite_Score'] = calculate_composite_score(
    df['Number_of_Customers_Per_Day'], 
    df['Daily_Revenue'], 
    df['Marketing_Spend_Per_Day']
)

# Feature Engineering
# Create efficiency metrics
df['Revenue_per_Customer'] = df['Daily_Revenue'] / df['Number_of_Customers_Per_Day']
df['Revenue_per_Hour'] = df['Daily_Revenue'] / df['Operating_Hours_Per_Day']
df['Customers_per_Hour'] = df['Number_of_Customers_Per_Day'] / df['Operating_Hours_Per_Day']
df['Marketing_Efficiency'] = df['Number_of_Customers_Per_Day'] / df['Marketing_Spend_Per_Day']
df['Employee_Efficiency'] = df['Number_of_Customers_Per_Day'] / df['Number_of_Employees']

# Prepare data for modeling
# 1. Model for Number_of_Customers_Per_Day
X_customers = df.drop(['Number_of_Customers_Per_Day', 'Daily_Revenue', 'Composite_Score', 
                       'Revenue_per_Customer', 'Revenue_per_Hour', 'Customers_per_Hour',
                       'Marketing_Efficiency', 'Employee_Efficiency'], axis=1)
y_customers = df['Number_of_Customers_Per_Day']

# 2. Model for Daily_Revenue
X_revenue = df.drop(['Daily_Revenue', 'Composite_Score', 
                     'Revenue_per_Customer', 'Revenue_per_Hour', 'Customers_per_Hour',
                     'Marketing_Efficiency', 'Employee_Efficiency'], axis=1)
y_revenue = df['Daily_Revenue']

# 3. Model for Composite Score (multi-objective)
X_composite = df.drop(['Composite_Score', 'Revenue_per_Customer', 'Revenue_per_Hour', 
                       'Customers_per_Hour', 'Marketing_Efficiency', 'Employee_Efficiency'], axis=1)
y_composite = df['Composite_Score']

# Split the data
X_train_customers, X_test_customers, y_train_customers, y_test_customers = train_test_split(
    X_customers, y_customers, test_size=0.2, random_state=42)

X_train_revenue, X_test_revenue, y_train_revenue, y_test_revenue = train_test_split(
    X_revenue, y_revenue, test_size=0.2, random_state=42)

X_train_composite, X_test_composite, y_train_composite, y_test_composite = train_test_split(
    X_composite, y_composite, test_size=0.2, random_state=42)

# Scale the features
scaler_customers = StandardScaler()
X_train_customers_scaled = scaler_customers.fit_transform(X_train_customers)
X_test_customers_scaled = scaler_customers.transform(X_test_customers)

scaler_revenue = StandardScaler()
X_train_revenue_scaled = scaler_revenue.fit_transform(X_train_revenue)
X_test_revenue_scaled = scaler_revenue.transform(X_test_revenue)

scaler_composite = StandardScaler()
X_train_composite_scaled = scaler_composite.fit_transform(X_train_composite)
X_test_composite_scaled = scaler_composite.transform(X_test_composite)

# Train Random Forest models
# 1. For customers
rf_customers = RandomForestRegressor(n_estimators=100, random_state=42)
rf_customers.fit(X_train_customers_scaled, y_train_customers)

# 2. For revenue
rf_revenue = RandomForestRegressor(n_estimators=100, random_state=42)
rf_revenue.fit(X_train_revenue_scaled, y_train_revenue)

# 3. For composite score
rf_composite = RandomForestRegressor(n_estimators=100, random_state=42)
rf_composite.fit(X_train_composite_scaled, y_train_composite)

# Evaluate the models
def evaluate_model(model, X_test, y_test, target_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{target_name} Model Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return y_pred

# Evaluate each model
y_pred_customers = evaluate_model(rf_customers, X_test_customers_scaled, y_test_customers, "Customer Count")
y_pred_revenue = evaluate_model(rf_revenue, X_test_revenue_scaled, y_test_revenue, "Daily Revenue")
y_pred_composite = evaluate_model(rf_composite, X_test_composite_scaled, y_test_composite, "Composite Score")

# Feature importance analysis
def analyze_feature_importance(model, X, feature_names, target_name):
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print(f"\nFeature ranking for {target_name}:")
    for f in range(X.shape[1]):
        print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances for {target_name}")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{target_name.replace(" ", "_")}_feature_importance.png')
    
    # Return sorted feature names and their importance values
    return [feature_names[i] for i in indices], importances[indices]

# Analyze feature importance for each model
customer_features, customer_importance = analyze_feature_importance(
    rf_customers, X_train_customers, X_customers.columns, "Customer Count")

revenue_features, revenue_importance = analyze_feature_importance(
    rf_revenue, X_train_revenue, X_revenue.columns, "Daily Revenue")

composite_features, composite_importance = analyze_feature_importance(
    rf_composite, X_train_composite, X_composite.columns, "Composite Score")

# Find optimal operating parameters
def find_optimal_parameters(df, top_features, target_column):
    # Group by various combinations of top features
    if 'Location_Foot_Traffic' in top_features:
        # For continuous variables like foot traffic, create bins
        df['Foot_Traffic_Bin'] = pd.qcut(df['Location_Foot_Traffic'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        grouped = df.groupby('Foot_Traffic_Bin')
        print(f"\nOptimal {target_column} by Location Foot Traffic:")
        print(grouped[target_column].mean().sort_values(ascending=False))
    
    if 'Operating_Hours_Per_Day' in top_features:
        grouped = df.groupby('Operating_Hours_Per_Day')
        print(f"\nOptimal {target_column} by Operating Hours:")
        print(grouped[target_column].mean().sort_values(ascending=False).head())
    
    if 'Number_of_Employees' in top_features:
        grouped = df.groupby('Number_of_Employees')
        print(f"\nOptimal {target_column} by Number of Employees:")
        print(grouped[target_column].mean().sort_values(ascending=False).head())
    
    if 'Marketing_Spend_Per_Day' in top_features:
        # Create bins for marketing spend
        df['Marketing_Bin'] = pd.qcut(df['Marketing_Spend_Per_Day'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        grouped = df.groupby('Marketing_Bin')
        print(f"\nOptimal {target_column} by Marketing Spend:")
        print(grouped[target_column].mean().sort_values(ascending=False))
    
    # Find the best composite score
    if target_column == 'Composite_Score':
        best_rows = df.nlargest(5, 'Composite_Score')
        print("\nTop 5 days with best composite score (balanced optimization):")
        print(best_rows[['Number_of_Customers_Per_Day', 'Daily_Revenue', 'Marketing_Spend_Per_Day', 
                         'Operating_Hours_Per_Day', 'Number_of_Employees', 'Location_Foot_Traffic', 
                         'Composite_Score']])

# Find optimal parameters for each target
print("\n=== OPTIMAL PARAMETERS ANALYSIS ===")
find_optimal_parameters(df, customer_features[:3], 'Number_of_Customers_Per_Day')
find_optimal_parameters(df, revenue_features[:3], 'Daily_Revenue')
find_optimal_parameters(df, composite_features[:3], 'Composite_Score')

# Conduct Permutation Importance Analysis for more reliable feature importance
def permutation_importance_analysis(model, X_test, y_test, feature_names, target_name):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Sort features by importance
    sorted_idx = result.importances_mean.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(result.importances[sorted_idx].T, 
                vert=False, labels=[feature_names[i] for i in sorted_idx])
    plt.title(f"Permutation Importance for {target_name}")
    plt.tight_layout()
    plt.savefig(f'{target_name.replace(" ", "_")}_permutation_importance.png')
    
    print(f"\nPermutation Importance for {target_name}:")
    for i in sorted_idx:
        print(f"{feature_names[i]}: {result.importances_mean[i]:.4f} ± {result.importances_std[i]:.4f}")

# Run permutation importance analysis
permutation_importance_analysis(
    rf_customers, X_test_customers_scaled, y_test_customers, 
    X_customers.columns, "Customer Count")

permutation_importance_analysis(
    rf_revenue, X_test_revenue_scaled, y_test_revenue, 
    X_revenue.columns, "Daily Revenue")

permutation_importance_analysis(
    rf_composite, X_test_composite_scaled, y_test_composite, 
    X_composite.columns, "Composite Score")

# Create a function to suggest actionable recommendations
def provide_recommendations(df, customer_features, revenue_features, composite_features):
    print("\n=== ACTIONABLE RECOMMENDATIONS ===")
    
    # Find optimal marketing spend
    df['Marketing_ROI'] = df['Daily_Revenue'] / df['Marketing_Spend_Per_Day']
    optimal_marketing = df.groupby(pd.qcut(df['Marketing_Spend_Per_Day'], 5))['Marketing_ROI'].mean().sort_values(ascending=False)
    
    print("\n1. Optimal Marketing Spend Range:")
    print(optimal_marketing)
    
    # Find optimal staffing levels
    employee_efficiency = df.groupby('Number_of_Employees')[['Daily_Revenue', 'Number_of_Customers_Per_Day']].mean()
    employee_efficiency['Revenue_per_Employee'] = employee_efficiency['Daily_Revenue'] / employee_efficiency.index
    
    print("\n2. Optimal Staffing Levels:")
    print(employee_efficiency.sort_values('Revenue_per_Employee', ascending=False))
    
    # Find optimal operating hours
    hour_efficiency = df.groupby('Operating_Hours_Per_Day')[['Daily_Revenue', 'Number_of_Customers_Per_Day']].mean()
    hour_efficiency['Revenue_per_Hour'] = hour_efficiency['Daily_Revenue'] / hour_efficiency.index
    hour_efficiency['Customers_per_Hour'] = hour_efficiency['Number_of_Customers_Per_Day'] / hour_efficiency.index
    
    print("\n3. Optimal Operating Hours:")
    print(hour_efficiency.sort_values('Revenue_per_Hour', ascending=False))
    
    print("\n4. Key Factors Summary:")
    print(f"Top factors for maximizing customers: {', '.join(customer_features[:3])}")
    print(f"Top factors for maximizing revenue: {', '.join(revenue_features[:3])}")
    print(f"Top factors for optimizing all objectives: {', '.join(composite_features[:3])}")

# Generate recommendations
provide_recommendations(df, customer_features, revenue_features, composite_features)

print("\nAnalysis completed successfully!")