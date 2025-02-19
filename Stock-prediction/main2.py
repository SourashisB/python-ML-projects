# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Load the dataset
file_path = "Ali_Baba_Stock_Data.csv"  # Change this to your actual file path
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
# Feature Engineering: Adding Moving Averages & Indicators
df['SMA_10'] = df['Adj Close'].rolling(window=10).mean()
df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
df['EMA_10'] = df['Adj Close'].ewm(span=10).mean()
df['Price Change'] = df['Adj Close'].pct_change()  # Percentage Change
df.dropna(inplace=True)  # Drop NaN values

# Create Lag Features (Past Prices as Input for Prediction)
for lag in range(1, 11):  # Using past 10 days as features
    df[f'Lag_{lag}'] = df['Adj Close'].shift(lag)

df.dropna(inplace=True)  # Drop NaN from lag features

# Selecting Features
features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50', 'EMA_10', 'Price Change'] + [f'Lag_{i}' for i in range(1, 11)]
target = 'Adj Close'

# Splitting into Train-Test (90% Train, 10% Test)
train_size = int(len(df) * 0.90)
train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# Create XGBoost Model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=500, 
    learning_rate=0.01, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)

# Train the Model
xgb_model.fit(X_train, y_train)

# Predictions
predictions = xgb_model.predict(X_test)

# Calculate RMSE & MAE
rmse = math.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
print(f'XGBoost RMSE: {rmse}, XGBoost MAE: {mae}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12,6))
plt.plot(test_df.index, y_test, label="Actual Price", color='blue')
plt.plot(test_df.index, predictions, label="Predicted Price", color='red')
plt.title('Alibaba Stock Price Prediction (XGBoost)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()