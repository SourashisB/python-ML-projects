import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
csv_path = 'NVIDIA_STOCK.csv'  # Change to your path
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. Feature Engineering: Add lag features
N_LAGS = 5
for lag in range(1, N_LAGS + 1):
    df[f'Adj_Close_Lag_{lag}'] = df['Adj Close'].shift(lag)

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + [f'Adj_Close_Lag_{lag}' for lag in range(1, N_LAGS + 1)]
df['Adj_Close_Next'] = df['Adj Close'].shift(-1)

# Drop rows with any NaN (due to shifting)
df = df.dropna().reset_index(drop=True)

# Features and target
X = df[feature_cols].values
y = df['Adj_Close_Next'].values

# 3. Train-test split (time series, no shuffling)
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
dates_test = df['Date'][split_index:]

# 4. Model Training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 7. Visualization
plt.figure(figsize=(14, 6))
plt.plot(dates_test, y_test, label='Actual', marker='o')
plt.plot(dates_test, y_pred, label='Predicted', marker='x')
plt.title('Next Day Adj Close: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Adj Close')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Residual Plot
plt.figure(figsize=(14, 4))
plt.plot(dates_test, y_test - y_pred, label='Residual (Actual - Predicted)')
plt.hlines(0, dates_test.iloc[0], dates_test.iloc[-1], colors='red', linestyles='dashed')
plt.title('Prediction Residuals')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()