import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import tensorflow as tf

import pandas_ta as ta

# ------------ Load Artifacts ------------
model = tf.keras.models.load_model('lstm_deploy_model.keras')
feature_scaler = joblib.load('feature_scaler.gz')
target_scaler = joblib.load('target_scaler.gz')
with open('features.json', 'r') as f:
    features = json.load(f)

SEQ_LEN = 30  # must match training

# ------------ Load and Prepare New Data ------------
# Replace with your new data file path
df = pd.read_csv('test.csv', delimiter='\t')
df["DateTime"] = pd.to_datetime(df["DateTime"])
df.sort_values("DateTime", inplace=True)

# Calculate all features
df['Close_lag1'] = df['Close'].shift(1)
df['Close_lag2'] = df['Close'].shift(2)
df['Close_lag3'] = df['Close'].shift(3)
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['log_return_lag1'] = df['log_return'].shift(1)
df['log_return_lag2'] = df['log_return'].shift(2)
df['Volume_lag1'] = df['Volume'].shift(1)
df['Volume_lag2'] = df['Volume'].shift(2)
df['RSI_14'] = ta.rsi(df['Close'], length=14)
macd = ta.macd(df['Close'])
df['MACD'] = macd['MACD_12_26_9']
df['MACD_signal'] = macd['MACDs_12_26_9']
df['MACD_hist'] = macd['MACDh_12_26_9']
bb = ta.bbands(df['Close'], length=20, std=2)
df['BBL_20_2.0'] = bb['BBL_20_2.0']
df['BBM_20_2.0'] = bb['BBM_20_2.0']
df['BBU_20_2.0'] = bb['BBU_20_2.0']
df['BB_width'] = df['BBU_20_2.0'] - df['BBL_20_2.0']

# Remove rows with NaN (due to indicators/lags)
df = df.dropna().reset_index(drop=True)

# ------------ Rolling Prediction ------------
actuals = []
preds = []
timestamps = []

for i in range(SEQ_LEN, len(df)):
    X_window = df.iloc[i-SEQ_LEN:i][features].values
    # Check for any missing (should not be after dropna, but just in case)
    if np.isnan(X_window).any():
        continue
    # Scale features
    X_window_scaled = feature_scaler.transform(X_window)
    X_window_scaled = np.expand_dims(X_window_scaled, axis=0)  # shape (1, SEQ_LEN, n_features)
    # Predict
    y_pred_scaled = model.predict(X_window_scaled, verbose=0)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)[0, 0]
    preds.append(y_pred)
    # Actual close is at the next time step after window
    actuals.append(df.iloc[i]['Close'])
    timestamps.append(df.iloc[i]['DateTime'])

# ------------ Plotting ------------
plt.figure(figsize=(14, 6))
plt.plot(timestamps, actuals, label='Actual Close', color='blue', alpha=0.7)
plt.plot(timestamps, preds, label='Predicted Close', color='orange', alpha=0.7)
plt.title('Predicted vs Actual Close Price')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()

# ------------ Optional: Metrics ------------
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(actuals, preds)
mae = mean_absolute_error(actuals, preds)
print(f"Test MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")