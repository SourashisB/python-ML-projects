import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM

# Load the dataset
data = pd.read_csv("stock_data.csv", parse_dates=["Date"], index_col="Date")

# Use only the "Close" price for prediction
prices = data["Close"].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Create training sequences
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(prices_scaled, seq_length)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, batch_size=16, epochs=20)

# Predict future values
future_steps = 30
future_predictions = []

last_sequence = X_test[-1]

for _ in range(future_steps):
    next_pred = model.predict(last_sequence.reshape(1, seq_length, 1))[0]
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[1:], next_pred).reshape(seq_length, 1)

# Convert predictions back to original scale
future_predictions = scaler.inverse_transform(future_predictions)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index[-100:], prices[-100:], label="Actual Prices", color="blue")
future_dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq='D')[1:]
plt.plot(future_dates, future_predictions, label="Predicted Prices", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Stock Price Prediction")
plt.show()