import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import pandas_ta as ta
import tensorflow as tf

# ------------- Data Preparation -------------
df = pd.read_csv('5m_data.csv', delimiter='\t')
df["DateTime"] = pd.to_datetime(df["DateTime"])
df.sort_values("DateTime", inplace=True)

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
df['BB_width'] = bb['BBU_20_2.0'] - bb['BBL_20_2.0']

df = df.dropna().reset_index(drop=True)

# ------------- Walk-Forward Split Function -------------
def walk_forward_split(df, train_ratio=0.7, test_ratio=0.15):
    n_samples = len(df)
    initial_train_size = int(n_samples * train_ratio)
    test_size = int(n_samples * test_ratio)
    initial_train_size = max(1, initial_train_size)
    test_size = max(1, test_size)
    train_start = 0
    train_end = initial_train_size
    while train_end + test_size <= n_samples:
        train_index = list(range(train_start, train_end))
        test_index = list(range(train_end, train_end + test_size))
        yield train_index, test_index
        train_end += test_size

splits = list(walk_forward_split(df, train_ratio=0.7, test_ratio=0.15))

features = [
    'Close_lag1', 'Close_lag2', 'Close_lag3',
    'log_return', 'log_return_lag1', 'log_return_lag2',
    'Volume_lag1', 'Volume_lag2',
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BB_width'
]
target = 'Close'

seq_length = 30
batch_size = 256
lstm1_units = 128
lstm2_units = 64
dense_units = 16
dropout_rate = 0.3
learning_rate = 0.001
epochs = 25

# --------- Use last split for deployment-ready model (train on as much as possible) ---------
train_idx, test_idx = splits[-1]
train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# Fit scalers on all available data for deployment
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train = train_df[features].values
y_train = train_df[target].values.reshape(-1, 1)
X_test = test_df[features].values
y_test = test_df[target].values.reshape(-1, 1)

# Fit on train, transform both
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)

# Save scalers for deployment
joblib.dump(feature_scaler, 'feature_scaler.gz')
joblib.dump(target_scaler, 'target_scaler.gz')
with open('features.json', 'w') as f:
    json.dump(features, f)

# Save last seq_length samples for demo/validation
np.save('last_seq_X.npy', X_test_scaled[-seq_length:])

# Train generator on all available data (could concatenate train+test for prod, but here train only)
train_gen = TimeseriesGenerator(
    X_train_scaled, y_train_scaled,
    length=seq_length, batch_size=batch_size
)

# --------- Model Definition ---------
model = Sequential([
    LSTM(lstm1_units, input_shape=(seq_length, len(features)), return_sequences=True),
    Dropout(dropout_rate),
    LSTM(lstm2_units),
    Dropout(dropout_rate),
    Dense(dense_units, activation='relu'),
    Dense(1)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',
    metrics=['mae']
)

es = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)

model.fit(
    train_gen,
    epochs=epochs,
    callbacks=[es],
    verbose=2
)

# Save model
model.save('lstm_deploy_model.keras')

print("Model, scalers, and feature list saved for deployment.")