import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import pandas_ta as ta
import tensorflow as tf
print(tf.__version__)
# ------------- Data Preparation (like your EDA.py) -------------
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
def walk_forward_split(df, train_ratio=0.6, test_ratio=0.2):
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
        train_end += test_size  # Expand training window

splits = list(walk_forward_split(df, train_ratio=0.7, test_ratio=0.15))

# ------------- Feature Scaling -------------
features = [
    'Close_lag1', 'Close_lag2', 'Close_lag3',
    'log_return', 'log_return_lag1', 'log_return_lag2',
    'Volume_lag1', 'Volume_lag2',
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BB_width'
]
target = 'Close'

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# ------------- TimeseriesGenerator LSTM Training -------------
seq_length = 30
batch_size = 512
results = []

for split_num, (train_idx, test_idx) in enumerate(splits):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Skip splits that are too small
    if len(train_df) <= seq_length or len(test_df) <= seq_length:
        continue

    # Create generators
    train_gen = TimeseriesGenerator(
        train_df[features].values, train_df[target].values,
        length=seq_length, batch_size=batch_size
    )
    test_gen = TimeseriesGenerator(
        test_df[features].values, test_df[target].values,
        length=seq_length, batch_size=batch_size
    )

    # Model definition
    model = Sequential([
        LSTM(64, input_shape=(seq_length, len(features)), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=20,
        callbacks=[es],
        verbose=2
    )

    y_pred = model.predict(test_gen).flatten()
    y_test = test_df[target].values[seq_length:]
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Split {split_num + 1}: Test MSE={mse:.5f}")
    results.append(mse)

if results:
    print("Average MSE over splits:", np.mean(results))
else:
    print("No valid splits for sequence length and dataset size.")