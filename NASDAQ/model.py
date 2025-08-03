import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import pandas_ta as ta
import tensorflow as tf
print(tf.__version__)

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
        train_end += test_size  # Expand training window

splits = list(walk_forward_split(df, train_ratio=0.7, test_ratio=0.15))

# ------------- Feature and Target Definitions -------------
features = [
    'Close_lag1', 'Close_lag2', 'Close_lag3',
    'log_return', 'log_return_lag1', 'log_return_lag2',
    'Volume_lag1', 'Volume_lag2',
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BB_width'
]
target = 'Close'

# ------------- Hyperparameters for Tuning -------------
seq_length = 30
batch_size = 256
lstm1_units = 128
lstm2_units = 64
dense_units = 16
dropout_rate = 0.3
learning_rate = 0.001
epochs = 25

results = []
split_mse_list = []

for split_num, (train_idx, test_idx) in enumerate(splits):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Skip splits that are too small
    if len(train_df) <= seq_length or len(test_df) <= seq_length:
        continue

    # --------- Fit Scalers on Train, Transform Train & Test Separately ---------
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train = train_df[features].values
    X_test = test_df[features].values
    y_train = train_df[target].values.reshape(-1, 1)
    y_test = test_df[target].values.reshape(-1, 1)

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)

    # Create generators USING SCALED TARGET
    train_gen = TimeseriesGenerator(
        X_train_scaled, y_train_scaled,
        length=seq_length, batch_size=batch_size
    )
    test_gen = TimeseriesGenerator(
        X_test_scaled, y_test_scaled,
        length=seq_length, batch_size=batch_size
    )

    # --------- Model Definition (Tuned) ---------
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

    es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        callbacks=[es],
        verbose=2
    )

    # --------- Predictions & Evaluation ---------
    y_pred_scaled = model.predict(test_gen).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_eval = y_test[seq_length:].flatten()  # True values in original scale

    mse = np.mean((y_pred - y_test_eval) ** 2)
    mae = np.mean(np.abs(y_pred - y_test_eval))
    print(f"Split {split_num + 1}: Test MSE={mse:.5f}, MAE={mae:.5f}")
    results.append({'split': split_num + 1, 'mse': mse, 'mae': mae})
    split_mse_list.append(mse)

    # --------- Visualization for This Split ---------
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_eval, label='Actual Close', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted Close', color='orange', alpha=0.7)
    plt.title(f'Split {split_num + 1}: Actual vs Predicted Close')
    plt.xlabel('Time Steps')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

if results:
    avg_mse = np.mean([r['mse'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    print("Average MSE over splits:", avg_mse)
    print("Average MAE over splits:", avg_mae)

    # --------- Visualization: MSE per Split ---------
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(split_mse_list) + 1), split_mse_list, marker='o', color='purple')
    plt.title('Test MSE per Walk-Forward Split')
    plt.xlabel('Split Number')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.show()
else:
    print("No valid splits for sequence length and dataset size.")