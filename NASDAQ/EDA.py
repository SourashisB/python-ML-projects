import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
import numpy as np


df = pd.read_csv('5m_data.csv', delimiter='\t')

df["DateTime"] = pd.to_datetime(df["DateTime"])

df.sort_values("DateTime", inplace=True)

# Lagged close price
df['Close_lag1'] = df['Close'].shift(1)
df['Close_lag2'] = df['Close'].shift(2)
df['Close_lag3'] = df['Close'].shift(3)

# Lagged log returns
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['log_return_lag1'] = df['log_return'].shift(1)
df['log_return_lag2'] = df['log_return'].shift(2)

# Lagged volume
df['Volume_lag1'] = df['Volume'].shift(1)
df['Volume_lag2'] = df['Volume'].shift(2)

# RSI
df['RSI_14'] = ta.rsi(df['Close'], length=14)

# MACD
macd = ta.macd(df['Close'])
df['MACD'] = macd['MACD_12_26_9']
df['MACD_signal'] = macd['MACDs_12_26_9']
df['MACD_hist'] = macd['MACDh_12_26_9']

# Bollinger Bands
bb = ta.bbands(df['Close'], length=20, std=2)
df['BBL_20_2.0'] = bb['BBL_20_2.0']  # Lower band
df['BBM_20_2.0'] = bb['BBM_20_2.0']  # Middle band
df['BBU_20_2.0'] = bb['BBU_20_2.0']  # Upper band
df['BB_width'] = bb['BBU_20_2.0'] - bb['BBL_20_2.0']

# Drop NA values from rolling calculations for machine learning
df = df.dropna().reset_index(drop=True)

print(df.head())

def walk_forward_split(df, train_ratio=0.6, test_ratio=0.2):
    """
    Generator for full walk-forward validation splits, 
    where train and test sizes depend on the dataset length.

    Parameters:
    - df: DataFrame, must be sorted by DateTime
    - train_ratio: float, proportion of data for initial training set (e.g., 0.6)
    - test_ratio: float, proportion of data for each test split (e.g., 0.2)

    Yields:
    - train_index: indices for training data
    - test_index: indices for test data
    """
    n_samples = len(df)
    initial_train_size = int(n_samples * train_ratio)
    test_size = int(n_samples * test_ratio)

    # Ensure at least 1 sample in train and test
    initial_train_size = max(1, initial_train_size)
    test_size = max(1, test_size)

    train_start = 0
    train_end = initial_train_size

    while train_end + test_size <= n_samples:
        train_index = list(range(train_start, train_end))
        test_index = list(range(train_end, train_end + test_size))
        yield train_index, test_index
        train_end += test_size  # Expand training window

# Example usage:
splits = list(walk_forward_split(df, train_ratio=0.7, test_ratio=0.15))