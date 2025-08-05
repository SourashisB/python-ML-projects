import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


dow = pd.read_csv('DOW.csv', delimter='\t', parse_dates=['Date'])
nasdaq = pd.read_csv('NASDAQ.csv', delimter='\t', parse_dates=['Date'])

dow = dow.add_prefix('DOW_')
nasdaq = nasdaq.add_prefix('NASDAQ_')
dow = dow.rename(columns={'DOW_Date': 'DateTime'})
nasdaq = nasdaq.rename(columns={'NASDAQ_Date': 'DateTime'})

merged_data = pd.merge(dow, nasdaq, on='DateTime', suffixes=('_DOW', '_NASDAQ'))
merged_data = merged_data.sort_values('DateTime').reset_index(drop=True)

#feature engineering
merged_data['DOW_Return'] = np.log(merged_data['DOW_Close'] / merged_data['DOW_Close'].shift(1))
merged_data['NASDAQ_Return'] = np.log(merged_data['NASDAQ_Close'] / merged_data['NASDAQ_Close'].shift(1))

window = 12 #1 hour approx

#st deviation of return
merged_data['DOW_Volatility'] = merged_data['DOW_Return'].rolling(window=window).std()
merged_data['NASDAQ_Volatility'] = merged_data['NASDAQ_Return'].rolling(window=window).std()

#Z-score of volume
merged_data['DOW_Volume_zscore'] = (merged_data['DOW_Volume'] - merged_data['DOW_Volume'].rolling(window).mean()) / merged_data['DOW_Volume'].rolling(window).std()
merged_data['NASDAQ_Volume_zscore'] = (merged_data['NASDAQ_Volume'] - merged_data['NASDAQ_Volume'].rolling(window).mean()) / merged_data['NASDAQ_Volume'].rolling(window).std()

#rolling correlation
merged_data['Rolling_Correlation'] = merged_data['DOW_Return'].rolling(window=window).corr(merged_data['NASDAQ_Return'])

#now make features dataset
features = merged_data.dropna().reset_index(drop=True)

feature_columns = [
        'DOW_Return', 'DOW_Volatility', 'DOW_Volume_zscore',
    'NASDAQ_Return', 'NASDAQ_Volatility', 'NASDAQ_Volume_zscore',
    'Rolling_Correlation'
]

X = features[feature_columns].values

#Scale features