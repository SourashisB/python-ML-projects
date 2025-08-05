import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import mplfinance as mpf
dow = pd.read_csv('dow.csv', delimiter='\t')
nasdaq = pd.read_csv('nasdaq.csv', delimiter='\t')
dow['DateTime'] = pd.to_datetime(dow['DateTime'])
nasdaq['DateTime'] = pd.to_datetime(nasdaq['DateTime'])
dow.columns = dow.columns.str.strip()
nasdaq.columns = nasdaq.columns.str.strip()

dow = dow.rename(columns={col: f"DOW_{col}" for col in dow.columns if col != 'DateTime'})
nasdaq = nasdaq.rename(columns={col: f"NASDAQ_{col}" for col in nasdaq.columns if col != 'DateTime'})

merged_data = pd.merge(dow, nasdaq, on='DateTime', suffixes=('_DOW', '_NASDAQ'))
merged_data = merged_data.sort_values('DateTime').reset_index(drop=True)
#feature engineering

window = 12 #1 hour approx

merged_data['DOW_Return'] = np.log(merged_data['DOW_Close'] / merged_data['DOW_Close'].shift(1))
merged_data['NASDAQ_Return'] = np.log(merged_data['NASDAQ_Close'] / merged_data['NASDAQ_Close'].shift(1))

#st deviation of return
merged_data['DOW_Volatility'] = merged_data['DOW_Return'].rolling(window=window).std()
merged_data['NASDAQ_Volatility'] = merged_data['NASDAQ_Return'].rolling(window=window).std()

#rolling correlation
merged_data['Rolling_Correlation'] = merged_data['DOW_Return'].rolling(window=window).corr(merged_data['NASDAQ_Return'])

print(merged_data.isna().sum())
#now make features dataset

feature_columns = [
    'DOW_Return', 'DOW_Volatility',
    'NASDAQ_Return', 'NASDAQ_Volatility',
    'Rolling_Correlation',
]
features = merged_data[['DateTime', 'DOW_Open', 'DOW_High', 'DOW_Low', 'DOW_Close',
               'NASDAQ_Open', 'NASDAQ_High', 'NASDAQ_Low', 'NASDAQ_Close'] + feature_columns].dropna().reset_index(drop=True)

print("features shape after dropna:", features.shape)

X = features[feature_columns].values

#Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#KMeans clustering
clusters = 5
kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
features['Regime'] = kmeans.fit_predict(X_scaled)


#Regimes Statistics
regime_stats = features.groupby('Regime').agg({
    'DOW_Return': ['mean', 'std'],
    'NASDAQ_Return': ['mean', 'std'],
    'DOW_Volatility': 'mean',
    'NASDAQ_Volatility': 'mean',
    'Rolling_Correlation': 'mean',
    'DateTime': 'count'
}).rename(columns={'DateTime': 'Count'})
print("\n=== Regime statistics (use this to assign labels) ===\n")
print(regime_stats)

#edit map
regime_label_map = {
    0: 'Bull',
    1: 'Bear',
    2: 'Crisis',
    3: 'Systemic',
    4: 'Neutral'
}
features['Regime_Label'] = features['Regime'].map(regime_label_map)

#Plot
regime_colors = {
    'Bull': 'limegreen',
    'Bear': 'firebrick',
    'Crisis': 'black',
    'Systemic': 'orange',
    'Neutral': 'royalblue'
}

features_for_export = features[['DateTime', 'DOW_Close', 'NASDAQ_Close', 'Regime', 'Regime_Label']]
features_for_export.to_csv('market_regimes_for_frontend.csv', index=False)
print("Data saved to market_regimes_for_frontend.csv")

def plot_line_with_regimes(features, col, name):
    plt.figure(figsize=(18,6))
    # Plot the line itself in gray
    plt.plot(features['DateTime'], features[col], color='gray', alpha=0.5, linewidth=1, label=f'{name} Close')
    # Plot each regime as colored points
    for regime_label in features['Regime_Label'].unique():
        mask = features['Regime_Label'] == regime_label
        plt.scatter(features.loc[mask, 'DateTime'],
                    features.loc[mask, col],
                    color=regime_colors.get(regime_label, 'gray'),
                    label=regime_label,
                    s=12, alpha=0.8)
    plt.title(f'{name} Close with Regimes')
    plt.xlabel('DateTime')
    plt.ylabel('Close')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=regime_colors[label], label=label, markersize=10) for label in regime_colors]
    plt.legend(handles=handles, title="Regime", fontsize=12)
    plt.tight_layout()
    plt.show()

# Plot DOW
plot_line_with_regimes(features, 'DOW_Close', 'DOW')

# Plot NASDAQ
plot_line_with_regimes(features, 'NASDAQ_Close', 'NASDAQ')