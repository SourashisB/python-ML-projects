import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load dataset
df = pd.read_csv("average-monthly-surface-temperature.csv")

# Fix duplicate column names by renaming them properly
df.columns = ['Entity', 'Code', 'Year', 'Day', 'Daily Temp', 'Monthly Temp']

# Convert 'Day' column to datetime format
df['Day'] = pd.to_datetime(df['Day'])

# Drop rows with missing values
df.dropna(inplace=True)

# Feature Engineering: Extract month and year
df['Month'] = df['Day'].dt.month
df['Year'] = df['Day'].dt.year

# Normalize temperature values
scaler = StandardScaler()
df[['Daily Temp', 'Monthly Temp']] = scaler.fit_transform(df[['Daily Temp', 'Monthly Temp']])

# Apply K-Means Clustering to identify temperature patterns
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['Daily Temp', 'Monthly Temp']])

# Apply PCA for visualization
pca = PCA(n_components=2)
df[['PCA1', 'PCA2']] = pca.fit_transform(df[['Daily Temp', 'Monthly Temp']])

# Save processed data
df.to_csv("processed_temperature_data.csv", index=False)

print("Data preprocessing complete. Processed data saved to 'processed_temperature_data.csv'.")


# Load processed data
df = pd.read_csv("processed_temperature_data.csv")

# Initialize Dash app
app = dash.Dash(__name__)

# Scatter plot of PCA clusters
fig_clusters = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', hover_data=['Entity', 'Year'],
                          title="Temperature Clusters (PCA)")

# Line chart of global temperature trends
fig_trends = px.line(df.groupby(['Year'])[['Daily Temp', 'Monthly Temp']].mean().reset_index(),
                     x='Year', y=['Daily Temp', 'Monthly Temp'],
                     title="Global Temperature Trends")

# Dropdown for year selection
year_options = [{'label': str(year), 'value': year} for year in sorted(df['Year'].unique())]

app.layout = html.Div([
    html.H1("Surface Temperature Analysis", style={'textAlign': 'center'}),
    
    html.Label("Select Year:"),
    dcc.Dropdown(id="year-dropdown", options=year_options, value=df['Year'].min()),

    html.Div([
        html.H3("Temperature Clusters"),
        dcc.Graph(figure=fig_clusters)
    ], style={'width': '50%', 'display': 'inline-block'}),

    html.Div([
        html.H3("Temperature Trends Over Time"),
        dcc.Graph(figure=fig_trends)
    ], style={'width': '50%', 'display': 'inline-block'}),

    html.Div([
        html.H3("Daily Temperature by Date"),
        dcc.Graph(id="date-temp-chart")
    ])
])

@app.callback(
    dash.Output("date-temp-chart", "figure"),
    [dash.Input("year-dropdown", "value")]
)
def update_graph(selected_year):
    filtered_df = df[df['Year'] == selected_year]
    fig = px.scatter(filtered_df, x='Day', y='Daily Temp', color='Cluster',
                     title=f"Daily Temperature for {selected_year}")
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)