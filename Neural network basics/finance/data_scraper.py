import yfinance as yf
import pandas as pd

# Function to fetch stock data using yfinance
def fetch_stock_data(symbol="AAPL", period="2y"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)

    if not data.empty:
        data.to_csv("stock_data.csv")
        print("Data successfully saved to stock_data.csv")
    else:
        print("Failed to retrieve data.")

# Collect stock data
fetch_stock_data()