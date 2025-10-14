import yfinance as yf
import pandas as pd
import os

TICKERS = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN']
START = '2022-01-01'
END = '2024-01-01'
DATA_PATH = 'data/sample_prices.csv'

os.makedirs('data', exist_ok=True)

def main():
    data = yf.download(TICKERS, start=START, end=END, progress=False)['Close']
    data = yf.download(TICKERS, start=START, end=END, progress=False)
    if 'Adj Close' in data:
        data = data['Adj Close'].dropna()
    else:
        print("Warning: 'Adj Close' not found in the data.")

    data = data.dropna()
    data.to_csv(DATA_PATH)
    print(f"Saved data to {DATA_PATH}")

if __name__ == "__main__":
    main()

