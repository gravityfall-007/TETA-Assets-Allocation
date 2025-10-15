"""
Data pipeline to fetch historical prices with yfinance and store CSV for backtests.

Workflow:
- Download OHLCV for `TICKERS` between `START` and `END` using yfinance.
- Prefer adjusted close ('Adj Close') if available; otherwise warn and use 'Close'.
- Drop missing data and persist to `DATA_PATH` for consumption by workflows.

Usage:
    python workflows/data_pipeline.py
"""

import yfinance as yf
import pandas as pd
import os

TICKERS = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN']
START = '2022-01-01'
END = '2024-01-01'
DATA_PATH = 'data/sample_prices.csv'

os.makedirs('data', exist_ok=True)

def main():
    """
    Download prices from Yahoo Finance and save a clean price table to CSV.

    Returns:
        None
    """
    raw = yf.download(TICKERS, start=START, end=END, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        if 'Adj Close' in raw:
            data = raw['Adj Close'].dropna()
        elif 'Close' in raw:
            print("Warning: 'Adj Close' not found, falling back to 'Close'.")
            data = raw['Close'].dropna()
        else:
            raise ValueError("Expected 'Adj Close' or 'Close' in yfinance output.")
    else:
        # Flat columns; assume it's close-like prices
        data = raw.dropna()

    data = data.dropna()
    data.to_csv(DATA_PATH)
    print(f"Saved data to {DATA_PATH}")

if __name__ == "__main__":
    main()

