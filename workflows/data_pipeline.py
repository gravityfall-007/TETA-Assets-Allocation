"""
Data pipeline to fetch historical prices with yfinance and store CSV for backtests.

Output format:
- Two-level column MultiIndex with names ['Price', 'Ticker'] in the order:
  Close, High, Low, Open, Volume x TICKERS
- Index named 'Date'

This matches the expected CSV format used by the workflows and dashboard, e.g.:
  Row 1 first cell: Price
  Row 2 first cell: Ticker
  Row 3 first cell: Date

Usage:
    python workflows/data_pipeline.py
"""

import yfinance as yf
import pandas as pd
import os

TICKERS = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN', 'GC=F']
START = '2020-01-01'
END = '2025-01-01'
DATA_PATH = '/home/gravityfall_kevin/Desktop/TETA-Assets-Allocation/workflows/data/sample_prices.csv'

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

def main():
    """
    Download prices from Yahoo Finance and save a clean price table to CSV.

    Returns:
        None
    """
    raw = yf.download(TICKERS, start=START, end=END, progress=False, auto_adjust=False)
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Unexpected yfinance format; expected MultiIndex columns with OHLCV.")

    # Ensure we have the desired fields and order
    desired_fields = ['Close', 'High', 'Low', 'Open', 'Volume']
    present_fields = [f for f in desired_fields if f in raw.columns.get_level_values(0)]
    if len(present_fields) == 0:
        raise ValueError("No OHLCV fields found in yfinance output.")

    # Build MultiIndex columns (Price level -> field, Ticker level -> ticker)
    frames = []
    for field in desired_fields:
        if field in present_fields:
            sub = raw[field]
            # Reindex columns to the requested tickers order if available
            available = [t for t in TICKERS if t in sub.columns]
            sub = sub.reindex(columns=available)
            frames.append(sub)

    if not frames:
        raise ValueError("No data frames constructed from OHLCV fields.")

    # Concatenate across fields to form MultiIndex columns with desired order
    price_panel = pd.concat(frames, axis=1, keys=[f for f in desired_fields if f in present_fields])
    price_panel.columns.names = ['Price', 'Ticker']
    price_panel.index.name = 'Date'
    price_panel = price_panel.dropna(how='all')

    price_panel.to_csv(DATA_PATH)
    print(f"Saved data to {DATA_PATH}")

if __name__ == "__main__":
    main()

