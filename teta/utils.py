import numpy as np

def compute_daily_returns(price_df):
    return price_df.pct_change().dropna()

def compute_annualized_stats(returns_df):
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    return mean_returns.values, cov_matrix.values
