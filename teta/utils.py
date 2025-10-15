import numpy as np

def compute_daily_returns(price_df):
    """
    Compute daily percentage returns from a price DataFrame.

    Args:
        price_df (DataFrame): Prices indexed by date.

    Returns:
        DataFrame: Daily returns with the first row dropped (NaNs removed).
    """
    return price_df.pct_change().dropna()

def compute_annualized_stats(returns_df):
    """
    Compute annualized mean returns and covariance matrix from daily returns.

    Args:
        returns_df (DataFrame): Daily returns indexed by date.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (annualized_means, annualized_covariance)
    """
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    return mean_returns.values, cov_matrix.values
