"""
backtesting.py

Performs backtest of portfolio allocation using Time Evolution Travel Algorithm (TETA)
on historical asset price data with yearly rebalancing over a multi-year horizon.

Workflow:
- Use rolling 1-year windows for training TETA optimizer on mean/covariance of returns.
- Optimize portfolio weights to maximize Sharpe ratio on training data.
- Apply weights on following year’s price returns to simulate portfolio value evolution.
- Repeat over multiple yearly periods to evaluate strategy performance over time.

Requirements:
- Requires 'data/sample_prices.csv' with historical adjusted close prices.
- Assumes usage of teta_optimizer.py and utils.py modules for optimization logic and data utilities.

Outputs:
- Prints initial and final capital after backtest.
- Plots portfolio value curve over the backtesting period.

Usage:
$ python workflows/backtesting.py
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from teta.teta_optimizer import TETA_Optimizer
from teta.utils import compute_daily_returns, compute_annualized_stats

# Parameters
DATA_PATH = '/home/gravityfall_kevin/Desktop/TETA-Assets-Allocation/data/sample_prices.csv'
RISK_FREE_RATE = 0.01
INITIAL_CAPITAL = 100_000
REBALANCE_DAYS = 60  # or 90

def sharpe_fitness(weights, mean_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    """
    Computes Sharpe ratio as fitness for given portfolio weights.

    Args:
        weights (ndarray): Portfolio weights vector.
        mean_returns (ndarray): Annualized mean returns.
        cov_matrix (ndarray): Annualized covariance matrix.
        risk_free_rate (float): Annual risk-free rate.

    Returns:
        float: Sharpe ratio (to maximize).
    """
    weights = np.array(weights)
    weights /= np.sum(weights)
    port_return = np.dot(mean_returns, weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if port_vol == 0:
        return -1e6  # Penalize zero volatility portfolios
    sharpe = (port_return - risk_free_rate) / port_vol
    return sharpe

def backtest_portfolio(price_df):
    """
    Runs rolling backtest of portfolio optimization using TETA over price data.

    Args:
        price_df (DataFrame): Historical adjusted close prices indexed by date.

    Returns:
        Series: Portfolio values indexed by date over backtesting horizon.
    """
    n_assets = len(price_df.columns)
    optimizer = TETA_Optimizer(num_coords=n_assets, popSize=50)

    portfolio_values = []
    dates = []
    current_capital = INITIAL_CAPITAL

    # Loop through rolling training windows and next-year test periods
    for start_idx in range(0, len(price_df) - 2 * REBALANCE_DAYS, REBALANCE_DAYS):
        # Define training and test slices
        train_df = price_df.iloc[start_idx : start_idx + REBALANCE_DAYS]
        test_df = price_df.iloc[start_idx + REBALANCE_DAYS : start_idx + 2 * REBALANCE_DAYS]
        if len(test_df) == 0:
            break

        # Compute return statistics for training period
        returns_df = compute_daily_returns(train_df)
        mean_returns, cov_matrix = compute_annualized_stats(returns_df)

        R_MIN = [0.0] * n_assets
        R_MAX = [1.0] * n_assets
        R_STEP = [0.01] * n_assets

        def fitness(x):
            x = np.array(x)
            x = np.clip(x, 0, 1)
            if np.sum(x) == 0:
                return -1e6
            x /= np.sum(x)
            return sharpe_fitness(x, mean_returns, cov_matrix)

        # Optimize portfolio weights using TETA
        best_weights, best_fitness = optimizer.optimize(
            fitness_function=fitness,
            max_iterations=50,
            rangeMinP=R_MIN,
            rangeMaxP=R_MAX,
            rangeStepP=R_STEP
        )

        best_weights = np.clip(best_weights, 0, 1)
        best_weights /= np.sum(best_weights)

        # Calculate portfolio value evolution during test period
        start_price = test_df.iloc[0].values
        for prices in test_df.values:
            returns = (prices - start_price) / start_price
            portfolio_val = current_capital * (1 + np.dot(best_weights, returns))
            portfolio_values.append(portfolio_val)
        dates.extend(test_df.index)
        current_capital = portfolio_values[-1]

    return pd.Series(data=portfolio_values, index=dates)

def main():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Load price data
    price_df = pd.read_csv(
        DATA_PATH,
        header=[0, 1],
        index_col=0,
        parse_dates=True
    )

    if isinstance(price_df.columns, pd.MultiIndex):
        price_df = price_df['Close']

    # Run backtest
    backtest_results = backtest_portfolio(price_df)

    print("Backtesting completed.")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    if backtest_results.empty:
        print("No backtest results — check data length or CSV parsing.")
        return
    print(f"Final portfolio value after backtest: ${backtest_results.iloc[-1]:,.2f}")

    # Convert absolute portfolio values to percentage returns
    backtest_returns = (backtest_results / INITIAL_CAPITAL - 1) * 100

    # Buy-and-hold benchmark
    n_assets = price_df.shape[1]
    initial_prices = price_df.iloc[0].values
    equal_weights = np.array([1/n_assets] * n_assets)
    # Align buy-and-hold to backtested period
    buy_hold_values = INITIAL_CAPITAL * (1 + ((price_df.values - initial_prices) / initial_prices) @ equal_weights)
    aligned_buy_hold = pd.Series(buy_hold_values, index=price_df.index)
    aligned_buy_hold = aligned_buy_hold.loc[backtest_returns.index]
    buy_hold_returns = (buy_hold_values / INITIAL_CAPITAL - 1) * 100

    # Compute rolling volatility (20-day rolling std) for backtested portfolio
    backtest_vol = pd.Series(backtest_returns.values, index=backtest_returns.index).rolling(window=20).std()

    # Plot enhanced chart
    plt.figure(figsize=(14, 7))
    plt.plot(backtest_returns.index, backtest_returns.values, label='TETA Backtested Portfolio', color='blue')
    plt.plot(backtest_returns.index, buy_hold_returns, label='Buy-and-Hold Portfolio', color='orange', linestyle='--')

    # Add shaded volatility region
    plt.fill_between(backtest_returns.index,
                     backtest_returns.values - backtest_vol.values,
                     backtest_returns.values + backtest_vol.values,
                     color='blue', alpha=0.2, label='Backtested Volatility (20-day)')

    # Annotate rebalance points
    REBALANCE_DAYS = 60
    for i in range(0, len(price_df), REBALANCE_DAYS):
        if i < len(backtest_returns):
            plt.axvline(backtest_returns.index[i], color='grey', linestyle=':', alpha=0.5)
            plt.text(backtest_returns.index[i], plt.ylim()[1]*0.95, 'Rebalance', rotation=90,
                     verticalalignment='top', fontsize=8, color='grey')

    plt.title("Backtested Portfolio vs Buy-and-Hold (% Returns)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Returns (%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtest_enhanced.png", dpi=300, bbox_inches='tight')
    print("Enhanced plot saved as backtest_enhanced.png")

if __name__ == "__main__":
    main()
