"""
Backtesting workflow using TETA (Time Evolution Travel Algorithm).

This script performs a walk-forward portfolio backtest over historical prices:
- Train on a rolling window of length `TRAIN_DAYS` using annualized mean/covariance
  of daily returns computed from `teta.utils`.
- Optimize portfolio weights to maximize the Sharpe ratio via `TETA_Optimizer`.
- Apply the resulting weights out-of-sample for `TEST_DAYS` with daily compounding,
  volatility targeting, weight caps, a cash buffer, and turnover-based costs.
- Repeat until data exhaustion and plot the resulting performance vs. buy-and-hold.

Key parameters:
- `TRAIN_DAYS`: number of trading days used for estimation per rebalance.
- `TEST_DAYS`: number of trading days weights are held before the next rebalance.
- `VOL_TARGET_ANNUAL`: ex-ante annualized volatility target used to scale weights.
- `MAX_WEIGHT`: hard cap on individual asset weights (before cash buffer normalization).
- `CASH_BUFFER`: portion of capital left in cash to reduce drawdowns.
- `TRANS_COST_BPS`/`SLIPPAGE_BPS`: per-dollar trading costs applied to turnover.
- `SHRINKAGE`: intensity of covariance shrinkage toward the diagonal for robustness.

Inputs:
- CSV at `DATA_PATH` with Date index and either a flat column set or a
  MultiIndex where the top level includes 'Close' or 'Adj Close'.

Outputs:
- Printed summary and a saved plot `backtest_enhanced.png`.

Usage:
    python workflows/backtesting.py
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
TRAIN_DAYS = 252      # ~1 trading year for estimation
TEST_DAYS = 90        # rebalance frequency (apply weights for this many days)

# Risk and cost parameters
VOL_TARGET_ANNUAL = 0.10  # 10% annualized vol target
MAX_WEIGHT = 0.40         # cap single-asset weight
CASH_BUFFER = 0.10        # keep some cash uninvested to reduce drawdowns
TRANS_COST_BPS = 10       # transaction cost in basis points per dollar traded
SLIPPAGE_BPS = 5          # slippage in basis points per dollar traded
SHRINKAGE = 0.2           # covariance shrinkage intensity toward diagonal

def sharpe_fitness(weights, mean_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    """
    Compute the Sharpe ratio used as a fitness function for portfolio weights.

    Args:
        weights (ndarray): Portfolio weights vector.
        mean_returns (ndarray): Annualized mean returns for each asset.
        cov_matrix (ndarray): Annualized covariance matrix across assets.
        risk_free_rate (float): Annual risk-free rate for the Sharpe excess return.

    Returns:
        float: Sharpe ratio (to maximize).
    """
    weights = np.array(weights)
    if np.sum(weights) <= 0:
        return -1e6
    weights /= np.sum(weights)
    port_return = np.dot(mean_returns, weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if port_vol == 0:
        return -1e6  # Penalize zero volatility portfolios
    sharpe = (port_return - risk_free_rate) / port_vol
    return sharpe

def backtest_portfolio(price_df):
    """
    Run a rolling walk-forward backtest using TETA-optimized weights.

    Args:
        price_df (DataFrame): Historical adjusted/close prices indexed by date.

    Returns:
        Series: Portfolio values indexed by date over the backtesting horizon.
    """
    n_assets = len(price_df.columns)
    optimizer = TETA_Optimizer(num_coords=n_assets, popSize=50)

    portfolio_values = []
    dates = []
    current_capital = INITIAL_CAPITAL
    prev_weights = np.array([1.0 / n_assets] * n_assets)

    # Loop through rolling training windows and test periods
    for start_idx in range(0, len(price_df) - (TRAIN_DAYS + TEST_DAYS), TEST_DAYS):
        # Define training and test slices
        train_df = price_df.iloc[start_idx : start_idx + TRAIN_DAYS]
        test_df = price_df.iloc[start_idx + TRAIN_DAYS : start_idx + TRAIN_DAYS + TEST_DAYS]
        if len(test_df) == 0:
            break

        # Compute return statistics for training period
        returns_df = compute_daily_returns(train_df)
        mean_returns, cov_matrix = compute_annualized_stats(returns_df)

        # Covariance shrinkage toward diagonal to improve robustness
        diag_cov = np.diag(np.diag(cov_matrix))
        cov_matrix = (1 - SHRINKAGE) * cov_matrix + SHRINKAGE * diag_cov

        R_MIN = [0.0] * n_assets
        R_MAX = [1.0] * n_assets
        R_STEP = [0.01] * n_assets

        def fitness(x):
            x = np.array(x)
            x = np.clip(x, 0, 1)
            if np.sum(x) == 0:
                return -1e6
            # Enforce max weight cap softly: penalize if any exceed
            if np.any(x > MAX_WEIGHT):
                # strong penalty to discourage violating weight cap
                return -1e4
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
        if np.sum(best_weights) == 0:
            # fallback to equal weights if optimizer failed
            best_weights = np.array([1.0 / n_assets] * n_assets)
        else:
            best_weights /= np.sum(best_weights)

        # Hard cap and renormalize within investable bucket (excluding cash buffer)
        best_weights = np.minimum(best_weights, MAX_WEIGHT)
        investable_weight = 1.0 - CASH_BUFFER
        best_weights /= np.sum(best_weights)
        best_weights *= investable_weight

        # Volatility targeting: scale to target portfolio vol on training stats
        ex_ante_vol = np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))
        if ex_ante_vol > 0:
            scale = min(1.5, max(0.5, VOL_TARGET_ANNUAL / ex_ante_vol))
            best_weights *= scale
            # Ensure we still respect investable bucket and caps
            best_weights = np.minimum(best_weights, MAX_WEIGHT)
            if np.sum(best_weights) > 0:
                best_weights *= investable_weight / np.sum(best_weights)

        # Apply transaction costs on rebalance due to turnover
        turnover = np.sum(np.abs(best_weights - prev_weights))
        cost_rate = (TRANS_COST_BPS + SLIPPAGE_BPS) / 10_000.0
        transaction_cost = current_capital * turnover * cost_rate
        current_capital -= transaction_cost

        # Calculate portfolio value evolution during test period with daily compounding
        test_returns = test_df.pct_change().dropna()
        for dt, daily_rets in test_returns.iterrows():
            daily_port_ret = float(np.dot(best_weights, daily_rets.values))
            current_capital *= (1.0 + daily_port_ret)
            portfolio_values.append(current_capital)
            dates.append(dt)

        prev_weights = best_weights.copy()

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

    # Buy-and-hold benchmark aligned EXACTLY to backtest index
    n_assets = price_df.shape[1]
    equal_weights = np.array([1 / n_assets] * n_assets)
    if not backtest_results.empty:
        bh_idx = backtest_results.index
        # Align prices to backtest index and forward-fill gaps (e.g., holidays)
        price_aligned = price_df.reindex(bh_idx).ffill()
        # Daily returns on aligned index; drops the first date automatically
        bh_daily_rets = price_aligned.pct_change().dropna()
        # Compound starting capital along the same dates as backtest_results
        bh_values = INITIAL_CAPITAL * (1.0 + bh_daily_rets.values @ equal_weights)
        aligned_buy_hold = pd.Series(bh_values, index=bh_daily_rets.index)
        # Ensure both series share identical indices
        aligned_buy_hold = aligned_buy_hold.reindex(backtest_results.index)
        buy_hold_returns = (aligned_buy_hold / INITIAL_CAPITAL - 1) * 100
    else:
        aligned_buy_hold = pd.Series(dtype=float)
        buy_hold_returns = pd.Series(dtype=float)

    # Compute rolling volatility (20-day rolling std) for backtested portfolio
    backtest_vol = pd.Series(backtest_returns.values, index=backtest_returns.index).rolling(window=20).std()

    # Plot enhanced chart
    plt.figure(figsize=(14, 7))
    plt.plot(backtest_returns.index, backtest_returns.values, label='TETA Backtested Portfolio', color='blue')
    if len(buy_hold_returns) > 0:
        plt.plot(buy_hold_returns.index, buy_hold_returns.values, label='Buy-and-Hold Portfolio', color='orange', linestyle='--')

    # Add shaded volatility region
    plt.fill_between(backtest_returns.index,
                     backtest_returns.values - backtest_vol.values,
                     backtest_returns.values + backtest_vol.values,
                     color='blue', alpha=0.2, label='Backtested Volatility (20-day)')

    # Annotate rebalance points
    for i in range(0, len(backtest_returns), TEST_DAYS):
        plt.axvline(backtest_returns.index[i], color='grey', linestyle=':', alpha=0.5)
        plt.text(backtest_returns.index[i], plt.ylim()[1] * 0.95, 'Rebalance', rotation=90,
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
