import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from teta.utils import compute_daily_returns, compute_annualized_stats
from teta.teta_optimizer import TETA_Optimizer

DATA_PATH = '/home/gravityfall_kevin/Desktop/TETA-Assets-Allocation/data/sample_prices.csv'
CAPITAL = 100_000

def sharpe_fitness(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    weights = np.array(weights)
    weights /= np.sum(weights)
    port_return = np.dot(mean_returns, weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if port_vol == 0:
        return -1e6
    sharpe = (port_return - risk_free_rate) / port_vol
    return sharpe

def main():
    price_df = pd.read_csv(
    DATA_PATH,
    header=[0, 1],     # two header rows: Price/Ticker
    index_col=0,       # Date as index
    parse_dates=True   # parse date index
)
    returns_df = compute_daily_returns(price_df)
    mean_returns, cov_matrix = compute_annualized_stats(returns_df)
    n_assets = len(price_df.columns)
    optimizer = TETA_Optimizer(num_coords=n_assets, popSize=50)
    R_MIN = [0.0] * n_assets
    R_MAX = [1.0] * n_assets
    R_STEP = [0.01] * n_assets
    
    def fitness(x):
        x = np.array(x)
        x = np.clip(x, 0, 1)
        x /= np.sum(x)
        return sharpe_fitness(x, mean_returns, cov_matrix)
    
    best_weights, best_fitness = optimizer.optimize(
        fitness_function=fitness,
        max_iterations=100,
        rangeMinP=R_MIN,
        rangeMaxP=R_MAX,
        rangeStepP=R_STEP
    )
    best_weights = np.clip(best_weights, 0, 1)
    best_weights /= np.sum(best_weights)
    print(f"Optimal weights: {best_weights}")
    print(f"Optimal Sharpe Ratio: {best_fitness:.3f}")
    
    last_prices = price_df.iloc[-1].values
    shares = np.floor(best_weights * CAPITAL / last_prices).astype(int)
    order_df = pd.DataFrame({
        "Asset": price_df.columns,
        "Action": ["BUY"] * n_assets,
        "Quantity": shares,
        "Current Price": last_prices,
        "Market Value": shares * last_prices,
        "Weight": best_weights
    })
    print("\nSimulated Orders:\n", order_df)

if __name__ == "__main__":
    main()
