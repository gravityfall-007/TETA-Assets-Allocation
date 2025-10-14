### Project Overview

Demonstrates use of the TETA optimization algorithm for portfolio allocation, using Yahoo Finance data, going from data acquisition to simulated trade orders.

### Workflow Steps

#### 1. Data Pipeline

- Download sample OHLCV data for selected tickers using yfinance (e.g. AAPL, MSFT, GOOG).
- Store processed daily returns in data/sample_prices.csv.

#### 2. Preprocessing

- Calculate returns, mean, covariance, and any other features needed for optimization.
- Define constraints (e.g. fully invested, no shorting, bounds per asset).

#### 3. TETA Optimization

- Use the Python TETA implementation.
- Fitness function: Maximize Sharpe ratio or risk-adjusted return for given asset universe.

#### 4. Asset Allocation

- Output: Optimal portfolio weights (as suggested by TETA).
- Ensure constraints are respected.

#### 5. Order Generation

- Given a simulated initial capital (e.g., $100,000), convert weights to shares to "buy".
- Generate dummy order tickets (e.g., asset, action, quantity, price).


## 📦 requirements.txt
```
numpy
pandas
yfinance
jupyter
```

***

## 📝 Sample workflow (notebooks/demo_workflow.ipynb)

**Sections:**
1. **Data Download**: Use yfinance to pull 2 years of daily price data for e.g. ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN'].
2. **Feature Engineering**: Compute daily returns, mean annualized return, annualized sample covariance.
3. **TETA Setup**: 
   - Variables: Portfolio weights for N assets.
   - Fitness: Maximize expected annualized return / annualized volatility (Sharpe-like).
   - Constraints: Weights sum to 1, each weight 0-1.
4. **Optimization Loop**: Run TETA for 100 iterations.
5. **Result Interpretation**: Show optimal weights, expected return, volatility.
6. **Order Simulation**: Assign total capital ($100k), compute buy amounts, generate summary DataFrame.

***

## 🔑 Example TETA Fitness Function

```python
def sharpe_fitness(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    # Normalize weights to sum to 1 (soft constraint)
    weights = np.array(weights)
    weights /= np.sum(weights)
    port_return = np.dot(mean_returns, weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (port_return - risk_free_rate) / port_vol
    return sharpe_ratio  # For TETA maximization
```


## 📝 Example Order Output

| Asset | Action | Quantity | Current Price | Market Value |
|-------|--------|----------|---------------|--------------|
| AAPL  | BUY    | 20       | 172.10        | 3442.00      |
| MSFT  | BUY    | 15       | 407.13        | 6106.95      |
| ...   | ...    | ...      | ...           | ...          |


## 🚀 How to Run

1. `pip install -r requirements.txt`
2. Run `python workflows/data_pipeline.py` to fetch and preprocess Yahoo data.
3. Run `python workflows/trading_workflow.py` for end-to-end optimization.
4. Explore/modify in `notebooks/demo_workflow.ipynb`.

### Backtesting

This will simulate portfolio value changes from an initial capital of $100,000, using 1-year training windows and 1-year test periods, and display a performance plot.

Run the following script to backtest the TETA-driven portfolio allocation over historical data with yearly rebalancing:

