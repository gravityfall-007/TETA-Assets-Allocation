"""
Streamlit dashboard for TETA Backtesting.

Run:
    streamlit run dashboards/streamlit_app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workflows.backtesting import (
    backtest_portfolio_with_details,
    INITIAL_CAPITAL,
    TRAIN_DAYS,
    TEST_DAYS,
    VOL_TARGET_ANNUAL,
    MAX_WEIGHT,
    CASH_BUFFER,
    TRANS_COST_BPS,
    SLIPPAGE_BPS,
    SHRINKAGE,
)

DATA_PATH_DEFAULT = '/home/gravityfall_kevin/Desktop/TETA-Assets-Allocation/workflows/data/sample_prices.csv'

st.set_page_config(page_title="TETA Backtester", layout="wide")

st.title("TETA Backtesting Dashboard")

with st.sidebar:
    st.header("Data")
    data_path = st.text_input("CSV path", value=DATA_PATH_DEFAULT)
    uploaded = st.file_uploader("...or upload CSV", type=["csv"]) 

    st.header("Windows")
    train_days = st.number_input("Train days", min_value=60, max_value=1000, value=TRAIN_DAYS, step=21)
    test_days = st.number_input("Test days (rebalance)", min_value=5, max_value=252, value=TEST_DAYS, step=5)

    st.header("Risk & Constraints")
    vol_target = st.number_input("Vol target (annual)", min_value=0.01, max_value=1.0, value=float(VOL_TARGET_ANNUAL), step=0.01, format="%.2f")
    max_w = st.number_input("Max weight per asset", min_value=0.05, max_value=1.0, value=float(MAX_WEIGHT), step=0.05, format="%.2f")
    cash_buf = st.number_input("Cash buffer", min_value=0.0, max_value=0.5, value=float(CASH_BUFFER), step=0.01, format="%.2f")
    shrink = st.number_input("Covariance shrinkage", min_value=0.0, max_value=1.0, value=float(SHRINKAGE), step=0.05, format="%.2f")

    st.header("Trading Frictions")
    trans_bps = st.number_input("Transaction cost (bps)", min_value=0, max_value=100, value=int(TRANS_COST_BPS), step=1)
    slip_bps = st.number_input("Slippage (bps)", min_value=0, max_value=100, value=int(SLIPPAGE_BPS), step=1)

    st.header("Analytics")
    roll_window = st.number_input("Rolling window (days)", min_value=10, max_value=252, value=60, step=5)

    run_btn = st.button("Run backtest")

@st.cache_data(show_spinner=False)
def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer Adjusted Close if present
        for level in ('Adj Close', 'Close'):
            if level in df:
                return df[level].dropna()
        # fallback: try to flatten
        return df.droplevel(0, axis=1)
    return df

def render_results(results: dict, roll_window: int):
    values = results.get("values")
    benchmark = results.get("benchmark")
    weights = results.get("weights")
    params = results.get("params")

    if values is None or values.empty:
        st.warning("No backtest results. Check data length, windows, or CSV path.")
        return

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    final_val = float(values.iloc[-1])
    total_ret = final_val / INITIAL_CAPITAL - 1.0
    ann_ret = (final_val / INITIAL_CAPITAL) ** (252.0 / len(values)) - 1.0 if len(values) > 0 else 0.0
    daily_rets = values.pct_change().dropna()
    ann_vol = float(daily_rets.std() * np.sqrt(252)) if not daily_rets.empty else 0.0
    sharpe = (ann_ret - 0.01) / ann_vol if ann_vol > 0 else 0.0

    col1.metric("Final Value", f"${final_val:,.0f}")
    col2.metric("Total Return", f"{total_ret*100:.1f}%")
    col3.metric("Ann. Return", f"{ann_ret*100:.1f}%")
    col4.metric("Ann. Vol | Sharpe", f"{ann_vol*100:.1f}% | {sharpe:.2f}")

    # Charts
    st.subheader("Equity Curves")
    eq_df = pd.DataFrame({
        "TETA": values,
    })
    if benchmark is not None and not benchmark.empty:
        eq_df["Buy&Hold"] = benchmark
    st.line_chart(eq_df)

    st.subheader("Weights at Rebalance")
    if weights is not None and not weights.empty:
        st.dataframe(weights.style.format("{:.2%}"))
        st.bar_chart(weights)
    else:
        st.info("Weights log unavailable for the selected configuration.")

    # Rolling analytics
    st.subheader("Rolling Analytics")
    # Daily returns aligned to values
    daily = values.pct_change().dropna()
    roll_vol = daily.rolling(roll_window).std() * np.sqrt(252)
    roll_ret = daily.rolling(roll_window).mean() * 252
    roll_sharpe = roll_ret / (roll_vol.replace(0, np.nan))
    roll_df = pd.DataFrame({
        "Rolling Vol": roll_vol,
        "Rolling Sharpe": roll_sharpe,
    }).dropna()
    if not roll_df.empty:
        st.line_chart(roll_df)

    # Turnover between rebalances
    st.subheader("Turnover (per rebalance)")
    if weights is not None and not weights.empty and len(weights) > 1:
        turnover = (weights.diff().abs().sum(axis=1)).dropna()
        st.bar_chart(turnover)
    else:
        st.info("Insufficient weight history to compute turnover.")

    # Drawdowns
    st.subheader("Drawdowns")
    cummax = values.cummax()
    drawdown = (values / cummax - 1.0).fillna(0.0)
    st.line_chart(pd.DataFrame({"Drawdown": drawdown}))

    st.subheader("Parameters Used")
    st.json(params)

if run_btn:
    try:
        if uploaded is not None:
            price_df = pd.read_csv(uploaded, header=[0, 1], index_col=0, parse_dates=True)
            if isinstance(price_df.columns, pd.MultiIndex):
                for level in ('Adj Close', 'Close'):
                    if level in price_df:
                        price_df = price_df[level].dropna()
                        break
                else:
                    price_df = price_df.droplevel(0, axis=1)
        else:
            price_df = load_prices(data_path)
        results = backtest_portfolio_with_details(
            price_df=price_df,
            train_days=train_days,
            test_days=test_days,
            vol_target_annual=vol_target,
            max_weight=max_w,
            cash_buffer=cash_buf,
            trans_cost_bps=trans_bps,
            slippage_bps=slip_bps,
            shrinkage=shrink,
        )
        render_results(results, roll_window=roll_window)
    except Exception as e:
        st.exception(e)
else:
    st.info("Configure parameters in the sidebar and click Run backtest.")


