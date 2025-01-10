import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from PIL import Image
from io import BytesIO

from streamlit import columns

NUM_TRADING_DAYS = 252

def download_data(stocks, start_date, end_date, period="1d", data_type="Close"):
    stocks_str = " ".join(stocks)
    tickers = yf.Tickers(stocks_str)
    df = tickers.history(start=start_date, end=end_date, period=period)
    return df[data_type]

def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:]

def generate_portfolios(stocks, returns, num_portfolios=10_000):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    for _ in range(num_portfolios):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(
            np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w)))
        )
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights))
    )
    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])

def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(stocks, weights, returns):
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(
        fun=min_function_sharpe,
        x0=weights[0],
        args=returns,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    return Image.open(buf)

# Streamlit App
st.title("Portfolio Optimization")

# User Inputs
st.sidebar.header("Portfolio Settings")
stocks = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL,WMT,TSLA,GE,AMZN,DB")

# Date range slider
start_date = dt.date(2010, 1, 1)
end_date = dt.date.today()

date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, end_date),
    step=dt.timedelta(days=1),
)
start_date, end_date = date_range

num_portfolios = st.sidebar.number_input("Number of Portfolios", value=10000, step=1000)

tickers = [ticker.strip() for ticker in stocks.split(",")]

if st.sidebar.button("Run Optimization"):
    try:
        # Data download and calculations
        with st.spinner("### Downloading Data..."):
            stock_data = download_data(tickers, start_date, end_date)

        st.write("### Visualizing Stock Prices")
        st.line_chart(stock_data)

        with st.spinner("### Generating Portfolios"):
            log_daily_returns = calculate_return(stock_data)
            weights, means, risks = generate_portfolios(tickers, log_daily_returns, num_portfolios)

        # Portfolio optimization
        with st.spinner("### Optimizing Portfolio"):
            optimum = optimize_portfolio(tickers, weights, log_daily_returns)

        optimal_weights = [round(opt, 3) for opt in optimum["x"]]
        metrics = statistics(optimum["x"], log_daily_returns)

        st.write("### Optimal Weights and Metrics")
        # Creating formatted text to display weights and metrics in a table-like format
        container = st.container()
        expected_return, expected_volatility, sharpe_ratio = container.columns(3)

        container2 = st.container()
        columns = st.columns(len(tickers))
        for i, col in enumerate(columns):
            col.metric(tickers[i], f"{optimal_weights[i]:.2f}")

        # Highlighting optimal portfolio
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        scatter = ax2.scatter(risks, means, c=means / risks, cmap="viridis")
        plt.colorbar(scatter, label="Sharpe Ratio")
        plt.xlabel("Expected Volatility")
        plt.ylabel("Expected Return")
        plt.title("Optimal Portfolio")
        ax2.scatter(
            metrics[1],
            metrics[0],
            color="red",
            label="Optimal Portfolio",
            marker="*",
            s=200,
        )
        plt.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred: {e}")
