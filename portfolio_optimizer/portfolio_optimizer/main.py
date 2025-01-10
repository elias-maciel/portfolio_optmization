from typing import List

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

NUM_TRADING_DAYS = 252

def download_data(stocks: List[str], start_date: str, end_date: str, period="1d", data_type="Close") -> pd.DataFrame:
    stocks_str = " ".join(stocks)
    tickers = yf.Tickers(stocks_str)
    df = tickers.history(start=start_date, end=end_date, period=period)
    return df[data_type]

def show_data(data: pd.DataFrame):
    data.plot(figsize=(10, 5))
    plt.show()


def calculate_return(data: pd.DataFrame):
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

def show_statistics(returns: pd.DataFrame):
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)


def generate_portfolios(stocks: List[str], returns: pd.DataFrame, num_portfolios = 10_000):
    portfolio_means = []
    portfolio_risks =  []
    portfolio_weights = []
    for _ in range(num_portfolios):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def show_mean_variance(returns: pd.DataFrame, weights: List[float]):
    returns_mean = returns.mean()
    returns_cov = returns.cov()
    portfolio_return = np.sum(returns_mean * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_cov*NUM_TRADING_DAYS, weights)))
    print(f"Return: {portfolio_return:.2f}")
    print(f"Volatility: {portfolio_volatility:.2f}")

def show_portfolios(returns: np.ndarray, volatilities):
    plt.figure(figsize=(10, 5))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker="o")
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(stocks, weights, returns):
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method="SLSQP", bounds=bounds, constraints=constraints)


def print_optimal_portfolio(optimum, returns):
    print("Optimal weights:", optimum["x"].round(3))
    print("Expected return, volatility and Sharpe ratio:", statistics(optimum["x"].round(3), returns))

def show_optimal_portfolio(optimum, returns, portfolio_returns, portfolio_volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns/portfolio_volatilities, marker="o")
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.plot(statistics(optimum["x"], returns)[1], statistics(optimum["x"], returns)[0], "g*", markersize=20.0)
    plt.show()

if __name__ == "__main__":
    stocks = ["AAPL", "WMT", "TSLA", "GE", "AMZN", "DB"]
    start_date = "2010-01-01"
    end_date = "2017-01-01"


    stock_data = download_data(stocks=stocks, start_date=start_date, end_date=end_date)
    show_data(stock_data)
    log_daily_returns = calculate_return(stock_data)

    # show_statistics(log_daily_returns)


    weights, means, risks = generate_portfolios(stocks, log_daily_returns)
    show_portfolios(means, risks)

    optimum = optimize_portfolio(stocks, weights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)
