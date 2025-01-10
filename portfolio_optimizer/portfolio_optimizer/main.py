import time

import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
from scipy.optimize import linprog
from PIL import Image
from io import BytesIO
from lib.utils import TICKERS_B3

NUM_TRADING_DAYS = 252


def download_data(stocks, start_date, end_date, period="1d", data_type="Close"):
    stocks_str = " ".join(stocks)
    tickers = yf.Tickers(stocks_str)
    df = tickers.history(start=start_date, end=end_date, period=period)
    return df[data_type]


def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:].dropna()


def linear_programming_portfolio(returns, max_allocation=0.5):
    n = returns.shape[1]

    c = -returns.mean().values

    A_eq = np.ones((1, n))
    b_eq = [1]


    bounds = [(0, max_allocation) for _ in range(n)]

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        st.error(f"Falha na otimização: {result.message}")
        return None

    return result


def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    return Image.open(buf)


st.title("Otimização de Portfólio")


st.sidebar.header("Configurações do Portfólio")
tickers = st.sidebar.multiselect("TESTE", TICKERS_B3)
# tickers = [ticker.strip() for ticker in stocks.split(",")]


start_date = dt.date(2010, 1, 1)
end_date = dt.date.today()

date_range = st.sidebar.slider(
    "Selecione o intervalo de datas:",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, end_date),
    step=dt.timedelta(days=1),
)
start_date, end_date = date_range

max_allocation = st.sidebar.slider("Alocação Máxima por Ativo:", min_value=1/max(len(tickers), 1), max_value=1.0, value=0.5, step=0.05)

if st.sidebar.button("Executar Otimização"):
    try:
        with st.spinner("### Calculando Retornos"):
            stock_data = download_data(tickers, start_date, end_date)
            st.write("### Visualização dos Preços das Ações")
            st.line_chart(stock_data)

            log_daily_returns = calculate_return(stock_data)
            if log_daily_returns.isnull().values.any():
                raise ValueError("Dados de retornos contêm valores ausentes. Verifique os dados de entrada.")
        time.sleep(2)
        with st.spinner("### Otimizando Portfólio"):
            result = linear_programming_portfolio(log_daily_returns,
                                                  max_allocation)
            if result is None:
                st.error("Falha na otimização. Verifique as restrições e os dados de entrada.")
                st.stop()

            optimal_weights = result.x
            container1 = st.container()
            container1.header("Pesos Ótimos do Portfólio")
            columns = container1.columns(len(tickers))
            for ticker, weight, column in zip(tickers, optimal_weights, columns):
                column.metric(ticker, f"{weight:.2%}")


            expected_return = -result.fun * NUM_TRADING_DAYS
            cov_matrix = log_daily_returns.cov() * NUM_TRADING_DAYS
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

            container2 = st.container()
            header = container2.header("Métricas do Portfólio")

            expected_return_column, portfolio_risk_column = container2.columns(2)

            expected_return_column.metric(f"Retorno Anual Esperado", f"{expected_return:.2%}")
            portfolio_risk_column.metric(f"Risco do Portfólio (Volatilidade)", f"{portfolio_risk:.2%}")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
