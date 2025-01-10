from typing import List

import streamlit as st
import pandas as pd
import yfinance as yf


@st.cache_data
def fecth_date(tickers: List[str], start_date="2010-01-01", end_date="2024-12-25", period="1d", data_tupe="Close") -> pd.DataFrame:
    tickers_str = " ".join(tickers)
    yf_ticker = yf.Tickers(tickers_str)
    df = yf_ticker.history(start=start_date, end=end_date, period=period)
    df = df[data_tupe]
    return df

@st.cache_data
def fetch_stock_tickers() -> List[str]:
    tickers_list = pd.read_csv("C:\\Users\\elias\\Documents\\Faculdade\\4° Semestre\\Otimização Linear\\portfolio_optimizer\\portfolio_optimizer\\in\\IBOV.csv", sep=";", encoding="UTF-8")["Codigo"].to_list()
    return [f"{ticker}.SA" for ticker in tickers_list]

if __name__ == "__main__":
    # Fetch data
    tickers = fetch_stock_tickers()
    data = fecth_date(tickers)

    # Create the streamlit interface
    st.write("""
    # App Preço de Ações
    O gráfico abaixo representa a evolução do preço das ações do Itaú (ITUB4) ao longo dos anos
    """)

    # prepare filter view
    st.sidebar.header("Filters")

    # Filter tickers
    selected_tickers = st.multiselect("Selecione as empresas", data.columns)
    if selected_tickers:
        data = data[selected_tickers]
        if len(selected_tickers) == 1:
            unique_ticker = selected_tickers[0]
            data = data.rename(columns={unique_ticker: "Close"})

    # Filter date
    start_date = data.index.min().to_pydatetime()
    end_date = data.index.max().to_pydatetime()
    date_period = st.sidebar.slider("Select period", min_value=start_date, max_value=end_date, value=(start_date, end_date))
    data = data.loc[date_period[0]:date_period[1]]


    # Create a line chart
    st.line_chart(data)

    st.write(
        """
        # Fim do app
        """
    )