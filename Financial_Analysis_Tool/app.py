"""
Main Streamlit application for the Finance Analytical Tool.
"""

import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd

from managers.stock_manager import StockDataManager
from managers.news_manager import NewsManager
from managers.nlp_manager import NLPManager
from managers.portfolio_optimizer import PortfolioOptimizer
from managers.modeling_manager import ModelingManager
from config import DEFAULT_START_DATE

# --- Streamlit UI ---
st.title("Finance Analytical Tool")
st.write("This tool provides various financial analytics and modeling features.")

# Section: Stock Data Input
st.header("1. Enter Your Stock Holdings")
if 'user_stock_data' not in st.session_state:
    st.session_state['user_stock_data'] = {}

if st.button("Input Stock Data"):
    st.info("Please use the console to input your stock data interactively.")
    user_stock_data = StockDataManager.get_user_stock_data()
    st.session_state['user_stock_data'] = user_stock_data

st.write("Current Stock Data:")
st.write(st.session_state['user_stock_data'])

# Section: Fetch Stock Data
st.header("2. Fetch Historical Stock Data")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)")
start_date = st.date_input("Start Date", datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d"))
end_date = st.date_input("End Date", datetime.today())

if st.button("Fetch Stock Data"):
    if stock_symbol:
        data = StockDataManager.fetch_stock_data(stock_symbol, str(start_date), str(end_date))
        if data is not None and not data.empty:
            st.write(data.tail())
            st.line_chart(data['Close'])
        else:
            st.warning("No data found for the given symbol and date range.")
    else:
        st.warning("Please enter a stock symbol.")

# Section: Fetch News Headlines
st.header("3. Fetch Latest News Headlines")
news_stock = st.text_input("Enter Stock Name for News", key="news")
num_headlines = st.slider("Number of Headlines", 1, 10, 5)
if st.button("Fetch Headlines"):
    if news_stock:
        headlines = NewsManager.fetch_stock_headlines(news_stock, num_headlines)
        if headlines:
            for h in headlines:
                st.markdown(f"- [{h['title']}]({h['url']})")
        else:
            st.warning("No headlines found.")
    else:
        st.warning("Please enter a stock name.")

# Section: Analyze Financial Text
st.header("4. Analyze Financial News/Text")
user_text = st.text_area("Paste Financial News or Text Here")
if st.button("Analyze Text"):
    if user_text:
        analysis = NLPManager.analyze_financial_text(user_text)
        st.write(analysis)
    else:
        st.warning("Please enter some text to analyze.")

# Section: Portfolio Optimization (Demo)
st.header("5. Portfolio Optimization (Demo)")
if st.button("Run Portfolio Optimization Demo"):
    # Demo data
    returns = np.array([0.12, 0.18, 0.15])
    cov_matrix = np.array([
        [0.005, -0.010, 0.004],
        [-0.010, 0.040, -0.002],
        [0.004, -0.002, 0.023]
    ])
    try:
        result = PortfolioOptimizer.optimize_portfolio(returns, cov_matrix)
        st.write("Optimal Weights:", result["Optimal Weights"])
    except Exception as e:
        st.error(str(e))

# Section: Stock Modeling (Demo)
st.header("6. Stock Price Modeling (Demo)")
if st.button("Run Modeling Demo"):
    # Generate synthetic data for demo
    dates = pd.date_range(start=DEFAULT_START_DATE, periods=200)
    prices = np.cumsum(np.random.randn(200)) + 100
    stock_data = pd.DataFrame({"Close": prices}, index=dates)
    
    X_train, X_test, y_train, y_test = ModelingManager.prepare_stock_data_for_modeling(stock_data)
    
    _, rf_pred = ModelingManager.random_forest_model(X_train, y_train, X_test)
    _, dt_pred = ModelingManager.decision_tree_model(X_train, y_train, X_test)
    _, svr_pred = ModelingManager.svr_model(X_train, y_train, X_test)
    _, nn_pred = ModelingManager.neural_network_model(X_train, y_train, X_test)
    
    st.write("Random Forest Prediction (first 5):", rf_pred[:5])
    st.write("Decision Tree Prediction (first 5):", dt_pred[:5])
    st.write("SVR Prediction (first 5):", svr_pred[:5])
    st.write("Neural Network Prediction (first 5):", nn_pred[:5]) 