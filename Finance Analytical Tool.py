import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import linalg
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics
from pandas_datareader import data as web
import yfinance as yf
from scipy.optimize import minimize
from statsmodels.tsa.api import acf, pacf, graphics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import scipy.optimize as sco
import spacy
import openai
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import process

# Initialize spaCy NLP pipeline
nlp = spacy.load("en_core_web_sm")

# Initialize ChatGPT API
openai.api_key = "sk-efghijklmnop5678efghijklmnop5678efghijkl"


def get_stock_list():
    """
    Fetches a list of major stocks using Yahoo Finance's pre-defined market tickers.

    Returns:
        list: A list of dictionaries with 'symbol' and 'name'.
    """
    stock_symbols = [
        "^GSPC",  # S&P 500
        "^DJI",   # Dow Jones
        "^IXIC",  # Nasdaq
    ]

    stock_list = []
    for symbol in stock_symbols:
        ticker_data = yf.Ticker(symbol)
        constituents = ticker_data.history(period="1d")
        if constituents is not None:
            for ticker in constituents.index:
                stock_list.append({"symbol": ticker, "name": ticker_data.info.get("shortName", "Unknown")})

    return stock_list


def get_user_stock_data():
    """
    Collect stock data from the user, including stock names and the number of shares held.
    
    Returns:
        dict: A dictionary where keys are stock names and values are the number of shares.
    """
    stocks_list = get_stock_list()
    stock_data = {}
    print("Enter stock data. Type 'done' when finished.")
    
    while True:
        stock_name = input("Enter the name of the stock (or 'done' to finish): ").strip()
        if stock_name.lower() == 'done':
            break
        if not stock_name:
            print("Stock name cannot be empty. Please try again.")
            continue
        
        try:
            num_shares = int(input(f"Enter the number of shares for {stock_name}: ").strip())
            if num_shares < 0:
                print("Number of shares cannot be negative. Please try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a positive integer for the number of shares.")
            continue
        
        correct_stock_name_list = process.extract(stock_name, stocks_list, limit=5)
        #correct_stock_name = correct_stock_name_list[0][0].upper()
        print(correct_stock_name_list)
        stock_data[stock_name] = num_shares
    
    return stock_data


def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock data for a given symbol and date range.

    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL").
        start_date (str): Start date in the format "YYYY-MM-DD".
        end_date (str): End date in the format "YYYY-MM-DD".

    Returns:
        pandas.DataFrame: DataFrame containing the stock's historical data.
    """
    try:
        print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            print(f"No data found for {symbol} in the given range.")
        else:
            print(f"Data for {symbol} fetched successfully!")
        
        return stock_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def fetch_stock_headlines(stock_name, num_headlines=5):
    """
    Fetches the latest news headlines for a given stock name from Google News.

    Args:
        stock_name (str): The name of the stock to search for.
        num_headlines (int): Number of headlines to return (default is 5).

    Returns:
        list of dict: A list of dictionaries containing headlines and URLs.
    """
    # Prepare the Google News search URL
    search_url = f"https://www.google.com/search?q={stock_name}+news&hl=en&tbm=nws"

    # Define headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Send the GET request to Google
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract headlines and URLs
        headlines = []
        for result in soup.select("div.dbsr", limit=num_headlines):  # Use "dbsr" class for news results
            title = result.select_one("div.JheGif.nDgy9d").text
            link = result.a["href"]
            headlines.append({"title": title, "url": link})

        return headlines

    except Exception as e:
        print(f"An error occurred while fetching headlines: {e}")
        return []


def analyze_financial_text(text):
    """
    Analyze financial text using spaCy for named entity recognition (NER) 
    and ChatGPT API for sentiment analysis.

    Args:
        text (str): Financial news or text to analyze.

    Returns:
        dict: A dictionary containing named entities and sentiment analysis.
    """
    # Named Entity Recognition with spaCy
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Sentiment Analysis with OpenAI API
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Analyze the sentiment of the following financial news:\n\n{text}\n\nSentiment:",
            max_tokens=50,
        )
        sentiment = response['choices'][0]['text'].strip()
    except Exception as e:
        sentiment = f"Error during sentiment analysis: {e}"

    # Combine results
    analysis = {
        "Named Entities": entities,
        "Sentiment Analysis": sentiment
    }
    return analysis


def optimize_portfolio(returns, cov_matrix):
    """
    Optimize portfolio weights to minimize risk for a given target return.

    Args:
        returns (numpy.ndarray): Expected returns of the assets.
        cov_matrix (numpy.ndarray): Covariance matrix of the asset returns.

    Returns:
        dict: A dictionary containing the optimal weights and the optimization result.
    """
    # Objective function: Portfolio variance (risk)
    def portfolio_variance(weights, cov_matrix):
        return weights.T @ cov_matrix @ weights

    # Constraints: Weights must sum to 1
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Sum of weights equals 1
    ]

    # Bounds: Weights must be between 0 and 1 (no short-selling)
    bounds = [(0, 1) for _ in range(len(returns))]

    # Initial guess: Equal allocation
    initial_weights = np.ones(len(returns)) / len(returns)

    # Optimization using SLSQP method
    result = sco.minimize(
        portfolio_variance,
        x0=initial_weights,
        args=(cov_matrix,),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds
    )

    if result.success:
        return {
            "Optimal Weights": result.x,
            "Optimization Result": result
        }
    else:
        raise ValueError(f"Optimization failed: {result.message}")


def prepare_stock_data_for_modeling(stock_data, target_column="Close", test_size=0.2, random_state=42):
    """
    Prepares stock data for modeling by using the date index as a feature and a specified column as the target.

    Args:
        stock_data (pandas.DataFrame): Stock data with a datetime index and target column.
        target_column (str): The column to use as the target variable (default: "Close").
        test_size (float): Proportion of data to use for testing (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        tuple: X_train, X_test, y_train, y_test - Training and testing datasets.
    """
    # Ensure the index is datetime
    if not np.issubdtype(stock_data.index.dtype, np.datetime64):
        raise ValueError("The stock data index must be a datetime type.")

    # Use date index as a numerical feature
    X = np.arange(len(stock_data)).reshape(-1, 1)  # Numerical representation of time
    y = stock_data[target_column].values  # Target values (e.g., closing prices)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def random_forest_model(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def decision_tree_model(X_train, y_train, X_test):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def svr_model(X_train, y_train, X_test):
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def neural_network_model(X_train, y_train, X_test):
    model = Sequential([
        Dense(64, activation='relu', input_dim=1),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)  # Silent training
    y_pred = model.predict(X_test).flatten()
    return model, y_pred





user_stock_data = get_user_stock_data()