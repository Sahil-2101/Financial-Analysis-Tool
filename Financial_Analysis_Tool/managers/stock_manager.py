"""
Stock data management module for fetching and processing stock data.
"""

import yfinance as yf
from fuzzywuzzy import process
from config import DEFAULT_START_DATE, DEFAULT_END_DATE

class StockDataManager:
    @staticmethod
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

    @staticmethod
    def get_user_stock_data():
        """
        Collect stock data from the user, including stock names and the number of shares held.
        
        Returns:
            dict: A dictionary where keys are stock names and values are the number of shares.
        """
        stocks_list = StockDataManager.get_stock_list()
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
            print(correct_stock_name_list)
            stock_data[stock_name] = num_shares
        
        return stock_data

    @staticmethod
    def fetch_stock_data(symbol, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
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