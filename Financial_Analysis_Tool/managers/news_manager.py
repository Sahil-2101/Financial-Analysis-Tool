"""
News management module for fetching and processing financial news.
"""

import requests
from bs4 import BeautifulSoup
from config import DEFAULT_NUM_HEADLINES

class NewsManager:
    @staticmethod
    def fetch_stock_headlines(stock_name, num_headlines=DEFAULT_NUM_HEADLINES):
        """
        Fetches the latest news headlines for a given stock name from Google News.

        Args:
            stock_name (str): The name of the stock to search for.
            num_headlines (int): Number of headlines to return.

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