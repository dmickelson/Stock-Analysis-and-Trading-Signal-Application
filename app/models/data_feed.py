"""
This module defines the abstract base class for data feeds in the Stock Analysis and Trading Signal Application.
"""

from abc import ABC, abstractmethod
import pandas as pd


class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass


class DataFetchError(Exception):
    """Custom exception for data fetching errors."""
    pass

class DataFeedBase(ABC):
    @abstractmethod
    def get_data(self, stock: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch stock data for a given symbol within a specified date range.

        Args:
            stock (str): The stock symbol.
            start (str): The start date in YYYY-MM-DD format.
            end (str): The end date in YYYY-MM-DD format.

        Returns:
            pd.DataFrame: A DataFrame containing the stock data.
        """
        pass

    @abstractmethod
    def fetch_market_indicators(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch current market indicators within a specified date range..

        Returns:
            pd.DataFrame: A DataFrame containing market indicator data.
        """
        pass

    @abstractmethod
    def reset_database(self):
        """
        Reset the database, removing all stored data.
        """
        pass
