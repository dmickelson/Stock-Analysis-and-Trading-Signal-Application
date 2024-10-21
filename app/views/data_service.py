"""
This module implements the DataService class, which serves as an intermediary between the data controller and the data feed.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
from app.models.data_feed import DataFeedBase, DatabaseError, DataFetchError
import logging


class DataServiceError(Exception):
    """Custom exception for data service errors."""
    pass


class DataService:
    def __init__(self, data_feed: DataFeedBase):
        """
        Initialize the DataService with a Data Feed instance.

        Args:
            data_feed (DataFeedBase): Data FeedBase instance.
        """
        self.data_feed = data_feed
        logging.basicConfig(level=logging.INFO)

    def get_stock_data(self, stock: str, start: Optional[str] = None, end: Optional[str] = None) -> Dict:
        """
        Fetch stock data for a given symbol within a specified date range.

        Args:
            stock (str): The stock symbol.
            start (str): The start date in YYYY-MM-DD format.
            end (str): The end date in YYYY-MM-DD format.

        Returns:
            pd.DataFrame: A DataFrame containing the stock data.

        Raises:
            DataServiceError: If there's an error retrieving the stock data.
        """
        try:
            if not end:
                end = datetime.now().strftime('%Y-%m-%d')
            if not start:
                start = (datetime.now() - timedelta(days=90)
                         ).strftime('%Y-%m-%d')
            data = self.data_feed.get_data(stock, start, end)
            self.logger.info(
                f"Successfully retrieved data for {stock} from {start} to {end}")
            self.logger.debug(data.head())
            return data.to_dict(orient="records")
        except (DatabaseError, DataFetchError) as e:
            logging.error(f"Error in get_stock_data: {e}")
            raise DataServiceError(f"Failed to retrieve stock data: {e}")

    def get_market_indicators(self, start: Optional[str] = None, end: Optional[str] = None):
        """
        Fetch current market indicators.

        Args:
            start (str, optional): The start date in YYYY-MM-DD format. If not provided, defaults to 30 days ago.
            end (str, optional): The end date in YYYY-MM-DD format. If not provided, defaults to today.

        Returns:
            pd.DataFrame: A DataFrame containing market indicator data.

        Raises:
            DataServiceError: If there's an error retrieving the market indicators.
        """
        try:
            if not end:
                end = datetime.now().strftime('%Y-%m-%d')
            if not start:
                start = (datetime.now() - timedelta(days=90)
                         ).strftime('%Y-%m-%d')
            data = self.data_feed.fetch_market_indicators(start, end)
            indices = data['symbol'].unique()
            logging.info(
                f"uccessfully retrieved market indicators for {indices} from {start} to {end}")
            logging.debug(data.head())
            return data.to_dict(orient="records")
        except (DatabaseError, DataFetchError) as e:
            logging.error(f"Error in get_market_indicators: {e}")
            raise DataServiceError(
                f"Failed to retrieve market indicators: {e}")

    def reset_database(self):
        try:
            self.data_feed.reset_database()
            return "Database reset successfully"
        except DatabaseError as e:
            logging.error(f"Error in reset_database: {e}")
            raise DataServiceError(f"Failed to reset database: {e}")
