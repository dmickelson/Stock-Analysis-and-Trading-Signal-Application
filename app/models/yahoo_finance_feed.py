"""
This module implements the YahooFinanceFeed class, which fetches stock data and market indicators from Yahoo Finance.
"""

import yfinance as yf
from data_feed import DataFeedBase, DatabaseError, DataFetchError
from config.config import settings
import pandas as pd
import sqlite3
from sqlite3 import Error
import logging


class YahooFinanceFeed(DataFeedBase):
    def __init__(self, db_path=settings.db_path):
        self.db_path = db_path
        self._create_tables()
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=settings.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S %m/%d/%Y',
            filename='yahoo_finance_feed.log',
            filemode='a'
        )
        # logging.Formatter.converter = lambda *args: datetime.now(timezone('US/Eastern')).timetuple()

    def _create_tables(self):
        """Create necessary tables in the SQLite database if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS stock_data
                                (symbol TEXT, date TEXT, close REAL,
                                PRIMARY KEY (symbol, date))''')
                conn.execute('''CREATE TABLE IF NOT EXISTS market_indicators
                                (symbol TEXT, date TEXT, close REAL,
                                PRIMARY KEY (symbol, date))''')
        except Error as e:
            logging.error(f"Database error: {e}")
            raise DatabaseError(f"Failed to create tables: {e}")

    def get_data(self, stock: str, start: str, end: str, force_refresh: bool = True) -> pd.DataFrame:
        """
        Fetch stock data for a given symbol within a specified date range.

        Args:
            stock (str): The stock symbol.
            start (str): The start date in YYYY-MM-DD format.
            end (str): The end date in YYYY-MM-DD format.
            force_refresh (bool): Whether to force refresh data from Yahoo Finance.

        Returns:
            pd.DataFrame: A DataFrame containing the stock data.

        Raises:
            DatabaseError: If there's an issue with database operations.
            DataFetchError: If there's an issue fetching data from Yahoo Finance.
        """
        logging.info(
            f"Fetching data for {stock} from {start} to {end}. Force refresh: {force_refresh}")
        try:
            if force_refresh:
                ticker = yf.Ticker(stock)
                df = ticker.history(start=start, end=end)[['Close']]
                if df.empty:
                    raise DataFetchError(
                        f"No data available for {stock} between {start} and {end}")
                df.reset_index(inplace=True)
                df['symbol'] = stock

                with sqlite3.connect(self.db_path) as conn:
                    df.to_sql('stock_data', conn,
                              if_exists='replace', index=False)
            else:
                with sqlite3.connect(self.db_path) as conn:
                    df = pd.read_sql_query(f"SELECT * FROM stock_data WHERE symbol = ? AND date BETWEEN ? AND ?",
                                           conn, params=(stock, start, end))

                if df.empty:
                    return self.get_data(stock, start, end, force_refresh=True)

            logging.info(
                f"Successfully fetched data for {stock}. Shape: {df.shape}")
            logging.debug(f"Result: {df.head()}")
            return df.set_index('Date', inplace=True)
        except Error as e:
            logging.error(
                f"Database error while fetching data for {stock}: {e}")
            raise DatabaseError(
                f"Failed to fetch or store data for {stock}: {e}")
        except Exception as e:
            logging.error(f"Error fetching data for {stock}: {e}")
            raise DataFetchError(f"Failed to fetch data for {stock}: {e}")

    def fetch_market_indicators(self, start: str, end: str, force_refresh: bool = True) -> pd.DataFrame:
        """
        Fetch market indicators (S&P 500 and NASDAQ) for a specified date range.

        Args:
            start (str): The start date in YYYY-MM-DD format.
            end (str): The end date in YYYY-MM-DD format.
            force_refresh (bool): Whether to force refresh data from Yahoo Finance.

        Returns:
            pd.DataFrame: A DataFrame containing market indicator data.

        Raises:
            DatabaseError: If there's an issue with database operations.
            DataFetchError: If there's an issue fetching data from Yahoo Finance.
        """
        logging.info(
            f"Fetching market indicators from {start} to {end}. Force refresh: {force_refresh}")
        indices = ['^GSPC', '^IXIC']  # S&P 500 and NASDAQ
        try:
            if force_refresh:
                data = yf.download(indices, start=start, end=end)
                if data.empty:
                    raise DataFetchError(
                        f"No data available for market indicators between {start} and {end}")
                df = data['Close'].reset_index()
                df = df.melt(id_vars=['Date'],
                             var_name='symbol', value_name='close')

                with sqlite3.connect(self.db_path) as conn:
                    df.to_sql('market_indicators', conn,
                              if_exists='replace', index=False)
            else:
                with sqlite3.connect(self.db_path) as conn:
                    df = pd.read_sql_query(
                        "SELECT * FROM market_indicators WHERE date BETWEEN ? AND ?",
                        conn, params=(start, end))

                if df.empty:
                    return self.fetch_market_indicators(start, end, force_refresh=True)
            logging.debug(f"Fetched initial data:")
            logging.debug(df.head())
            # df.set_index('Date', inplace=True)
            logging.info(
                f"Successfully fetched market indicators. Shape: {df.shape}")
            logging.debug(f"Result: {df.head()}")
            return df
        except Error as e:
            logging.error(
                f"Database error while fetching market indicators: {e}")
            raise DatabaseError(
                f"Failed to fetch or store market indicators: {e}")
        except Exception as e:
            logging.error(f"Error fetching market indicators: {e}")
            raise DataFetchError(f"Failed to fetch market indicators: {e}")

    def reset_database(self) -> bool:
        logging.info("Resetting the database")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM stock_data")
                conn.execute("DELETE FROM market_indicators")
            logging.info("Database reset successfully")
            return True
        except Error as e:
            logging.error(f"Error resetting database: {e}")
            raise DatabaseError(f"Failed to reset database: {e}")
