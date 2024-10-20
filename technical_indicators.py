from abc import ABC, abstractmethod
import pandas as pd


class TechnicalIndicatorBase(ABC):
    """
    Abstract base class for technical indicators.
    """

    @abstractmethod
    def set_params(self, params: dict):
        """
        Set the parameters for the indicator dynamically.

        :param params: Dictionary of parameters for the indicator.
        """
        pass

    @abstractmethod
    def calculate_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the technical indicator.

        :param data: DataFrame containing stock price data.
        :return: DataFrame with the indicator values.
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on the indicator values.

        :param data: DataFrame containing stock price data.
        :return: DataFrame with trading signals.
        """
        pass
