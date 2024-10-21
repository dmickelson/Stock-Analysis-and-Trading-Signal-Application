from typing import List
from app.models.portfolio import Portfolio
from app.models.technical_indicators import TechnicalIndicatorBase
import pandas as pd


class PortfolioService:
    def __init__(self, portfolio: Portfolio, indicators: List[TechnicalIndicatorBase]):
        self.portfolio = portfolio
        self.indicators = indicators

    def update_portfolio_and_signals(self, price_data: pd.DataFrame) -> dict:
        self.portfolio.update_prices()

        results = {}
        for indicator in self.indicators:
            indicator_data = indicator.calculate_indicator(price_data)
            signals = indicator.generate_signals(indicator_data)
            results[indicator.__class__.__name__] = signals

        return {
            "portfolio_value": self.portfolio.calculate_portfolio_value(),
            "signals": results
        }

    def add_stock_to_portfolio(self, symbol: str, shares: float, purchase_price: float):
        self.portfolio.add_stock(symbol, shares, purchase_price)

    def remove_stock_from_portfolio(self, symbol: str):
        self.portfolio.remove_stock(symbol)

    def get_portfolio_holdings(self):
        return self.portfolio.get_holdings()
