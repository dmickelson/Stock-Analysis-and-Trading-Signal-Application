from pydantic import BaseModel
from typing import Dict
from data_feed import DataFeedBase


class Portfolio(BaseModel):
    holdings: Dict[str, Dict[str, float]]
    data_feed: DataFeedBase
    initial_value: float = 0.0

    def update_prices(self):
        for stock in self.holdings.keys():
            latest_price = self.data_feed.get_data(
                stock, start=None, end=None).iloc[-1]['Close']
            self.holdings[stock]['current_price'] = latest_price

    def add_stock(self, symbol: str, shares: float, purchase_price: float):
        self.holdings[symbol] = {"shares": shares,
                                 "purchase_price": purchase_price}

    def remove_stock(self, symbol: str):
        if symbol in self.holdings:
            del self.holdings[symbol]

    def get_holdings(self) -> Dict[str, Dict[str, float]]:
        return self.holdings

    def calculate_portfolio_value(self) -> float:
        return sum(holding['shares'] * holding.get('current_price', holding['purchase_price'])
                   for holding in self.holdings.values())
