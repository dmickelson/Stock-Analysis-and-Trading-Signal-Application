from fastapi import APIRouter, Depends
from pydantic import BaseModel
from portfolio_service import PortfolioService
import pandas as pd

router = APIRouter()


class StockRequest(BaseModel):
    symbol: str
    shares: float
    purchase_price: float


class PortfolioController:
    def __init__(self, portfolio_service: PortfolioService = Depends()):
        self.portfolio_service = portfolio_service

    @router.post("/portfolio/add_stock")
    async def add_stock(self, stock_req: StockRequest):
        self.portfolio_service.add_stock_to_portfolio(
            stock_req.symbol, stock_req.shares, stock_req.purchase_price
        )
        return {"status": "Stock added successfully"}

    @router.delete("/portfolio/remove_stock/{symbol}")
    async def remove_stock(self, symbol: str):
        self.portfolio_service.remove_stock_from_portfolio(symbol)
        return {"status": "Stock removed successfully"}

    @router.get("/portfolio/holdings")
    async def get_holdings(self):
        holdings = self.portfolio_service.get_portfolio_holdings()
        return {"holdings": holdings}

    @router.post("/portfolio/update")
    async def update_portfolio(self, price_data: dict):
        results = self.portfolio_service.update_portfolio_and_signals(
            pd.DataFrame(price_data))
        return results


# Create router instance
portfolio_router = router
