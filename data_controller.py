"""
This module contains the DataController class which handles API routes for stock data and market indicators.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from data_service import DataService, DataServiceError
from yahoo_finance_feed import YahooFinanceFeed
from models import StockSymbol, DateRange
from typing import Optional, Dict, Any

router = APIRouter()


def get_data_service():
    return DataService(YahooFinanceFeed())


class DataController:
    @staticmethod
    @router.get("/stock/{symbol}", response_model=Dict[str, Any],
                summary="Get Stock Data",
                description="Retrieve historical stock data for a given symbol within a specified date range. Defaults to last 30 days if no range is provided.")
    async def get_stock_data(
        symbol: str = Path(..., description="Stock symbol"),
        start: Optional[str] = Query(
            None, description="Start date (YYYY-MM-DD)"),
        end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
        data_service: DataService = Depends(get_data_service)
    ):
        try:
            stock = StockSymbol(symbol=symbol)
            date_range = DateRange(
                start=start, end=end) if start and end else None
            result = data_service.get_stock_data(
                stock.symbol, date_range.start if date_range else None, date_range.end if date_range else None)
            return {"data": result}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except DataServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    @router.get("/market-indicators", response_model=Dict[str, Any],
                summary="Get Market Indicators",
                description="Retrieve market indicators including major indices for a specified date range. Defaults to last 30 days if no range is provided.")
    async def get_market_indicators(
        start: Optional[str] = Query(
            None, description="Start date (YYYY-MM-DD)"),
        end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
        data_service: DataService = Depends(get_data_service)
    ):
        try:
            date_range = DateRange(
                start=start, end=end) if start and end else None
            result = data_service.get_market_indicators(
                date_range.start if date_range else None, date_range.end if date_range else None)
            return {"data": result}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except DataServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    @router.post("/reset-database", response_model=Dict[str, Any],
                 summary="Reset Database",
                 description="Reset the stock and market databases, removing all stored data.")
    async def reset_database(
        data_service: DataService = Depends(get_data_service)
    ):
        try:
            result = data_service.reset_database()
            return {"message": result}
        except DataServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))


# Create router instance
data_router = router
