"""
This is the main entry point for the Stock Analysis and Trading Signal API.
It sets up the FastAPI application and includes the data router.
"""

from fastapi import FastAPI
from data_controller import data_router
# from portfolio_controller import portfolio_router

app = FastAPI(
    title="Stock Analysis and Trading Signal API",
    description="An API for stock analysis and trading signal generation",
    version="1.0.0",
)

# Include the data router
app.include_router(data_router, prefix="/api/v1", tags=["data"])
# app.include_router(portfolio_router, prefix="/api/v1", tags=["portfolio"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
