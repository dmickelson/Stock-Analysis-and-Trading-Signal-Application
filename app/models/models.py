from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class StockSymbol(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)

    @field_validator('symbol')
    def validate_symbol(cls, v):
        if not v.isalpha():
            raise ValueError(
                "Stock symbol must contain only alphabetic characters")
        return v.upper()


class DateRange(BaseModel):
    start: str = Field(..., description="Start date in YYYY-MM-DD format")
    end: str = Field(..., description="End date in YYYY-MM-DD format")

    @field_validator('start', 'end')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD")
        return v

    @field_validator('end')
    def validate_end_date(cls, v, values):
        if 'start' in values and v < values['start']:
            raise ValueError("End date must be after start date")
        return v
