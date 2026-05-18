"""
schemas.py — InvestEng Pydantic Schemas
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Auth ────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username    : str = Field(min_length=3, max_length=40)
    email       : EmailStr
    password    : str = Field(min_length=6, max_length=128)
    display_name: Optional[str] = None

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, numbers, _ and -")
        return v.lower()


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token : str
    token_type   : str = "bearer"
    user_id      : int
    username     : str
    display_name : Optional[str]
    avatar_color : str


class UserOut(BaseModel):
    id          : int
    username    : str
    email       : str
    display_name: Optional[str]
    avatar_color: str
    created_at  : datetime

    class Config:
        from_attributes = True


# ── Market Data ──────────────────────────────────────────────────────────────

class QuoteOut(BaseModel):
    ticker      : str
    company_name: Optional[str]
    price       : float
    change      : float
    change_pct  : float
    volume      : float
    day_high    : Optional[float]
    day_low     : Optional[float]
    open_price  : Optional[float]
    market_cap  : Optional[float]
    asset_type  : str
    fetched_at  : datetime


class SearchResult(BaseModel):
    ticker      : str
    company_name: str
    asset_type  : str
    exchange    : Optional[str]


# ── Trading ──────────────────────────────────────────────────────────────────

class OrderRequest(BaseModel):
    ticker    : str = Field(min_length=1, max_length=20)
    order_type: str = Field(pattern="^(BUY|SELL)$")
    quantity  : float = Field(gt=0)
    note      : Optional[str] = Field(default=None, max_length=200)

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v):
        return v.upper().strip()


class OrderOut(BaseModel):
    id          : int
    ticker      : str
    company_name: Optional[str]
    asset_type  : str
    order_type  : str
    quantity    : float
    price       : float
    total_value : float
    commission  : float
    status      : str
    note        : Optional[str]
    executed_at : datetime

    class Config:
        from_attributes = True


# ── Portfolio ─────────────────────────────────────────────────────────────────

class PositionOut(BaseModel):
    id           : int
    ticker       : str
    company_name : Optional[str]
    asset_type   : str
    quantity     : float
    avg_cost     : float
    current_price: float = 0.0
    current_value: float = 0.0
    cost_basis   : float = 0.0
    unrealized_pnl    : float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl : float = 0.0
    day_change   : float = 0.0
    day_change_pct: float = 0.0

    class Config:
        from_attributes = True


class PortfolioOut(BaseModel):
    id              : int
    cash_balance    : float
    initial_balance : float
    total_value     : float
    invested_value  : float
    total_pnl       : float
    total_pnl_pct   : float
    day_pnl         : float
    day_pnl_pct     : float
    positions       : list[PositionOut]
    updated_at      : datetime

    class Config:
        from_attributes = True


class SnapshotOut(BaseModel):
    total_value: float
    cash       : float
    invested   : float
    pnl        : float
    pnl_pct    : float
    snapped_at : datetime

    class Config:
        from_attributes = True


# ── Watchlist ─────────────────────────────────────────────────────────────────

class WatchlistAdd(BaseModel):
    ticker: str = Field(min_length=1, max_length=20)

    @field_validator("ticker")
    @classmethod
    def upper(cls, v):
        return v.upper().strip()


class WatchlistOut(BaseModel):
    id          : int
    ticker      : str
    company_name: Optional[str]
    asset_type  : str
    added_at    : datetime

    class Config:
        from_attributes = True


# ── Leaderboard ───────────────────────────────────────────────────────────────

class LeaderboardEntry(BaseModel):
    rank        : int
    username    : str
    display_name: Optional[str]
    avatar_color: str
    total_value : float
    pnl         : float
    pnl_pct     : float
