"""
models.py — InvestEng ORM Models
=================================
Tables:
  users            — accounts with demo balance
  portfolios       — one per user (expandable to multiple)
  positions        — current holdings (ticker, qty, avg_cost)
  orders           — all order history (buy/sell)
  watchlists       — user watchlist items
  price_cache      — last-known prices to survive restart
  leaderboard      — cached portfolio value snapshots
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, Text, UniqueConstraint, Index,
)
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(40),  unique=True, nullable=False, index=True)
    email         = Column(String(120), unique=True, nullable=False, index=True)
    hashed_pw     = Column(String(256), nullable=False)
    display_name  = Column(String(80),  nullable=True)
    avatar_color  = Column(String(7),   default="#7c3aed")   # hex
    is_active     = Column(Boolean,     default=True)
    created_at    = Column(DateTime,    default=datetime.utcnow)

    portfolio     = relationship("Portfolio", back_populates="user", uselist=False)
    watchlists    = relationship("Watchlist",  back_populates="user")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    cash_balance    = Column(Float,   default=100_000.0)   # demo money
    initial_balance = Column(Float,   default=100_000.0)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user      = relationship("User",     back_populates="portfolio")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    orders    = relationship("Order",    back_populates="portfolio", cascade="all, delete-orphan")


class Position(Base):
    __tablename__ = "positions"

    id           = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker       = Column(String(20), nullable=False)
    company_name = Column(String(100), nullable=True)
    asset_type   = Column(String(20),  default="us_stocks")
    quantity     = Column(Float,       default=0.0)
    avg_cost     = Column(Float,       default=0.0)   # average purchase price
    realized_pnl = Column(Float,       default=0.0)
    created_at   = Column(DateTime,    default=datetime.utcnow)
    updated_at   = Column(DateTime,    default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="positions")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "ticker", name="uq_portfolio_ticker"),
        Index("ix_positions_portfolio_ticker", "portfolio_id", "ticker"),
    )


class Order(Base):
    __tablename__ = "orders"

    id             = Column(Integer, primary_key=True, index=True)
    portfolio_id   = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker         = Column(String(20),  nullable=False)
    company_name   = Column(String(100), nullable=True)
    asset_type     = Column(String(20),  default="us_stocks")
    order_type     = Column(String(4),   nullable=False)    # "BUY" | "SELL"
    quantity       = Column(Float,       nullable=False)
    price          = Column(Float,       nullable=False)    # execution price
    total_value    = Column(Float,       nullable=False)    # qty × price
    commission     = Column(Float,       default=0.0)       # $0 demo
    status         = Column(String(12),  default="FILLED")  # always FILLED (demo)
    note           = Column(Text,        nullable=True)
    executed_at    = Column(DateTime,    default=datetime.utcnow, index=True)

    portfolio = relationship("Portfolio", back_populates="orders")

    __table_args__ = (
        Index("ix_orders_portfolio_executed", "portfolio_id", "executed_at"),
    )


class Watchlist(Base):
    __tablename__ = "watchlists"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker       = Column(String(20),  nullable=False)
    company_name = Column(String(100), nullable=True)
    asset_type   = Column(String(20),  default="us_stocks")
    added_at     = Column(DateTime,    default=datetime.utcnow)

    user = relationship("User", back_populates="watchlists")

    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_watchlist_user_ticker"),
    )


class PriceCache(Base):
    """Stores last-fetched price per ticker for offline resilience."""
    __tablename__ = "price_cache"

    ticker      = Column(String(20), primary_key=True)
    price       = Column(Float,      nullable=False)
    change      = Column(Float,      default=0.0)     # $ change
    change_pct  = Column(Float,      default=0.0)     # % change
    volume      = Column(Float,      default=0.0)
    market_cap  = Column(Float,      nullable=True)
    day_high    = Column(Float,      nullable=True)
    day_low     = Column(Float,      nullable=True)
    open_price  = Column(Float,      nullable=True)
    company_name= Column(String(100), nullable=True)
    fetched_at  = Column(DateTime,   default=datetime.utcnow)


class PortfolioSnapshot(Base):
    """Hourly snapshots of portfolio value for equity curve chart."""
    __tablename__ = "portfolio_snapshots"

    id           = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    total_value  = Column(Float,   nullable=False)
    cash         = Column(Float,   nullable=False)
    invested     = Column(Float,   nullable=False)
    pnl          = Column(Float,   nullable=False)
    pnl_pct      = Column(Float,   nullable=False)
    snapped_at   = Column(DateTime, default=datetime.utcnow, index=True)
