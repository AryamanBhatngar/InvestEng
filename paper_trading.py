"""
paper_trading.py — InvestEng Paper Trading Engine
===================================================
Executes simulated orders against real market prices.
Manages positions (avg cost, quantity), P&L, and portfolio snapshots.
All trades are INSTANT FILL at live market price — no slippage simulation
in demo mode (can be enabled via DEMO_SLIPPAGE_BPS env var).
"""

import logging
import os
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from models import Portfolio, Position, Order, PortfolioSnapshot
from market_feed import get_live_quote, get_cached_price, ALL_TICKERS

logger = logging.getLogger("InvestEng.Trading")

SLIPPAGE_BPS = float(os.getenv("DEMO_SLIPPAGE_BPS", "0"))  # 0 = no slippage
MAX_POSITION_PCT = 0.40   # max 40% of portfolio in one position


async def execute_order(
    db      : Session,
    portfolio: Portfolio,
    ticker  : str,
    order_type: str,
    quantity: float,
    note    : Optional[str] = None,
) -> Order:
    """
    Execute a BUY or SELL order at live market price.

    BUY:  deduct cash, increase / open position, update avg_cost
    SELL: reduce / close position, add cash, realise P&L

    Raises HTTPException on validation failure.
    """
    ticker = ticker.upper()

    # ── 1. Validate ticker is in universe ────────────────────────────────
    if ticker not in ALL_TICKERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ticker '{ticker}' is not in the InvestEng tradeable universe.",
        )

    meta = ALL_TICKERS[ticker]

    # ── 2. Get live price ────────────────────────────────────────────────
    quote = await get_live_quote(ticker)
    if not quote or quote["price"] <= 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to fetch live price for {ticker}. Markets may be closed.",
        )

    slippage_factor = 1 + (SLIPPAGE_BPS / 10_000) * (1 if order_type == "BUY" else -1)
    exec_price = round(quote["price"] * slippage_factor, 4)
    total_value = round(exec_price * quantity, 2)

    # ── 3. BUY validation ────────────────────────────────────────────────
    if order_type == "BUY":
        if total_value > portfolio.cash_balance:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Insufficient funds. Order total ${total_value:,.2f} "
                    f"exceeds cash balance ${portfolio.cash_balance:,.2f}."
                ),
            )
        # Position size check
        portfolio_value = _compute_portfolio_value(db, portfolio)
        post_position_value = _get_position_value(db, portfolio, ticker) + total_value
        if post_position_value / portfolio_value > MAX_POSITION_PCT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Position size limit: single asset cannot exceed "
                    f"{MAX_POSITION_PCT * 100:.0f}% of portfolio value."
                ),
            )

    # ── 4. SELL validation ───────────────────────────────────────────────
    if order_type == "SELL":
        position = db.query(Position).filter(
            Position.portfolio_id == portfolio.id,
            Position.ticker == ticker,
        ).first()
        if not position or position.quantity < quantity - 1e-8:
            held = position.quantity if position else 0
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Cannot sell {quantity} shares of {ticker}. "
                    f"You hold {held:.4f} shares."
                ),
            )

    # ── 5. Execute ───────────────────────────────────────────────────────
    order = Order(
        portfolio_id = portfolio.id,
        ticker       = ticker,
        company_name = meta.get("name", ticker),
        asset_type   = meta.get("asset_type", "us_stocks"),
        order_type   = order_type,
        quantity     = quantity,
        price        = exec_price,
        total_value  = total_value,
        commission   = 0.0,
        status       = "FILLED",
        note         = note,
        executed_at  = datetime.utcnow(),
    )
    db.add(order)

    if order_type == "BUY":
        _process_buy(db, portfolio, ticker, meta, quantity, exec_price)
    else:
        _process_sell(db, portfolio, ticker, quantity, exec_price)

    portfolio.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(order)

    logger.info(
        "ORDER FILLED | %s %s x %.4f @ $%.4f | portfolio=%d",
        order_type, ticker, quantity, exec_price, portfolio.id,
    )
    return order


def _process_buy(
    db       : Session,
    portfolio: Portfolio,
    ticker   : str,
    meta     : dict,
    quantity : float,
    price    : float,
):
    """Deduct cash, update or create position with new avg cost."""
    total = quantity * price
    portfolio.cash_balance = round(portfolio.cash_balance - total, 4)

    position = db.query(Position).filter(
        Position.portfolio_id == portfolio.id,
        Position.ticker == ticker,
    ).first()

    if position:
        # Weighted average cost
        old_total  = position.quantity * position.avg_cost
        new_total  = quantity * price
        position.quantity  = round(position.quantity + quantity, 8)
        position.avg_cost  = round((old_total + new_total) / position.quantity, 4)
        position.updated_at = datetime.utcnow()
    else:
        db.add(Position(
            portfolio_id = portfolio.id,
            ticker       = ticker,
            company_name = meta.get("name", ticker),
            asset_type   = meta.get("asset_type", "us_stocks"),
            quantity     = round(quantity, 8),
            avg_cost     = round(price, 4),
            realized_pnl = 0.0,
        ))


def _process_sell(
    db       : Session,
    portfolio: Portfolio,
    ticker   : str,
    quantity : float,
    price    : float,
):
    """Add cash, reduce position, realise P&L."""
    total = quantity * price
    portfolio.cash_balance = round(portfolio.cash_balance + total, 4)

    position = db.query(Position).filter(
        Position.portfolio_id == portfolio.id,
        Position.ticker == ticker,
    ).first()

    # Realise P&L on the sold quantity
    realised = round((price - position.avg_cost) * quantity, 4)
    position.realized_pnl = round(position.realized_pnl + realised, 4)
    position.quantity     = round(position.quantity - quantity, 8)
    position.updated_at   = datetime.utcnow()

    # Remove position if fully closed
    if position.quantity <= 1e-8:
        db.delete(position)


def _get_position_value(db: Session, portfolio: Portfolio, ticker: str) -> float:
    pos = db.query(Position).filter(
        Position.portfolio_id == portfolio.id,
        Position.ticker == ticker,
    ).first()
    if not pos:
        return 0.0
    q = get_cached_price(ticker)
    price = q["price"] if q else pos.avg_cost
    return pos.quantity * price


def _compute_portfolio_value(db: Session, portfolio: Portfolio) -> float:
    """Total portfolio value = cash + market value of all positions."""
    positions = db.query(Position).filter(Position.portfolio_id == portfolio.id).all()
    invested = 0.0
    for pos in positions:
        q = get_cached_price(pos.ticker)
        price = q["price"] if q else pos.avg_cost
        invested += pos.quantity * price
    return portfolio.cash_balance + invested


def build_portfolio_response(db: Session, portfolio: Portfolio) -> dict:
    """
    Compute the full portfolio snapshot with live P&L for every position.
    Returns a dict ready to serialise as PortfolioOut.
    """
    positions = db.query(Position).filter(
        Position.portfolio_id == portfolio.id,
        Position.quantity > 1e-8,
    ).all()

    position_outs = []
    total_invested = 0.0
    total_day_pnl  = 0.0

    for pos in positions:
        q     = get_cached_price(pos.ticker)
        price = q["price"] if q else pos.avg_cost
        prev  = price - (q["change"] if q else 0)

        current_value     = round(pos.quantity * price, 2)
        cost_basis        = round(pos.quantity * pos.avg_cost, 2)
        unrealized_pnl    = round(current_value - cost_basis, 2)
        unrealized_pnl_pct = round((unrealized_pnl / cost_basis * 100) if cost_basis else 0, 2)
        day_change        = round(pos.quantity * (price - prev), 2)
        day_change_pct    = round(((price - prev) / prev * 100) if prev else 0, 2)

        total_invested += current_value
        total_day_pnl  += day_change

        position_outs.append({
            "id"               : pos.id,
            "ticker"           : pos.ticker,
            "company_name"     : pos.company_name,
            "asset_type"       : pos.asset_type,
            "quantity"         : pos.quantity,
            "avg_cost"         : pos.avg_cost,
            "current_price"    : price,
            "current_value"    : current_value,
            "cost_basis"       : cost_basis,
            "unrealized_pnl"   : unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "realized_pnl"     : pos.realized_pnl,
            "day_change"       : day_change,
            "day_change_pct"   : day_change_pct,
        })

    total_value = round(portfolio.cash_balance + total_invested, 2)
    total_pnl   = round(total_value - portfolio.initial_balance, 2)
    total_pnl_pct = round((total_pnl / portfolio.initial_balance * 100), 2)
    day_pnl_pct = round((total_day_pnl / (total_value - total_day_pnl) * 100)
                        if (total_value - total_day_pnl) else 0, 2)

    return {
        "id"             : portfolio.id,
        "cash_balance"   : portfolio.cash_balance,
        "initial_balance": portfolio.initial_balance,
        "total_value"    : total_value,
        "invested_value" : total_invested,
        "total_pnl"      : total_pnl,
        "total_pnl_pct"  : total_pnl_pct,
        "day_pnl"        : round(total_day_pnl, 2),
        "day_pnl_pct"    : day_pnl_pct,
        "positions"      : position_outs,
        "updated_at"     : portfolio.updated_at.isoformat(),
    }


async def take_portfolio_snapshot(db: Session, portfolio_id: int):
    """Called periodically to record equity curve data points."""
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        return
    data = build_portfolio_response(db, portfolio)
    snap = PortfolioSnapshot(
        portfolio_id = portfolio_id,
        total_value  = data["total_value"],
        cash         = data["cash_balance"],
        invested     = data["invested_value"],
        pnl          = data["total_pnl"],
        pnl_pct      = data["total_pnl_pct"],
        snapped_at   = datetime.utcnow(),
    )
    db.add(snap)
    db.commit()
