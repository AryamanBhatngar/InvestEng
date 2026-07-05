"""
main.py — InvestEng FastAPI Application
=========================================
The complete backend for the InvestEng paper trading platform.

Routes
------
  POST /auth/register          — create account
  POST /auth/login             — get JWT token
  GET  /auth/me                — current user profile

  GET  /market/prices          — all live prices
  GET  /market/quote/{ticker}  — single live quote
  GET  /market/search          — search by ticker/name
  GET  /market/universe        — full tradeable universe

  GET  /portfolio              — portfolio with live P&L
  GET  /portfolio/history      — equity curve snapshots
  GET  /portfolio/orders       — order history

  POST /trade                  — execute BUY or SELL order

  GET  /watchlist              — user watchlist
  POST /watchlist              — add to watchlist
  DELETE /watchlist/{ticker}   — remove from watchlist

  GET  /leaderboard            — top portfolios ranked by P&L

  WS   /ws/{token}             — real-time price stream

Run:  uvicorn main:app --reload --port 8000
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import (
    FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect,
    status, Query, BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend import database
from backend.database import get_db, engine
from backend.models import(
    Base, User, Portfolio, Position, Order, Watchlist,
    PortfolioSnapshot, PriceCache,
)
from schemas import (
    RegisterRequest, LoginRequest, TokenResponse, UserOut,
    OrderRequest, OrderOut, PortfolioOut,
    WatchlistAdd, WatchlistOut, LeaderboardEntry,
    SnapshotOut,
)
from backend.auth import hash_password, verify_password, create_token, get_current_user
from backend.market_feed import (
    price_refresh_loop,
    get_live_quote,
    get_all_cached,
    search_tickers,
    TRADEABLE_UNIVERSE,
    ALL_TICKERS,
    register_ws,
    unregister_ws,
)
from backend.paper_trading import (
    execute_order, build_portfolio_response, take_portfolio_snapshot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("InvestEng.API")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all DB tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised")

    # Start market data feed in background
    feed_task = asyncio.create_task(price_refresh_loop())
    logger.info("Market feed started")

    # Start hourly snapshot task
    snapshot_task = asyncio.create_task(periodic_snapshots())

    yield

    feed_task.cancel()
    snapshot_task.cancel()
    logger.info("InvestEng API shut down")


async def periodic_snapshots():
    """Take portfolio snapshots every 30 minutes."""
    while True:
        await asyncio.sleep(1800)
        db: Session = database.SessionLocal()
        try:
            portfolios = db.query(Portfolio).all()
            for p in portfolios:
                await take_portfolio_snapshot(db, p.id)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="InvestEng API",
    description="Real-time paper trading platform",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
import os
from pathlib import Path
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already taken.")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")

    colors = ["#7c3aed","#059669","#dc2626","#d97706","#2563eb","#db2777","#0891b2"]
    color  = colors[hash(req.username) % len(colors)]

    user = User(
        username     = req.username,
        email        = req.email,
        hashed_pw    = hash_password(req.password),
        display_name = req.display_name or req.username.title(),
        avatar_color = color,
    )
    db.add(user)
    db.flush()

    # Create portfolio with $100,000 demo balance
    portfolio = Portfolio(user_id=user.id, cash_balance=100_000.0, initial_balance=100_000.0)
    db.add(portfolio)
    db.commit()
    db.refresh(user)

    token = create_token(user.id, user.username)
    return TokenResponse(
        access_token = token,
        user_id      = user.id,
        username     = user.username,
        display_name = user.display_name,
        avatar_color = user.avatar_color,
    )


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username.lower()).first()
    if not user or not verify_password(req.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_token(user.id, user.username)
    return TokenResponse(
        access_token = token,
        user_id      = user.id,
        username     = user.username,
        display_name = user.display_name,
        avatar_color = user.avatar_color,
    )


@app.get("/auth/me", response_model=UserOut, tags=["Auth"])
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# ---------------------------------------------------------------------------
# Market Data Routes
# ---------------------------------------------------------------------------

@app.get("/market/prices", tags=["Market"])
def get_all_prices():
    prices = get_all_cached()
    return {"prices": prices, "count": len(prices), "timestamp": datetime.utcnow().isoformat()}


@app.get("/market/quote/{ticker}", tags=["Market"])
async def get_quote(ticker: str):
    ticker = ticker.upper()
    quote  = await get_live_quote(ticker)
    if not quote:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")
    return quote


@app.get("/market/search", tags=["Market"])
def search_market(q: str = Query(min_length=1), limit: int = 10):
    return {"results": search_tickers(q, limit=limit)}


@app.get("/market/universe", tags=["Market"])
def get_universe():
    return {
        "universe": TRADEABLE_UNIVERSE,
        "total": len(ALL_TICKERS),
    }


@app.get("/market/movers", tags=["Market"])
def get_movers():
    """Top gainers, losers, and most active from cached prices."""
    prices = list(get_all_cached().values())
    if not prices:
        return {"gainers": [], "losers": [], "active": []}
    sorted_by_pct = sorted(prices, key=lambda x: x.get("change_pct", 0))
    sorted_by_vol = sorted(prices, key=lambda x: x.get("volume", 0), reverse=True)
    return {
        "gainers": list(reversed(sorted_by_pct[-5:])),
        "losers" : sorted_by_pct[:5],
        "active" : sorted_by_vol[:5],
    }


# ---------------------------------------------------------------------------
# Portfolio Routes
# ---------------------------------------------------------------------------

@app.get("/portfolio", tags=["Portfolio"])
def get_portfolio(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found.")
    return build_portfolio_response(db, portfolio)


@app.get("/portfolio/history", tags=["Portfolio"])
def get_portfolio_history(
    limit: int = Query(default=168, le=720),   # default: 1 week of hourly
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found.")

    snaps = (
        db.query(PortfolioSnapshot)
        .filter(PortfolioSnapshot.portfolio_id == portfolio.id)
        .order_by(PortfolioSnapshot.snapped_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "history": [
            {
                "total_value": s.total_value,
                "cash"       : s.cash,
                "invested"   : s.invested,
                "pnl"        : s.pnl,
                "pnl_pct"    : s.pnl_pct,
                "snapped_at" : s.snapped_at.isoformat(),
            }
            for s in reversed(snaps)
        ]
    }


@app.get("/portfolio/orders", tags=["Portfolio"])
def get_orders(
    limit: int = Query(default=50, le=200),
    ticker: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found.")

    q = db.query(Order).filter(Order.portfolio_id == portfolio.id)
    if ticker:
        q = q.filter(Order.ticker == ticker.upper())
    orders = q.order_by(Order.executed_at.desc()).limit(limit).all()
    return {
        "orders": [
            {
                "id"          : o.id,
                "ticker"      : o.ticker,
                "company_name": o.company_name,
                "asset_type"  : o.asset_type,
                "order_type"  : o.order_type,
                "quantity"    : o.quantity,
                "price"       : o.price,
                "total_value" : o.total_value,
                "status"      : o.status,
                "note"        : o.note,
                "executed_at" : o.executed_at.isoformat(),
            }
            for o in orders
        ],
        "count": len(orders),
    }


# ---------------------------------------------------------------------------
# Trading Route
# ---------------------------------------------------------------------------

@app.post("/trade", tags=["Trading"])
async def place_order(
    req: OrderRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found.")

    order = await execute_order(
        db         = db,
        portfolio  = portfolio,
        ticker     = req.ticker,
        order_type = req.order_type,
        quantity   = req.quantity,
        note       = req.note,
    )

    # Async snapshot after trade
    background_tasks.add_task(take_portfolio_snapshot, db, portfolio.id)

    return {
        "success"   : True,
        "order_id"  : order.id,
        "message"   : (
            f"{'Bought' if order.order_type == 'BUY' else 'Sold'} "
            f"{order.quantity:.4f} shares of {order.ticker} "
            f"@ ${order.price:,.2f} (Total: ${order.total_value:,.2f})"
        ),
        "order": {
            "id"         : order.id,
            "ticker"     : order.ticker,
            "order_type" : order.order_type,
            "quantity"   : order.quantity,
            "price"      : order.price,
            "total_value": order.total_value,
            "executed_at": order.executed_at.isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# Watchlist Routes
# ---------------------------------------------------------------------------

@app.get("/watchlist", tags=["Watchlist"])
def get_watchlist(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items = db.query(Watchlist).filter(Watchlist.user_id == current_user.id).all()
    result = []
    prices = get_all_cached()
    for item in items:
        q = prices.get(item.ticker, {})
        result.append({
            "id"          : item.id,
            "ticker"      : item.ticker,
            "company_name": item.company_name,
            "asset_type"  : item.asset_type,
            "price"       : q.get("price", 0),
            "change"      : q.get("change", 0),
            "change_pct"  : q.get("change_pct", 0),
            "added_at"    : item.added_at.isoformat(),
        })
    return {"watchlist": result}


@app.post("/watchlist", tags=["Watchlist"])
def add_to_watchlist(
    req: WatchlistAdd,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if req.ticker not in ALL_TICKERS:
        raise HTTPException(status_code=400, detail=f"{req.ticker} not in tradeable universe.")
    existing = db.query(Watchlist).filter(
        Watchlist.user_id == current_user.id,
        Watchlist.ticker  == req.ticker,
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"{req.ticker} already in watchlist.")
    meta = ALL_TICKERS[req.ticker]
    item = Watchlist(
        user_id      = current_user.id,
        ticker       = req.ticker,
        company_name = meta.get("name"),
        asset_type   = meta.get("asset_type", "us_stocks"),
    )
    db.add(item)
    db.commit()
    return {"success": True, "ticker": req.ticker}


@app.delete("/watchlist/{ticker}", tags=["Watchlist"])
def remove_from_watchlist(
    ticker: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = db.query(Watchlist).filter(
        Watchlist.user_id == current_user.id,
        Watchlist.ticker  == ticker.upper(),
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Not in watchlist.")
    db.delete(item)
    db.commit()
    return {"success": True}


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

@app.get("/leaderboard", tags=["Leaderboard"])
def get_leaderboard(db: Session = Depends(get_db)):
    portfolios = db.query(Portfolio).all()
    entries = []
    for portfolio in portfolios:
        user = portfolio.user
        data = build_portfolio_response(db, portfolio)
        entries.append({
            "username"    : user.username,
            "display_name": user.display_name,
            "avatar_color": user.avatar_color,
            "total_value" : data["total_value"],
            "pnl"         : data["total_pnl"],
            "pnl_pct"     : data["total_pnl_pct"],
        })
    entries.sort(key=lambda x: x["total_value"], reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    return {"leaderboard": entries[:50]}


# ---------------------------------------------------------------------------
# WebSocket — Real-time price stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """
    Live price stream. Clients subscribe by connecting with their JWT token.
    Broadcasts price updates every ~15 seconds.
    Also accepts client messages: {"action": "subscribe", "tickers": ["AAPL",...]}
    """
    # Validate token
    from backend.auth import decode_token
    payload = decode_token(token)
    if not payload:
        await websocket.close(code=4001, reason="Invalid token")
        return

    await websocket.accept()
    logger.info("WS connected | user=%s", payload.get("username"))

    # Send initial price dump
    prices = get_all_cached()
    if prices:
        await websocket.send_json({"type": "prices", "data": prices})

    # Register as a broadcast subscriber
    async def broadcast(payload: dict):
        await websocket.send_json(payload)

    register_ws(broadcast)

    try:
        while True:
            # Keep connection alive and handle client messages
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                data = json.loads(msg)
                # Clients can request a specific ticker's data
                if data.get("action") == "quote" and data.get("ticker"):
                    q = await get_live_quote(data["ticker"].upper())
                    if q:
                        await websocket.send_json({"type": "quote", "data": q})
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "ping", "ts": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        logger.info("WS disconnected | user=%s", payload.get("username"))
    finally:
        unregister_ws(broadcast)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    prices = get_all_cached()
    return {
        "status"       : "ok",
        "version"      : "2.0.0",
        "prices_cached": len(prices),
        "timestamp"    : datetime.utcnow().isoformat(),
    }
