"""
market_feed.py — InvestEng Real-Time Market Feed
=================================================
Fetches live prices from yfinance every 15 seconds and broadcasts
to all connected WebSocket clients. Uses an in-memory cache to serve
instant responses and a SQLite price_cache for offline resilience.

Supported asset classes mirror the full InvestEng universe.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session

from database import SessionLocal
from models import PriceCache

logger = logging.getLogger("InvestEng.MarketFeed")

REFRESH_INTERVAL = 15   # seconds between live price polls

# Full tradeable universe — exactly what users can buy/sell
TRADEABLE_UNIVERSE = {
    "us_stocks": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corp.",
        "META": "Meta Platforms Inc.",
        "TSLA": "Tesla Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "BRK-B": "Berkshire Hathaway",
        "UNH": "UnitedHealth Group",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc.",
        "JNJ": "Johnson & Johnson",
        "WMT": "Walmart Inc.",
        "PG": "Procter & Gamble",
        "XOM": "Exxon Mobil Corp.",
        "NFLX": "Netflix Inc.",
        "AMD": "Advanced Micro Devices",
        "INTC": "Intel Corp.",
        "DIS": "Walt Disney Co.",
    },
    "etfs": {
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco Nasdaq-100 ETF",
        "VTI": "Vanguard Total Market ETF",
        "BND": "Vanguard Total Bond ETF",
        "GLD": "SPDR Gold ETF",
        "IWM": "iShares Russell 2000 ETF",
        "VEA": "Vanguard Dev. Markets ETF",
        "VWO": "Vanguard Emerging Markets ETF",
        "ARKK": "ARK Innovation ETF",
        "SQQQ": "ProShares UltraPro Short QQQ",
    },
    "crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "BNB-USD": "Binance Coin",
        "SOL-USD": "Solana",
        "XRP-USD": "Ripple",
        "ADA-USD": "Cardano",
        "DOGE-USD": "Dogecoin",
        "AVAX-USD": "Avalanche",
    },
}

ALL_TICKERS = {
    t: {"name": name, "asset_type": atype}
    for atype, group in TRADEABLE_UNIVERSE.items()
    for t, name in group.items()
}

# In-memory price store (ticker → quote dict)
_price_store: dict[str, dict] = {}
# WebSocket broadcast callbacks
_ws_subscribers: set = set()


def register_ws(callback):
    _ws_subscribers.add(callback)


def unregister_ws(callback):
    _ws_subscribers.discard(callback)


def get_cached_price(ticker: str) -> Optional[dict]:
    """Get from in-memory store first, fall back to DB cache."""
    if ticker in _price_store:
        return _price_store[ticker]
    db: Session = SessionLocal()
    try:
        row = db.query(PriceCache).filter(PriceCache.ticker == ticker).first()
        if row:
            return {
                "ticker"      : row.ticker,
                "company_name": row.company_name,
                "price"       : row.price,
                "change"      : row.change,
                "change_pct"  : row.change_pct,
                "volume"      : row.volume,
                "day_high"    : row.day_high,
                "day_low"     : row.day_low,
                "open_price"  : row.open_price,
                "market_cap"  : row.market_cap,
                "asset_type"  : ALL_TICKERS.get(ticker, {}).get("asset_type", "us_stocks"),
                "fetched_at"  : row.fetched_at.isoformat(),
            }
    finally:
        db.close()
    return None


def get_all_cached_prices() -> dict[str, dict]:
    return dict(_price_store)


def _fetch_prices(tickers: list[str]) -> dict[str, dict]:
    """Fetch live quotes for a batch of tickers via yfinance."""
    if not tickers:
        return {}
    results = {}
    try:
        data = yf.download(
            tickers=tickers,
            period="2d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        info_cache = {}
        for ticker in tickers:
            try:
                meta = ALL_TICKERS.get(ticker, {})
                # Extract latest price
                if len(tickers) == 1:
                    df = data
                else:
                    if ticker not in data.columns.get_level_values(1 if isinstance(data.columns, pd.MultiIndex) else 0):
                        continue
                    df = data.xs(ticker, axis=1, level=1) if isinstance(data.columns, pd.MultiIndex) else data

                if df.empty or len(df) < 1:
                    continue

                latest = df.iloc[-1]
                prev   = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

                price       = float(latest.get("Close", latest.get("close", 0)) or 0)
                prev_close  = float(prev.get("Close",  prev.get("close",  price)) or price)
                change      = round(price - prev_close, 4)
                change_pct  = round((change / prev_close * 100) if prev_close else 0, 4)
                volume      = float(latest.get("Volume", latest.get("volume", 0)) or 0)
                day_high    = float(latest.get("High",   latest.get("high",  price)) or price)
                day_low     = float(latest.get("Low",    latest.get("low",   price)) or price)
                open_price  = float(latest.get("Open",   latest.get("open",  price)) or price)

                if price <= 0:
                    continue

                results[ticker] = {
                    "ticker"      : ticker,
                    "company_name": meta.get("name", ticker),
                    "asset_type"  : meta.get("asset_type", "us_stocks"),
                    "price"       : price,
                    "change"      : change,
                    "change_pct"  : change_pct,
                    "volume"      : volume,
                    "day_high"    : day_high,
                    "day_low"     : day_low,
                    "open_price"  : open_price,
                    "market_cap"  : None,
                    "fetched_at"  : datetime.utcnow().isoformat(),
                }
            except Exception as e:
                logger.debug("Price extract failed for %s: %s", ticker, e)
    except Exception as e:
        logger.warning("yfinance batch fetch failed: %s", e)
    return results


def _persist_prices(prices: dict[str, dict]):
    """Write latest prices to SQLite price_cache table."""
    db: Session = SessionLocal()
    try:
        for ticker, q in prices.items():
            row = db.query(PriceCache).filter(PriceCache.ticker == ticker).first()
            if row:
                row.price        = q["price"]
                row.change       = q["change"]
                row.change_pct   = q["change_pct"]
                row.volume       = q["volume"]
                row.day_high     = q.get("day_high")
                row.day_low      = q.get("day_low")
                row.open_price   = q.get("open_price")
                row.company_name = q.get("company_name")
                row.fetched_at   = datetime.utcnow()
            else:
                db.add(PriceCache(
                    ticker       = ticker,
                    price        = q["price"],
                    change       = q["change"],
                    change_pct   = q["change_pct"],
                    volume       = q["volume"],
                    day_high     = q.get("day_high"),
                    day_low      = q.get("day_low"),
                    open_price   = q.get("open_price"),
                    company_name = q.get("company_name"),
                    fetched_at   = datetime.utcnow(),
                ))
        db.commit()
    except Exception as e:
        logger.error("Price persist error: %s", e)
        db.rollback()
    finally:
        db.close()


async def price_refresh_loop():
    """
    Background coroutine — runs forever, refreshing prices every REFRESH_INTERVAL
    seconds and broadcasting to all WebSocket subscribers.
    """
    all_tickers = list(ALL_TICKERS.keys())
    # Split into batches of 20 to avoid yfinance rate limits
    batch_size = 20
    batches = [all_tickers[i:i+batch_size] for i in range(0, len(all_tickers), batch_size)]

    logger.info("Market feed started | %d tickers | refresh=%ds", len(all_tickers), REFRESH_INTERVAL)

    while True:
        try:
            fresh = {}
            for batch in batches:
                fetched = await asyncio.get_event_loop().run_in_executor(
                    None, _fetch_prices, batch
                )
                fresh.update(fetched)

            if fresh:
                _price_store.update(fresh)
                await asyncio.get_event_loop().run_in_executor(None, _persist_prices, fresh)
                # Broadcast to all connected WebSocket clients
                if _ws_subscribers:
                    payload = {"type": "prices", "data": fresh}
                    dead = set()
                    for cb in list(_ws_subscribers):
                        try:
                            await cb(payload)
                        except Exception:
                            dead.add(cb)
                    _ws_subscribers -= dead
                logger.debug("Prices updated | %d tickers broadcast to %d clients",
                             len(fresh), len(_ws_subscribers))
        except Exception as e:
            logger.error("Price refresh loop error: %s", e)

        await asyncio.sleep(REFRESH_INTERVAL)


async def get_live_quote(ticker: str) -> Optional[dict]:
    """Fetch a single ticker's live price on demand."""
    ticker = ticker.upper()
    # Try memory cache first (< 30s stale)
    if ticker in _price_store:
        cached = _price_store[ticker]
        fetched = datetime.fromisoformat(cached["fetched_at"].replace("Z", ""))
        age = (datetime.utcnow() - fetched).total_seconds()
        if age < 30:
            return cached

    # Fresh fetch
    result = await asyncio.get_event_loop().run_in_executor(None, _fetch_prices, [ticker])
    if result and ticker in result:
        _price_store[ticker] = result[ticker]
        await asyncio.get_event_loop().run_in_executor(None, _persist_prices, result)
        return result[ticker]

    return get_cached_price(ticker)


def search_tickers(query: str, limit: int = 10) -> list[dict]:
    """Search tradeable universe by ticker or company name."""
    q = query.upper().strip()
    results = []
    for ticker, meta in ALL_TICKERS.items():
        name = meta.get("name", "")
        if q in ticker.upper() or q in name.upper():
            results.append({
                "ticker"      : ticker,
                "company_name": name,
                "asset_type"  : meta.get("asset_type", "us_stocks"),
                "exchange"    : "DEMO",
            })
    return results[:limit]
