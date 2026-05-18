# 📈 InvestEng — Real-Time Paper Trading Platform

Trade with **$100,000 in demo money** at **live market prices**.  
Full-stack application: FastAPI backend + React frontend + real-time WebSocket prices.

---

## Features

| Feature | Details |
|---|---|
| 💰 Demo Balance | $100,000 virtual cash per account |
| 📡 Live Prices | Real market prices via yfinance, refreshed every 15 seconds |
| ⚡ WebSocket | Push price updates to every connected client instantly |
| 📊 Portfolio | Real-time P&L, unrealised gains, position tracking |
| 📈 Equity Curve | Automatic hourly portfolio value snapshots |
| 🏆 Leaderboard | Compete with other traders by portfolio return |
| 👁 Watchlist | Track instruments without trading them |
| 📋 Order History | Full audit trail of every trade |
| 🔒 Auth | JWT-based, sessions last 7 days |

**Tradeable Universe:** 20 US stocks · 10 ETFs · 8 Crypto pairs

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo>
cd investeng
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and change SECRET_KEY
```

### 3. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will:
- Create the SQLite database automatically on first run
- Start fetching live market prices in the background
- Serve the frontend at `http://localhost:8000/app`

### 4. Open the app

```
http://localhost:8000/app
```

Or open `frontend/index.html` directly in a browser (ensure backend is running on port 8000).

---

## Architecture

```
investeng/
├── backend/
│   ├── main.py           ← FastAPI app, all routes, WebSocket
│   ├── models.py         ← SQLAlchemy ORM (users, portfolios, positions, orders)
│   ├── schemas.py        ← Pydantic request/response models
│   ├── auth.py           ← JWT auth, password hashing
│   ├── market_feed.py    ← yfinance live price feed, WS broadcast
│   ├── paper_trading.py  ← Order execution, position management, P&L
│   └── database.py       ← SQLAlchemy engine, session factory
├── frontend/
│   └── index.html        ← React SPA (CDN imports, no build step needed)
├── data/
│   └── investeng.db      ← SQLite database (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Get JWT token |
| GET | `/auth/me` | Current user |
| GET | `/market/prices` | All cached live prices |
| GET | `/market/quote/{ticker}` | Single live quote |
| GET | `/market/search?q=` | Search tickers |
| GET | `/market/movers` | Top gainers/losers |
| GET | `/portfolio` | Full portfolio with live P&L |
| GET | `/portfolio/history` | Equity curve snapshots |
| GET | `/portfolio/orders` | Order history |
| POST | `/trade` | Execute BUY or SELL |
| GET | `/watchlist` | User watchlist |
| POST | `/watchlist` | Add to watchlist |
| DELETE | `/watchlist/{ticker}` | Remove from watchlist |
| GET | `/leaderboard` | Top traders |
| WS | `/ws/{token}` | Real-time price stream |

Interactive docs: `http://localhost:8000/docs`

---

## Trade Rules

- **Max position size:** 40% of portfolio value in any single asset
- **Fractional shares:** Supported (buy 0.001 BTC for example)
- **Commission:** $0 (demo mode)
- **Slippage:** Configurable via `DEMO_SLIPPAGE_BPS` (default 0)
- **Execution:** Instant fill at live market price

---

## Production Deployment

1. Set `DATABASE_URL` to a PostgreSQL connection string
2. Set a strong random `SECRET_KEY`
3. Run with: `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1`
   (Single worker required for in-memory WebSocket state)
4. Put Nginx in front for HTTPS

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + SQLAlchemy + SQLite/PostgreSQL |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| Market Data | yfinance (real prices, no API key needed) |
| Real-time | WebSocket (FastAPI native) |
| Frontend | React 18 + Recharts (CDN, no build step) |
| Fonts | Syne + DM Mono (Google Fonts) |
