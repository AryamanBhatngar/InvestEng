"""
market_data.py — InvestEng Data Ingestion Layer
================================================
Fetches historical OHLCV data for US Stocks, ETFs, Crypto, and Indices
using yfinance. Persists to both structured CSV files and a SQLite database
via SQLAlchemy. Designed to be imported by downstream feature engineering
and backtesting modules.

Usage:
    from market_data import MarketDataIngester
    ingester = MarketDataIngester()
    ingester.ingest_all()
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import create_engine, text, Column, String, Float, Date, DateTime
from sqlalchemy.orm import declarative_base, Session
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("InvestEng.Ingestion")


# ---------------------------------------------------------------------------
# Asset Universe
# ---------------------------------------------------------------------------

ASSET_UNIVERSE: dict[str, list[str]] = {
    "us_stocks": [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet
        "AMZN",  # Amazon
        "NVDA",  # NVIDIA
        "META",  # Meta
        "TSLA",  # Tesla
        "JPM",   # JPMorgan Chase
        "BRK-B", # Berkshire Hathaway
        "UNH",   # UnitedHealth
    ],
    "etfs": [
        "SPY",   # S&P 500 ETF
        "QQQ",   # Nasdaq-100 ETF
        "VTI",   # Total US Market ETF
        "BND",   # Total Bond Market ETF
        "GLD",   # Gold ETF
        "IWM",   # Russell 2000 ETF
        "VEA",   # Developed Markets ETF
        "VWO",   # Emerging Markets ETF
    ],
    "crypto": [
        "BTC-USD",  # Bitcoin
        "ETH-USD",  # Ethereum
        "BNB-USD",  # Binance Coin
        "SOL-USD",  # Solana
        "XRP-USD",  # Ripple
    ],
    "indices": [
        "^GSPC",    # S&P 500
        "^IXIC",    # NASDAQ Composite
        "^DJI",     # Dow Jones Industrial Average
        "^NSEI",    # NIFTY 50
        "^BSESN",   # BSE SENSEX
        "^FTSE",    # FTSE 100
        "^N225",    # Nikkei 225
    ],
}

# Readable labels mapped to yfinance ticker symbols
TICKER_LABELS: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^DJI": "Dow Jones",
    "^NSEI": "NIFTY 50",
    "^BSESN": "SENSEX",
    "^FTSE": "FTSE 100",
    "^N225": "Nikkei 225",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple",
}


# ---------------------------------------------------------------------------
# Storage Paths & DB Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(os.getenv("INVESTENG_DATA_DIR", "./data"))
CSV_DIR  = BASE_DIR / "csv"
DB_PATH  = BASE_DIR / "investeng.db"
DB_URL   = os.getenv("INVESTENG_DB_URL", f"sqlite:///{DB_PATH}")

# Create directory structure on import
for folder in [CSV_DIR / asset_type for asset_type in ASSET_UNIVERSE]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SQLAlchemy ORM Model
# ---------------------------------------------------------------------------

Base = declarative_base()


class OHLCVRecord(Base):
    """
    Represents a single daily OHLCV bar for any asset type.
    Table: ohlcv_data
    """
    __tablename__ = "ohlcv_data"

    # Composite primary key: ticker + date
    ticker      = Column(String(20),  primary_key=True, index=True)
    date        = Column(Date,        primary_key=True, index=True)
    asset_type  = Column(String(20),  nullable=False,   index=True)
    open        = Column(Float,       nullable=True)
    high        = Column(Float,       nullable=True)
    low         = Column(Float,       nullable=True)
    close       = Column(Float,       nullable=True)
    adj_close   = Column(Float,       nullable=True)
    volume      = Column(Float,       nullable=True)
    ingested_at = Column(DateTime,    default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<OHLCV {self.ticker} @ {self.date} close={self.close:.2f}>"


# ---------------------------------------------------------------------------
# Core Ingestion Class
# ---------------------------------------------------------------------------

class MarketDataIngester:
    """
    Fetches, validates, and stores OHLCV data for the full InvestEng
    asset universe. Writes to both CSV and SQLite.

    Parameters
    ----------
    start_date : str  — ISO date string, e.g. "2019-01-01"
    end_date   : str  — ISO date string, e.g. "2024-12-31" (defaults to today)
    interval   : str  — yfinance interval: "1d", "1wk", "1mo"
    batch_size : int  — number of tickers to fetch per yfinance batch call
    """

    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        batch_size: int = 10,
    ) -> None:
        self.start_date = start_date
        self.end_date   = end_date or datetime.today().strftime("%Y-%m-%d")
        self.interval   = interval
        self.batch_size = batch_size

        self.engine = create_engine(DB_URL, echo=False)
        Base.metadata.create_all(self.engine)  # DDL: create table if not exists

        logger.info(
            "MarketDataIngester initialised | %s → %s | interval=%s | db=%s",
            self.start_date, self.end_date, self.interval, DB_URL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_all(self) -> dict[str, pd.DataFrame]:
        """
        Fetch and store data for every asset class in ASSET_UNIVERSE.
        Returns a dict of {asset_type: combined_DataFrame}.
        """
        results: dict[str, pd.DataFrame] = {}
        for asset_type, tickers in ASSET_UNIVERSE.items():
            logger.info("── Starting ingestion: %s (%d tickers)", asset_type, len(tickers))
            df = self.ingest_asset_class(asset_type, tickers)
            if df is not None and not df.empty:
                results[asset_type] = df
        logger.info("✓ Ingestion complete. Asset classes processed: %d", len(results))
        return results

    def ingest_asset_class(self, asset_type: str, tickers: list[str]) -> Optional[pd.DataFrame]:
        """
        Fetch all tickers for a given asset class in batches, clean the data,
        and persist to CSV + SQL.
        """
        all_frames: list[pd.DataFrame] = []

        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i : i + self.batch_size]
            df_batch = self._fetch_batch(batch, asset_type)
            if df_batch is not None and not df_batch.empty:
                all_frames.append(df_batch)

        if not all_frames:
            logger.warning("No data returned for asset class: %s", asset_type)
            return None

        df_combined = pd.concat(all_frames, ignore_index=True)
        df_combined = self._validate(df_combined, asset_type)

        self._save_csv(df_combined, asset_type)
        self._save_sql(df_combined, asset_type)

        logger.info(
            "  ✓ %s | rows=%d | tickers=%d",
            asset_type, len(df_combined), df_combined["ticker"].nunique(),
        )
        return df_combined

    def fetch_single(self, ticker: str, asset_type: str = "us_stocks") -> Optional[pd.DataFrame]:
        """
        Convenience method to fetch, clean, and store a single ticker.
        Useful for on-demand updates without re-running the full pipeline.
        """
        logger.info("Fetching single ticker: %s [%s]", ticker, asset_type)
        df = self._fetch_batch([ticker], asset_type)
        if df is None or df.empty:
            logger.warning("No data for ticker: %s", ticker)
            return None
        df = self._validate(df, asset_type)
        self._save_csv(df, asset_type)
        self._save_sql(df, asset_type)
        return df

    def load_from_db(
        self,
        asset_type: Optional[str] = None,
        tickers: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query the SQLite database and return a DataFrame.
        Supports filtering by asset_type, tickers, and date range.
        """
        query = "SELECT * FROM ohlcv_data WHERE 1=1"
        params: dict = {}

        if asset_type:
            query += " AND asset_type = :asset_type"
            params["asset_type"] = asset_type
        if tickers:
            placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
            query += f" AND ticker IN ({placeholders})"
            params.update({f"t{i}": t for i, t in enumerate(tickers)})
        if start:
            query += " AND date >= :start"
            params["start"] = start
        if end:
            query += " AND date <= :end"
            params["end"] = end

        query += " ORDER BY ticker, date"

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params, parse_dates=["date"])

        logger.info("Loaded %d rows from DB (filter: asset_type=%s)", len(df), asset_type)
        return df

    def load_from_csv(self, asset_type: str, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Load persisted CSV data for a given asset class.
        If ticker is provided, returns data for that ticker only.
        """
        if ticker:
            path = CSV_DIR / asset_type / f"{ticker.replace('/', '_')}.csv"
            if not path.exists():
                raise FileNotFoundError(f"CSV not found: {path}")
            return pd.read_csv(path, parse_dates=["date"])

        # Load all CSVs in the asset_type folder
        folder = CSV_DIR / asset_type
        frames = []
        for csv_file in folder.glob("*.csv"):
            frames.append(pd.read_csv(csv_file, parse_dates=["date"]))
        if not frames:
            logger.warning("No CSVs found for asset class: %s", asset_type)
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _fetch_batch(self, tickers: list[str], asset_type: str) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data for a batch of tickers using yfinance.
        Returns a normalised long-format DataFrame.
        """
        try:
            raw = yf.download(
                tickers=tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                group_by="ticker",
                auto_adjust=False,    # keep Adj Close separate
                threads=True,
                progress=False,
            )
        except Exception as exc:
            logger.error("yfinance download failed for batch %s: %s", tickers, exc)
            return None

        if raw is None or raw.empty:
            logger.warning("Empty response for batch: %s", tickers)
            return None

        return self._normalise(raw, tickers, asset_type)

    def _normalise(
        self,
        raw: pd.DataFrame,
        tickers: list[str],
        asset_type: str,
    ) -> pd.DataFrame:
        """
        Convert yfinance wide-format output → long-format DataFrame with
        consistent column names regardless of single vs. multi-ticker download.
        """
        frames: list[pd.DataFrame] = []

        # yfinance returns multi-level columns for multi-ticker, flat for single
        is_multi = isinstance(raw.columns, pd.MultiIndex)

        for ticker in tickers:
            try:
                if is_multi:
                    # Slice the ticker's sub-DataFrame
                    if ticker not in raw.columns.get_level_values(1):
                        logger.debug("Ticker not in response: %s", ticker)
                        continue
                    df_t = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    df_t = raw.copy()

                df_t = df_t.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
                df_t.columns = [c.lower().replace(" ", "_") for c in df_t.columns]

                # Standardise column names (yfinance can vary)
                col_map = {
                    "open": "open", "high": "high", "low": "low",
                    "close": "close", "adj_close": "adj_close",
                    "volume": "volume",
                }
                df_t = df_t.rename(columns={k: v for k, v in col_map.items() if k in df_t.columns})

                df_t["ticker"]     = ticker
                df_t["asset_type"] = asset_type
                df_t["label"]      = TICKER_LABELS.get(ticker, ticker)

                # Keep only the columns we need
                keep = ["date", "ticker", "asset_type", "label",
                        "open", "high", "low", "close", "adj_close", "volume"]
                df_t = df_t[[c for c in keep if c in df_t.columns]]

                frames.append(df_t)

            except Exception as exc:
                logger.warning("Failed to normalise ticker %s: %s", ticker, exc)
                continue

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _validate(self, df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """
        Data quality layer:
          1. Drop rows where close price is NaN
          2. Remove duplicate (ticker, date) pairs — keep last
          3. Cap extreme outliers (price > 3σ from ticker mean)
          4. Ensure date column is datetime
          5. Sort by ticker, then date
        """
        initial_len = len(df)

        # 1. Drop missing close prices
        df = df.dropna(subset=["close"])

        # 2. Remove duplicates
        df = df.drop_duplicates(subset=["ticker", "date"], keep="last")

        # 3. Outlier detection — flag but don't remove (keep audit trail)
        #    For crypto, high volatility is normal, so we use a wider window.
        sigma = 5 if asset_type == "crypto" else 4
        df["_mean"]  = df.groupby("ticker")["close"].transform("mean")
        df["_std"]   = df.groupby("ticker")["close"].transform("std")
        df["is_outlier"] = (
            (df["close"] - df["_mean"]).abs() > sigma * df["_std"]
        ).astype(int)
        df = df.drop(columns=["_mean", "_std"])

        outlier_count = df["is_outlier"].sum()
        if outlier_count > 0:
            logger.warning(
                "%s: %d outlier rows flagged (not removed, check manually)",
                asset_type, outlier_count,
            )

        # 4. Ensure date is proper datetime
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # 5. Sort
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info("%s: %d rows dropped during validation", asset_type, dropped)

        return df

    def _save_csv(self, df: pd.DataFrame, asset_type: str) -> None:
        """
        Save one CSV file per ticker under data/csv/{asset_type}/{TICKER}.csv.
        Appends to existing files to support incremental updates.
        """
        folder = CSV_DIR / asset_type
        for ticker, group in df.groupby("ticker"):
            safe_name = str(ticker).replace("/", "_").replace("^", "IDX_")
            path = folder / f"{safe_name}.csv"

            if path.exists():
                existing = pd.read_csv(path, parse_dates=["date"])
                combined = pd.concat([existing, group], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last")
                combined = combined.sort_values("date").reset_index(drop=True)
            else:
                combined = group

            combined.to_csv(path, index=False)

        logger.debug("CSV saved: %s (%d files)", asset_type, df["ticker"].nunique())

    def _save_sql(self, df: pd.DataFrame, asset_type: str) -> None:
        """
        Upsert rows into the ohlcv_data table.
        SQLite doesn't support native upsert via pandas, so we use
        INSERT OR REPLACE via raw SQL for efficiency.
        """
        records = df.copy()
        records["date"]        = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
        records["ingested_at"] = datetime.utcnow().isoformat()

        # Ensure all expected columns exist (fill missing with None)
        sql_cols = ["ticker", "date", "asset_type", "open", "high", "low",
                    "close", "adj_close", "volume", "ingested_at"]
        for col in sql_cols:
            if col not in records.columns:
                records[col] = None
        records = records[sql_cols]

        # Replace NaN with None for SQL compatibility
        records = records.where(pd.notna(records), None)

        insert_sql = """
            INSERT OR REPLACE INTO ohlcv_data
                (ticker, date, asset_type, open, high, low, close, adj_close, volume, ingested_at)
            VALUES
                (:ticker, :date, :asset_type, :open, :high, :low, :close, :adj_close, :volume, :ingested_at)
        """

        with Session(self.engine) as session:
            session.execute(text(insert_sql), records.to_dict(orient="records"))
            session.commit()

        logger.debug("SQL upsert complete: %s (%d rows)", asset_type, len(records))


# ---------------------------------------------------------------------------
# Summary / Diagnostics
# ---------------------------------------------------------------------------

def get_ingestion_summary(ingester: MarketDataIngester) -> pd.DataFrame:
    """
    Returns a DataFrame summarising what's in the database:
    asset_type | ticker | first_date | last_date | rows | has_outliers
    """
    query = """
        SELECT
            asset_type,
            ticker,
            MIN(date)               AS first_date,
            MAX(date)               AS last_date,
            COUNT(*)                AS total_rows,
            SUM(is_outlier)         AS outlier_rows
        FROM ohlcv_data
        GROUP BY asset_type, ticker
        ORDER BY asset_type, ticker
    """
    with ingester.engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def print_summary(ingester: MarketDataIngester) -> None:
    """Pretty-print the ingestion summary to stdout."""
    summary = get_ingestion_summary(ingester)
    if summary.empty:
        print("No data found in the database.")
        return

    print("\n" + "=" * 72)
    print(f"  InvestEng — Market Data Ingestion Summary")
    print(f"  Database  : {DB_URL}")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    for asset_type, group in summary.groupby("asset_type"):
        print(f"\n  [{asset_type.upper()}]  {len(group)} tickers")
        print(f"  {'Ticker':<12} {'First Date':<14} {'Last Date':<14} {'Rows':>8} {'Outliers':>10}")
        print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*8} {'-'*10}")
        for _, row in group.iterrows():
            print(
                f"  {row['ticker']:<12} {str(row['first_date']):<14} "
                f"{str(row['last_date']):<14} {int(row['total_rows']):>8} "
                f"{int(row['outlier_rows'] or 0):>10}"
            )

    total = summary["total_rows"].sum()
    print(f"\n  Total rows across all assets: {int(total):,}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry Point (run standalone for a full ingest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingester = MarketDataIngester(
        start_date="2019-01-01",
        interval="1d",
    )
    ingester.ingest_all()
    print_summary(ingester)
