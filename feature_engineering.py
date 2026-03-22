"""
feature_engineering.py — InvestEng Feature Engineering Layer
=============================================================
Consumes clean OHLCV data from market_data.py and transforms it into
actionable financial features for the portfolio optimisation engine.

Feature Groups
--------------
  1. Return Features     — daily, log, cumulative, rolling returns
  2. Volatility Features — rolling std, annualised vol, Parkinson, Garman-Klass
  3. Momentum / Trend    — SMA, EMA, MACD, RSI, Bollinger Bands, ADX
  4. Volume Features     — OBV, VWAP, volume z-score
  5. Risk Metrics        — Sharpe, Sortino, Calmar, Max Drawdown, VaR, CVaR, Beta
  6. Cross-Asset         — rolling correlation matrix, covariance matrix

Usage
-----
    from market_data import MarketDataIngester
    from feature_engineering import FeatureEngineer

    ingester = MarketDataIngester()
    fe = FeatureEngineer(ingester)

    features = fe.build_all()            # full pipeline → persisted to DB + CSV
    risk_df  = fe.get_risk_summary()     # per-ticker risk scorecard
    corr_mat = fe.get_correlation_matrix(window=252)
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from market_data import MarketDataIngester, DB_URL, BASE_DIR

# ---------------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("InvestEng.Features")

FEATURES_CSV_DIR = BASE_DIR / "features"
FEATURES_CSV_DIR.mkdir(parents=True, exist_ok=True)

# Trading calendar constants
TRADING_DAYS_YEAR  = 252
TRADING_DAYS_MONTH = 21
RISK_FREE_RATE     = float(os.getenv("RISK_FREE_RATE", "0.05"))   # annualised, default 5%
BENCHMARK_TICKER   = os.getenv("BENCHMARK_TICKER", "^GSPC")       # S&P 500 as market benchmark


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Transforms raw OHLCV data into a rich feature DataFrame for every ticker.

    Parameters
    ----------
    ingester    : MarketDataIngester instance (provides load_from_db)
    short_window: short rolling window in trading days (default 20)
    mid_window  : medium rolling window                (default 63  ≈ 1 quarter)
    long_window : long rolling window                  (default 252 ≈ 1 year)
    """

    def __init__(
        self,
        ingester: MarketDataIngester,
        short_window: int = 20,
        mid_window: int   = 63,
        long_window: int  = 252,
    ) -> None:
        self.ingester     = ingester
        self.engine       = create_engine(DB_URL, echo=False)
        self.short_window = short_window
        self.mid_window   = mid_window
        self.long_window  = long_window

        logger.info(
            "FeatureEngineer ready | windows: short=%d mid=%d long=%d | rf=%.2f%%",
            short_window, mid_window, long_window, RISK_FREE_RATE * 100,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def build_all(self, asset_types: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Run the full feature pipeline for all (or selected) asset classes.
        Persists results to DB table `features` and per-asset-class CSVs.

        Returns the combined feature DataFrame.
        """
        asset_types = asset_types or ["us_stocks", "etfs", "crypto", "indices"]
        all_frames: list[pd.DataFrame] = []

        for asset_type in asset_types:
            logger.info("── Building features: %s", asset_type)
            df_raw = self.ingester.load_from_db(asset_type=asset_type)

            if df_raw.empty:
                logger.warning("No raw data found for %s — skipping.", asset_type)
                continue

            df_feat = self._build_ticker_features(df_raw, asset_type)

            if not df_feat.empty:
                self._save_features_csv(df_feat, asset_type)
                self._save_features_sql(df_feat)
                all_frames.append(df_feat)
                logger.info(
                    "  ✓ %s | %d rows | %d tickers | %d features",
                    asset_type,
                    len(df_feat),
                    df_feat["ticker"].nunique(),
                    len(df_feat.columns),
                )

        if not all_frames:
            logger.error("Feature engineering produced no output.")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        logger.info("✓ Feature engineering complete | total rows: %d", len(combined))
        return combined

    def build_for_tickers(self, tickers: list[str]) -> pd.DataFrame:
        """
        Build features for a specific list of tickers (cross asset-class).
        Useful for on-demand portfolio analysis.
        """
        df_raw = self.ingester.load_from_db(tickers=tickers)
        if df_raw.empty:
            logger.warning("No data found for tickers: %s", tickers)
            return pd.DataFrame()
        return self._build_ticker_features(df_raw, asset_type="mixed")

    def get_risk_summary(self, asset_type: Optional[str] = None) -> pd.DataFrame:
        """
        Returns a per-ticker risk scorecard aggregated from the features table.
        Columns: ticker | annual_return | annual_vol | sharpe | sortino |
                 max_drawdown | var_95 | cvar_95 | beta | calmar
        """
        filter_clause = f"AND asset_type = '{asset_type}'" if asset_type else ""
        query = f"""
            SELECT
                ticker,
                asset_type,
                AVG(daily_return)          * {TRADING_DAYS_YEAR}  AS annual_return,
                AVG(vol_20d)                                       AS avg_vol_20d,
                AVG(sharpe_252d)                                   AS avg_sharpe,
                AVG(sortino_252d)                                  AS avg_sortino,
                MIN(max_drawdown)                                  AS max_drawdown,
                AVG(var_95)                                        AS avg_var_95,
                AVG(cvar_95)                                       AS avg_cvar_95,
                AVG(beta_252d)                                     AS avg_beta,
                AVG(calmar_ratio)                                  AS avg_calmar
            FROM features
            WHERE 1=1 {filter_clause}
            GROUP BY ticker, asset_type
            ORDER BY avg_sharpe DESC
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    def get_correlation_matrix(
        self,
        tickers: Optional[list[str]] = None,
        window: int = 252,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Compute a rolling correlation matrix of daily returns for the given
        tickers over the most recent `window` trading days.

        Parameters
        ----------
        tickers : list of tickers; if None, uses all available
        window  : number of trailing days to use
        method  : 'pearson' | 'spearman' | 'kendall'

        Returns
        -------
        pd.DataFrame — symmetric correlation matrix (ticker × ticker)
        """
        query = "SELECT ticker, date, daily_return FROM features WHERE 1=1"
        params: dict = {}
        if tickers:
            placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
            query += f" AND ticker IN ({placeholders})"
            params = {f"t{i}": t for i, t in enumerate(tickers)}
        query += " ORDER BY date, ticker"

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params, parse_dates=["date"])

        if df.empty:
            logger.warning("No feature data for correlation matrix.")
            return pd.DataFrame()

        pivot = df.pivot(index="date", columns="ticker", values="daily_return")
        pivot = pivot.sort_index().tail(window)
        return pivot.corr(method=method)

    def get_covariance_matrix(
        self,
        tickers: Optional[list[str]] = None,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Annualised covariance matrix of daily returns.
        Used directly by the Portfolio Optimisation Engine.
        """
        corr   = self.get_correlation_matrix(tickers=tickers, window=window)
        query  = "SELECT ticker, AVG(vol_20d) AS avg_vol FROM features GROUP BY ticker"
        with self.engine.connect() as conn:
            vols = pd.read_sql(text(query), conn).set_index("ticker")["avg_vol"]

        # Σ = D · R · D  where D = diagonal matrix of annualised vols
        common = corr.index.intersection(vols.index)
        corr   = corr.loc[common, common]
        d      = vols.loc[common].values * np.sqrt(TRADING_DAYS_YEAR)
        cov    = pd.DataFrame(
            np.diag(d) @ corr.values @ np.diag(d),
            index=common, columns=common,
        )
        return cov

    def load_features(
        self,
        asset_type: Optional[str] = None,
        tickers: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load computed features from the DB with optional filters.
        Primary data source for the optimisation and backtesting layers.
        """
        query = "SELECT * FROM features WHERE 1=1"
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
            return pd.read_sql(text(query), conn, params=params, parse_dates=["date"])

    # -----------------------------------------------------------------------
    # Core Feature Pipeline
    # -----------------------------------------------------------------------

    def _build_ticker_features(self, df_raw: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """
        Apply all feature groups to every ticker in df_raw.
        Returns a long-format DataFrame sorted by (ticker, date).
        """
        # Load benchmark returns for Beta calculation
        benchmark_returns = self._get_benchmark_returns()

        results: list[pd.DataFrame] = []
        for ticker, group in df_raw.groupby("ticker"):
            try:
                df_t = group.copy().sort_values("date").reset_index(drop=True)
                df_t = self._add_return_features(df_t)
                df_t = self._add_volatility_features(df_t)
                df_t = self._add_momentum_features(df_t)
                df_t = self._add_volume_features(df_t)
                df_t = self._add_risk_metrics(df_t, benchmark_returns)
                results.append(df_t)
            except Exception as exc:
                logger.warning("Feature build failed for %s: %s", ticker, exc)
                continue

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # -----------------------------------------------------------------------
    # Feature Group 1 — Returns
    # -----------------------------------------------------------------------

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all return-based features.

        Features added
        --------------
        daily_return      : simple daily % change on adj_close
        log_return        : log(P_t / P_{t-1}) — additive over time
        cumulative_return : total return from start of series
        rolling_ret_5d    : 5-day rolling return   (1 week)
        rolling_ret_20d   : 20-day rolling return  (1 month)
        rolling_ret_63d   : 63-day rolling return  (1 quarter)
        rolling_ret_252d  : 252-day rolling return (1 year)
        excess_return     : daily_return minus daily risk-free rate
        """
        price = df["adj_close"].fillna(df["close"])

        df["daily_return"]     = price.pct_change()
        df["log_return"]       = np.log(price / price.shift(1))
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
        df["excess_return"]    = df["daily_return"] - (RISK_FREE_RATE / TRADING_DAYS_YEAR)

        for window, label in [(5, "5d"), (20, "20d"), (63, "63d"), (252, "252d")]:
            df[f"rolling_ret_{label}"] = price.pct_change(window)

        return df

    # -----------------------------------------------------------------------
    # Feature Group 2 — Volatility
    # -----------------------------------------------------------------------

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes close-to-close and range-based volatility estimators.

        Features added
        --------------
        vol_20d / vol_63d / vol_252d : annualised rolling std of daily returns
        parkinson_vol                : range-based vol (High-Low), less noise
        garman_klass_vol             : open-high-low-close estimator, most efficient
        vol_regime                   : 'low' | 'medium' | 'high' based on vol_20d percentile
        vol_ratio                    : vol_20d / vol_252d — measures vol spike vs long-term
        """
        r   = df["daily_return"]
        ann = np.sqrt(TRADING_DAYS_YEAR)

        for window, label in [(20, "20d"), (63, "63d"), (252, "252d")]:
            df[f"vol_{label}"] = r.rolling(window).std() * ann

        # Parkinson volatility (High–Low range estimator)
        hl_ratio = np.log(df["high"] / df["low"])
        df["parkinson_vol"] = (
            (1 / (4 * np.log(2)))
            * hl_ratio.pow(2)
            .rolling(self.short_window)
            .mean()
            .apply(np.sqrt)
            * ann
        )

        # Garman-Klass volatility estimator (uses O, H, L, C)
        log_hl = np.log(df["high"] / df["low"])
        log_co = np.log(df["close"] / df["open"])
        gk_daily = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        df["garman_klass_vol"] = (
            gk_daily.rolling(self.short_window).mean().apply(np.sqrt) * ann
        )

        # Volatility regime classification based on rolling percentile
        vol_pct = df["vol_20d"].rank(pct=True)
        df["vol_regime"] = pd.cut(
            vol_pct,
            bins=[0, 0.33, 0.66, 1.0],
            labels=["low", "medium", "high"],
        ).astype(str)

        # Volatility ratio: short-term spike vs long-term baseline
        df["vol_ratio"] = df["vol_20d"] / df["vol_252d"].replace(0, np.nan)

        return df

    # -----------------------------------------------------------------------
    # Feature Group 3 — Momentum & Trend Indicators
    # -----------------------------------------------------------------------

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes price-based technical momentum and trend indicators.

        Features added
        --------------
        sma_20 / sma_63 / sma_252  : simple moving averages
        ema_12 / ema_26 / ema_50   : exponential moving averages
        macd / macd_signal / macd_hist : MACD oscillator (12-26-9)
        rsi_14                     : Relative Strength Index (14-day)
        bb_upper / bb_lower / bb_mid / bb_pct : Bollinger Bands (20d, 2σ)
        bb_width                   : band width — measures volatility squeeze
        price_vs_sma20             : (close - sma_20) / sma_20
        price_vs_sma252            : 52-week price momentum signal
        momentum_1m / 3m / 12m     : raw momentum scores
        """
        price = df["adj_close"].fillna(df["close"])

        # --- Moving Averages ---
        for window, label in [(20, "20"), (63, "63"), (252, "252")]:
            df[f"sma_{label}"] = price.rolling(window).mean()

        for span, label in [(12, "12"), (26, "26"), (50, "50")]:
            df[f"ema_{label}"] = price.ewm(span=span, adjust=False).mean()

        # --- MACD (12-26-9) ---
        df["macd"]        = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # --- RSI (14-day Wilder smoothing) ---
        df["rsi_14"] = self._compute_rsi(price, period=14)

        # --- Bollinger Bands (20d, 2σ) ---
        rolling_mean = price.rolling(self.short_window).mean()
        rolling_std  = price.rolling(self.short_window).std()
        df["bb_mid"]   = rolling_mean
        df["bb_upper"] = rolling_mean + 2 * rolling_std
        df["bb_lower"] = rolling_mean - 2 * rolling_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"]   = (price - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # --- Price vs Moving Averages (normalised distance) ---
        df["price_vs_sma20"]  = (price - df["sma_20"])  / df["sma_20"]
        df["price_vs_sma252"] = (price - df["sma_252"]) / df["sma_252"]

        # --- Momentum Scores ---
        for window, label in [(20, "1m"), (63, "3m"), (252, "12m")]:
            df[f"momentum_{label}"] = price / price.shift(window) - 1

        return df

    # -----------------------------------------------------------------------
    # Feature Group 4 — Volume
    # -----------------------------------------------------------------------

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features indicating market participation and strength.

        Features added
        --------------
        volume_sma_20       : 20-day average volume
        volume_ratio        : today's volume / 20d avg — detects volume spikes
        volume_zscore       : z-score of volume (20d rolling)
        obv                 : On-Balance Volume — cumulative directional volume
        vwap_20d            : Volume-Weighted Average Price over 20 days
        price_volume_trend  : PVT — cumulative price×volume momentum
        """
        vol   = df["volume"].fillna(0)
        close = df["adj_close"].fillna(df["close"])

        # Volume moving average and ratio
        df["volume_sma_20"] = vol.rolling(self.short_window).mean()
        df["volume_ratio"]  = vol / df["volume_sma_20"].replace(0, np.nan)

        # Volume z-score (detects unusual activity)
        vol_mean = vol.rolling(self.short_window).mean()
        vol_std  = vol.rolling(self.short_window).std()
        df["volume_zscore"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # On-Balance Volume (OBV)
        direction = np.sign(close.diff()).fillna(0)
        df["obv"] = (direction * vol).cumsum()

        # VWAP over 20-day rolling window (using typical price)
        typical_price = (df["high"] + df["low"] + close) / 3
        df["vwap_20d"] = (
            (typical_price * vol).rolling(self.short_window).sum()
            / vol.rolling(self.short_window).sum()
        )

        # Price-Volume Trend (PVT)
        pvt_daily  = ((close - close.shift(1)) / close.shift(1)) * vol
        df["price_volume_trend"] = pvt_daily.cumsum()

        return df

    # -----------------------------------------------------------------------
    # Feature Group 5 — Risk Metrics
    # -----------------------------------------------------------------------

    def _add_risk_metrics(
        self, df: pd.DataFrame, benchmark_returns: Optional[pd.Series]
    ) -> pd.DataFrame:
        """
        Computes standard portfolio risk and performance metrics.

        Features added
        --------------
        sharpe_252d     : annualised Sharpe ratio (252-day rolling)
        sortino_252d    : Sortino ratio (downside deviation)
        max_drawdown    : maximum peak-to-trough drawdown at each date
        calmar_ratio    : annualised return / abs(max_drawdown)
        var_95 / var_99 : historical Value at Risk (95% and 99%)
        cvar_95         : Conditional VaR (Expected Shortfall) at 95%
        beta_252d       : rolling beta vs. S&P 500 benchmark
        alpha_252d      : Jensen's alpha (annualised)
        skewness_63d    : rolling skewness of returns
        kurtosis_63d    : rolling excess kurtosis
        """
        r   = df["daily_return"]
        rf  = RISK_FREE_RATE / TRADING_DAYS_YEAR  # daily risk-free
        ann = TRADING_DAYS_YEAR

        # ---- Sharpe Ratio (rolling 252d) ----
        excess = r - rf
        df["sharpe_252d"] = (
            excess.rolling(self.long_window).mean()
            / r.rolling(self.long_window).std()
            * np.sqrt(ann)
        )

        # ---- Sortino Ratio (rolling 252d) — penalises only downside vol ----
        downside = r.copy()
        downside[downside > 0] = 0
        df["sortino_252d"] = (
            excess.rolling(self.long_window).mean()
            / downside.rolling(self.long_window).std()
            * np.sqrt(ann)
        )

        # ---- Maximum Drawdown (expanding window — worst ever seen) ----
        price_idx = (1 + r).cumprod()
        rolling_max = price_idx.cummax()
        df["max_drawdown"] = (price_idx - rolling_max) / rolling_max

        # ---- Calmar Ratio = annualised return / |max drawdown| ----
        ann_ret = r.rolling(self.long_window).mean() * ann
        df["calmar_ratio"] = ann_ret / df["max_drawdown"].abs().replace(0, np.nan)

        # ---- Value at Risk (historical, 95% and 99%) ----
        df["var_95"] = r.rolling(self.long_window).quantile(0.05)
        df["var_99"] = r.rolling(self.long_window).quantile(0.01)

        # ---- Conditional VaR / Expected Shortfall (95%) ----
        def cvar_95(window_returns: pd.Series) -> float:
            threshold = window_returns.quantile(0.05)
            tail      = window_returns[window_returns <= threshold]
            return tail.mean() if len(tail) > 0 else np.nan

        df["cvar_95"] = r.rolling(self.long_window).apply(cvar_95, raw=False)

        # ---- Beta & Alpha vs Benchmark ----
        if benchmark_returns is not None and not benchmark_returns.empty:
            df = df.set_index("date")
            aligned_bench = benchmark_returns.reindex(df.index)

            def rolling_beta(window_r: pd.Series) -> float:
                bench_window = aligned_bench.loc[window_r.index]
                mask = bench_window.notna() & window_r.notna()
                if mask.sum() < 30:
                    return np.nan
                cov = np.cov(window_r[mask], bench_window[mask])
                return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan

            df["beta_252d"] = (
                r.reindex(df.index)
                 .rolling(self.long_window)
                 .apply(rolling_beta, raw=False)
            )

            # Jensen's Alpha = asset_return - (rf + beta * (market_return - rf))
            mkt_ann     = aligned_bench.rolling(self.long_window).mean() * ann
            df["alpha_252d"] = ann_ret - (
                RISK_FREE_RATE + df["beta_252d"] * (mkt_ann - RISK_FREE_RATE)
            )

            df = df.reset_index()
        else:
            df["beta_252d"]  = np.nan
            df["alpha_252d"] = np.nan

        # ---- Higher Moments ----
        df["skewness_63d"] = r.rolling(self.mid_window).apply(
            lambda x: stats.skew(x.dropna()), raw=False
        )
        df["kurtosis_63d"] = r.rolling(self.mid_window).apply(
            lambda x: stats.kurtosis(x.dropna()), raw=False
        )

        return df

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Wilder's RSI using EWM smoothing (standard implementation).
        Returns values in [0, 100].
        """
        delta = prices.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_benchmark_returns(self) -> Optional[pd.Series]:
        """
        Load daily returns for the benchmark ticker (S&P 500 by default)
        from the features table if available, else from raw ohlcv_data.
        """
        try:
            df_bench = self.ingester.load_from_db(tickers=[BENCHMARK_TICKER])
            if df_bench.empty:
                logger.warning("Benchmark %s not found in DB — Beta/Alpha will be NaN.", BENCHMARK_TICKER)
                return None
            df_bench = df_bench.sort_values("date")
            price    = df_bench["adj_close"].fillna(df_bench["close"])
            returns  = price.pct_change()
            returns.index = pd.to_datetime(df_bench["date"].values)
            return returns
        except Exception as exc:
            logger.warning("Could not load benchmark returns: %s", exc)
            return None

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _save_features_csv(self, df: pd.DataFrame, asset_type: str) -> None:
        """
        Save feature DataFrame as a single CSV per asset class.
        File: data/features/{asset_type}_features.csv
        """
        path = FEATURES_CSV_DIR / f"{asset_type}_features.csv"
        df["date"] = pd.to_datetime(df["date"])

        if path.exists():
            existing = pd.read_csv(path, parse_dates=["date"])
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
            combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
        else:
            combined = df

        combined.to_csv(path, index=False)
        logger.debug("Features CSV saved: %s", path)

    def _save_features_sql(self, df: pd.DataFrame) -> None:
        """
        Upsert feature rows into the `features` table in SQLite.
        Creates the table on first run via pandas `if_exists='replace'` logic
        with duplicate elimination.
        """
        records = df.copy()
        records["date"]         = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
        records["computed_at"]  = datetime.utcnow().isoformat()

        # Replace NaN/inf with None for SQL
        records = records.replace([np.inf, -np.inf], np.nan)
        records = records.where(pd.notna(records), None)

        # Create table schema on first call
        self._ensure_features_table()

        # Build upsert SQL from dynamic column list
        cols         = records.columns.tolist()
        col_str      = ", ".join(cols)
        placeholder  = ", ".join(f":{c}" for c in cols)
        upsert_sql   = f"""
            INSERT OR REPLACE INTO features ({col_str})
            VALUES ({placeholder})
        """

        with self.engine.connect() as conn:
            conn.execute(text(upsert_sql), records.to_dict(orient="records"))
            conn.commit()

        logger.debug("Features SQL upsert: %d rows", len(records))

    def _ensure_features_table(self) -> None:
        """
        Create the `features` table if it doesn't exist.
        Uses a wide schema to accommodate all computed columns.
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS features (
            ticker              TEXT    NOT NULL,
            date                TEXT    NOT NULL,
            asset_type          TEXT,
            open                REAL, high                REAL,
            low                 REAL, close               REAL,
            adj_close           REAL, volume              REAL,

            -- Returns
            daily_return        REAL, log_return          REAL,
            cumulative_return   REAL, excess_return       REAL,
            rolling_ret_5d      REAL, rolling_ret_20d     REAL,
            rolling_ret_63d     REAL, rolling_ret_252d    REAL,

            -- Volatility
            vol_20d             REAL, vol_63d             REAL,
            vol_252d            REAL, parkinson_vol       REAL,
            garman_klass_vol    REAL, vol_regime          TEXT,
            vol_ratio           REAL,

            -- Momentum & Trend
            sma_20              REAL, sma_63              REAL,
            sma_252             REAL, ema_12              REAL,
            ema_26              REAL, ema_50              REAL,
            macd                REAL, macd_signal         REAL,
            macd_hist           REAL, rsi_14              REAL,
            bb_mid              REAL, bb_upper            REAL,
            bb_lower            REAL, bb_width            REAL,
            bb_pct              REAL, price_vs_sma20      REAL,
            price_vs_sma252     REAL, momentum_1m         REAL,
            momentum_3m         REAL, momentum_12m        REAL,

            -- Volume
            volume_sma_20       REAL, volume_ratio        REAL,
            volume_zscore       REAL, obv                 REAL,
            vwap_20d            REAL, price_volume_trend  REAL,

            -- Risk
            sharpe_252d         REAL, sortino_252d        REAL,
            max_drawdown        REAL, calmar_ratio        REAL,
            var_95              REAL, var_99              REAL,
            cvar_95             REAL, beta_252d           REAL,
            alpha_252d          REAL, skewness_63d        REAL,
            kurtosis_63d        REAL,

            -- Metadata
            is_outlier          INTEGER,
            label               TEXT,
            computed_at         TEXT,

            PRIMARY KEY (ticker, date)
        )
        """
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()


# ---------------------------------------------------------------------------
# Standalone Summary Printer
# ---------------------------------------------------------------------------

def print_feature_summary(fe: FeatureEngineer, asset_type: Optional[str] = None) -> None:
    """
    Pretty-print the risk summary scorecard to stdout.
    """
    summary = fe.get_risk_summary(asset_type=asset_type)
    if summary.empty:
        print("No feature data found — run fe.build_all() first.")
        return

    pct = lambda x: f"{x * 100:+.1f}%"
    flt = lambda x: f"{x:.3f}"

    print("\n" + "=" * 90)
    print("  InvestEng — Feature Engineering Risk Scorecard")
    print(f"  Benchmark : {BENCHMARK_TICKER}  |  Risk-Free Rate: {RISK_FREE_RATE * 100:.1f}%")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    header = (
        f"  {'Ticker':<12} {'Class':<12} {'Ann.Ret':>8} {'Vol(20d)':>9} "
        f"{'Sharpe':>8} {'Sortino':>9} {'MaxDD':>9} {'VaR(95)':>9} {'Beta':>7}"
    )
    print(header)
    print("  " + "-" * 88)

    for _, row in summary.iterrows():
        print(
            f"  {str(row['ticker']):<12} {str(row['asset_type']):<12} "
            f"{pct(row['annual_return'] or 0):>8} "
            f"{pct(row['avg_vol_20d'] or 0):>9} "
            f"{flt(row['avg_sharpe'] or 0):>8} "
            f"{flt(row['avg_sortino'] or 0):>9} "
            f"{pct(row['max_drawdown'] or 0):>9} "
            f"{pct(row['avg_var_95'] or 0):>9} "
            f"{flt(row['avg_beta'] or 0):>7}"
        )
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingester = MarketDataIngester(start_date="2019-01-01")
    fe = FeatureEngineer(ingester)

    print("Running full feature engineering pipeline...")
    fe.build_all()

    print("\nRisk Scorecard — All Assets:")
    print_feature_summary(fe)

    print("\nCorrelation Matrix (US Stocks, 1-year):")
    from market_data import ASSET_UNIVERSE
    corr = fe.get_correlation_matrix(
        tickers=ASSET_UNIVERSE["us_stocks"],
        window=252,
    )
    print(corr.round(2).to_string())
