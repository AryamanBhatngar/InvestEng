"""
backtesting.py — InvestEng Walk-Forward Backtesting Engine
===========================================================
Simulates portfolio strategies historically using a walk-forward
methodology — the same approach used by quantitative hedge funds.

How It Works
------------
  1. Divide the full history into a rolling train window + test window
  2. On each rebalance date, re-optimise weights on the TRAIN window
  3. Apply those weights forward through the TEST window (out-of-sample)
  4. Roll the windows forward and repeat
  5. Stitch all out-of-sample periods into a single equity curve

This avoids look-ahead bias: the optimiser only ever sees data that
would have been available at that point in time.

Key Outputs
-----------
  BacktestResult       — full equity curve, per-period returns, all metrics
  PerformanceReport    — human-readable scorecard vs benchmark
  run_backtest()       — top-level convenience function

Performance Metrics Computed
-----------------------------
  CAGR, Volatility, Sharpe, Sortino, Calmar, Max Drawdown,
  Max Drawdown Duration, VaR 95/99, CVaR 95, Beta, Alpha,
  Information Ratio, Tracking Error, Win Rate, Avg Win/Loss,
  Profit Factor, Recovery Factor, Ulcer Index, Omega Ratio

Usage
-----
    from market_data import MarketDataIngester
    from feature_engineering import FeatureEngineer
    from portfolio_optimizer import PortfolioOptimizer, OptimisationConstraints
    from backtesting import Backtester, run_backtest

    ingester  = MarketDataIngester()
    fe        = FeatureEngineer(ingester)
    optimizer = PortfolioOptimizer(fe)

    result = run_backtest(
        optimizer   = optimizer,
        tickers     = ["AAPL","MSFT","GOOGL","BTC-USD","GLD","BND","SPY"],
        strategy    = "max_sharpe",
        start_date  = "2020-01-01",
        end_date    = "2024-12-31",
        rebalance   = "quarterly",
        train_years = 1,
    )
    print(result.report())
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine, text

from market_data import MarketDataIngester, DB_URL, BASE_DIR
from feature_engineering import FeatureEngineer, RISK_FREE_RATE, TRADING_DAYS_YEAR, BENCHMARK_TICKER
from portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioResult,
    OptimisationConstraints,
    RiskProfile,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("InvestEng.Backtest")

BACKTEST_DIR = BASE_DIR / "backtests"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

REBALANCE_FREQ = {
    "monthly"   : 1,
    "quarterly" : 3,
    "semi-annual": 6,
    "annual"    : 12,
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Full configuration for a single backtest run.

    Attributes
    ----------
    tickers         : asset universe to include
    strategy        : optimisation strategy name
    start_date      : first date of the out-of-sample period
    end_date        : last date of the backtest
    train_years     : years of history used to optimise on each window
    rebalance       : 'monthly' | 'quarterly' | 'semi-annual' | 'annual'
    initial_capital : starting portfolio value in USD
    transaction_cost: one-way cost per trade as fraction (e.g. 0.001 = 0.1%)
    slippage        : additional cost modelling market impact (fraction)
    benchmark       : ticker to compare against (default: ^GSPC)
    constraints     : OptimisationConstraints passed to the optimiser
    """
    tickers         : list[str]
    strategy        : str          = "max_sharpe"
    start_date      : str          = "2020-01-01"
    end_date        : str          = datetime.today().strftime("%Y-%m-%d")
    train_years     : int          = 1
    rebalance       : str          = "quarterly"
    initial_capital : float        = 100_000.0
    transaction_cost: float        = 0.001
    slippage        : float        = 0.0005
    benchmark       : str          = BENCHMARK_TICKER
    constraints     : OptimisationConstraints = field(
        default_factory=OptimisationConstraints
    )


@dataclass
class RebalanceEvent:
    """Records a single rebalance event in the walk-forward loop."""
    date            : date
    train_start     : date
    train_end       : date
    tickers_used    : list[str]
    weights_before  : dict[str, float]
    weights_after   : dict[str, float]
    turnover        : float          # fraction of portfolio traded
    transaction_cost: float          # cost in portfolio % terms
    strategy        : str
    solver_status   : str


@dataclass
class BacktestResult:
    """
    Complete output of a backtesting run.

    Attributes
    ----------
    config          : full BacktestConfig used
    equity_curve    : daily portfolio value series (index=date)
    returns         : daily portfolio return series
    benchmark_curve : benchmark equity curve (same base as portfolio)
    weights_history : DataFrame of weights at each rebalance date
    rebalance_log   : list of RebalanceEvent records
    metrics         : dict of all performance statistics
    drawdown_series : daily drawdown series (%)
    run_id          : unique identifier for this run
    """
    config           : BacktestConfig
    equity_curve     : pd.Series
    returns          : pd.Series
    benchmark_curve  : pd.Series
    weights_history  : pd.DataFrame
    rebalance_log    : list[RebalanceEvent]
    metrics          : dict
    drawdown_series  : pd.Series
    run_id           : str = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))

    def report(self) -> str:
        """Render a full human-readable performance report."""
        m   = self.metrics
        cfg = self.config
        pct = lambda v: f"{v * 100:+.2f}%" if v is not None else "  N/A"
        flt = lambda v: f"{v:.3f}"          if v is not None else "  N/A"
        dur = lambda v: f"{int(v)} days"    if v is not None else "  N/A"

        lines = [
            "",
            "=" * 72,
            "  InvestEng — Backtest Performance Report",
            f"  Strategy    : {cfg.strategy.replace('_',' ').title()}",
            f"  Universe    : {len(cfg.tickers)} assets",
            f"  Period      : {cfg.start_date}  →  {cfg.end_date}",
            f"  Rebalance   : {cfg.rebalance.title()}  |  Train window: {cfg.train_years}yr",
            f"  Capital     : ${cfg.initial_capital:,.0f}  |  Tx cost: {cfg.transaction_cost*100:.2f}%",
            f"  Benchmark   : {cfg.benchmark}",
            f"  Run ID      : {self.run_id}",
            "=" * 72,
            "",
            "  ── Returns ─────────────────────────────────────────",
            f"  CAGR                   : {pct(m.get('cagr'))}",
            f"  Total Return           : {pct(m.get('total_return'))}",
            f"  Best Year              : {pct(m.get('best_year'))}",
            f"  Worst Year             : {pct(m.get('worst_year'))}",
            f"  Avg Monthly Return     : {pct(m.get('avg_monthly_return'))}",
            f"  Win Rate (daily)       : {pct(m.get('win_rate'))}",
            "",
            "  ── Risk ────────────────────────────────────────────",
            f"  Annual Volatility      : {pct(m.get('annual_vol'))}",
            f"  Max Drawdown           : {pct(m.get('max_drawdown'))}",
            f"  Max Drawdown Duration  : {dur(m.get('max_dd_duration'))}",
            f"  VaR 95 (daily)         : {pct(m.get('var_95'))}",
            f"  CVaR 95 (daily)        : {pct(m.get('cvar_95'))}",
            f"  Ulcer Index            : {flt(m.get('ulcer_index'))}",
            "",
            "  ── Risk-Adjusted ───────────────────────────────────",
            f"  Sharpe Ratio           : {flt(m.get('sharpe'))}",
            f"  Sortino Ratio          : {flt(m.get('sortino'))}",
            f"  Calmar Ratio           : {flt(m.get('calmar'))}",
            f"  Omega Ratio            : {flt(m.get('omega'))}",
            f"  Recovery Factor        : {flt(m.get('recovery_factor'))}",
            "",
            "  ── Benchmark Relative ──────────────────────────────",
            f"  Benchmark CAGR         : {pct(m.get('benchmark_cagr'))}",
            f"  Alpha (annualised)     : {pct(m.get('alpha'))}",
            f"  Beta                   : {flt(m.get('beta'))}",
            f"  Information Ratio      : {flt(m.get('information_ratio'))}",
            f"  Tracking Error         : {pct(m.get('tracking_error'))}",
            f"  Up-Capture Ratio       : {pct(m.get('up_capture'))}",
            f"  Down-Capture Ratio     : {pct(m.get('down_capture'))}",
            "",
            "  ── Trading ─────────────────────────────────────────",
            f"  Rebalance Events       : {m.get('n_rebalances')}",
            f"  Avg Turnover           : {pct(m.get('avg_turnover'))}",
            f"  Total Tx Cost          : {pct(m.get('total_tx_cost'))}",
            f"  Profit Factor          : {flt(m.get('profit_factor'))}",
            "",
            "  ── Final Portfolio ─────────────────────────────────",
            f"  Final Value            : ${m.get('final_value', 0):,.2f}",
            f"  Total P&L              : ${m.get('total_pnl', 0):,.2f}",
            "=" * 72,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Walk-forward backtesting engine.

    Walk-Forward Logic
    ------------------
    Given start_date S and train_years T:

      Window 1:  train=[S-T, S]        test=[S, S+rebalance_months]
      Window 2:  train=[S-T+r, S+r]    test=[S+r, S+2r]
      ...

    At each window, weights are optimised on train data only,
    then applied (with costs) to generate returns on test data.
    """

    def __init__(self, optimizer: PortfolioOptimizer) -> None:
        self.optimizer = optimizer
        self.fe        = optimizer.fe
        self.engine    = create_engine(DB_URL, echo=False)
        self._ensure_backtest_table()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self, config: BacktestConfig) -> BacktestResult:
        """
        Execute a full walk-forward backtest and return a BacktestResult.
        """
        logger.info(
            "Starting backtest | strategy=%s | %s → %s | rebalance=%s",
            config.strategy, config.start_date, config.end_date, config.rebalance,
        )

        # ── Step 1: Load full return matrix ─────────────────────────────
        returns_matrix = self._load_returns_matrix(
            tickers    = config.tickers,
            start_date = self._shift_date(config.start_date, -config.train_years),
            end_date   = config.end_date,
        )
        benchmark_returns = self._load_benchmark_returns(
            config.benchmark, config.start_date, config.end_date
        )

        if returns_matrix.empty:
            raise RuntimeError("No return data available. Run fe.build_all() first.")

        # ── Step 2: Generate rebalance schedule ─────────────────────────
        rebalance_dates = self._get_rebalance_dates(
            returns_matrix.index,
            config.start_date,
            config.end_date,
            config.rebalance,
        )
        logger.info("Rebalance schedule: %d events", len(rebalance_dates))

        # ── Step 3: Walk-forward loop ────────────────────────────────────
        portfolio_returns : list[pd.Series] = []
        rebalance_log     : list[RebalanceEvent] = []
        weights_records   : list[dict] = []
        current_weights   : Optional[dict[str, float]] = None
        total_tx_cost     : float = 0.0

        for i, (rebal_start, rebal_end) in enumerate(
            zip(rebalance_dates[:-1], rebalance_dates[1:])
        ):
            train_end   = rebal_start
            train_start_dt = self._shift_date(
                train_end.strftime("%Y-%m-%d"), -config.train_years
            )
            train_start = pd.Timestamp(train_start_dt)

            logger.info(
                "  Window %d/%d | train: %s→%s | test: %s→%s",
                i + 1, len(rebalance_dates) - 1,
                train_start.date(), train_end.date(),
                rebal_start.date(), rebal_end.date(),
            )

            # ── Re-optimise on train window ──────────────────────────
            train_data = returns_matrix.loc[
                (returns_matrix.index >= train_start) &
                (returns_matrix.index < rebal_start)
            ]

            new_weights = self._optimise_window(
                train_data, config.strategy, config.constraints, config.tickers
            )

            if new_weights is None:
                new_weights = current_weights or self._equal_weights(config.tickers)
                logger.warning("  Optimisation failed on window %d — holding previous weights", i + 1)

            # ── Compute turnover and transaction cost ────────────────
            weights_before = current_weights or {}
            turnover = self._compute_turnover(weights_before, new_weights)
            tx_cost  = turnover * (config.transaction_cost + config.slippage)
            total_tx_cost += tx_cost

            # ── Log rebalance event ──────────────────────────────────
            rebalance_log.append(RebalanceEvent(
                date             = rebal_start.date(),
                train_start      = train_start.date(),
                train_end        = train_end.date(),
                tickers_used     = list(new_weights.keys()),
                weights_before   = weights_before,
                weights_after    = new_weights,
                turnover         = turnover,
                transaction_cost = tx_cost,
                strategy         = config.strategy,
                solver_status    = "optimal",
            ))

            weights_record = {"date": rebal_start.date(), **new_weights}
            weights_records.append(weights_record)
            current_weights = new_weights

            # ── Apply weights to out-of-sample test window ───────────
            test_data = returns_matrix.loc[
                (returns_matrix.index >= rebal_start) &
                (returns_matrix.index < rebal_end)
            ]

            if test_data.empty:
                continue

            period_returns = self._apply_weights(test_data, new_weights)

            # Deduct transaction cost on rebalance day
            if len(period_returns) > 0:
                period_returns.iloc[0] -= tx_cost

            portfolio_returns.append(period_returns)

        # ── Step 4: Stitch equity curve ──────────────────────────────────
        if not portfolio_returns:
            raise RuntimeError("No out-of-sample periods generated.")

        all_returns = pd.concat(portfolio_returns).sort_index()
        all_returns = all_returns[~all_returns.index.duplicated(keep="first")]

        equity_curve = (1 + all_returns).cumprod() * config.initial_capital
        drawdown     = self._compute_drawdown(equity_curve)

        # Align benchmark
        bench_aligned = benchmark_returns.reindex(all_returns.index).fillna(0)
        bench_curve   = (1 + bench_aligned).cumprod() * config.initial_capital

        weights_df = pd.DataFrame(weights_records).set_index("date") if weights_records else pd.DataFrame()

        # ── Step 5: Compute all metrics ──────────────────────────────────
        metrics = self._compute_metrics(
            returns          = all_returns,
            equity_curve     = equity_curve,
            benchmark_returns= bench_aligned,
            drawdown         = drawdown,
            rebalance_log    = rebalance_log,
            total_tx_cost    = total_tx_cost,
            initial_capital  = config.initial_capital,
        )

        result = BacktestResult(
            config          = config,
            equity_curve    = equity_curve,
            returns         = all_returns,
            benchmark_curve = bench_curve,
            weights_history = weights_df,
            rebalance_log   = rebalance_log,
            metrics         = metrics,
            drawdown_series = drawdown,
        )

        self._save_backtest(result)
        logger.info(
            "✓ Backtest complete | CAGR=%.2f%% | Sharpe=%.2f | MaxDD=%.2f%%",
            metrics.get("cagr", 0) * 100,
            metrics.get("sharpe", 0),
            metrics.get("max_drawdown", 0) * 100,
        )
        return result

    def compare_strategies(
        self,
        config    : BacktestConfig,
        strategies: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Run the same backtest across multiple strategies and return a
        side-by-side metrics comparison DataFrame.
        """
        strategies = strategies or ["max_sharpe", "min_variance", "risk_parity"]
        records = []
        for strategy in strategies:
            cfg = BacktestConfig(**{**config.__dict__, "strategy": strategy})
            try:
                result = self.run(cfg)
                m = result.metrics
                records.append({
                    "strategy"           : strategy,
                    "cagr"               : m.get("cagr"),
                    "annual_vol"         : m.get("annual_vol"),
                    "sharpe"             : m.get("sharpe"),
                    "sortino"            : m.get("sortino"),
                    "max_drawdown"       : m.get("max_drawdown"),
                    "calmar"             : m.get("calmar"),
                    "alpha"              : m.get("alpha"),
                    "beta"               : m.get("beta"),
                    "information_ratio"  : m.get("information_ratio"),
                    "win_rate"           : m.get("win_rate"),
                    "total_tx_cost"      : m.get("total_tx_cost"),
                    "final_value"        : m.get("final_value"),
                })
            except Exception as exc:
                logger.warning("Strategy %s failed: %s", strategy, exc)
        return pd.DataFrame(records).set_index("strategy")

    def sensitivity_analysis(
        self,
        config       : BacktestConfig,
        param        : str,
        values       : list,
    ) -> pd.DataFrame:
        """
        Test how a single config parameter affects performance.

        Parameters
        ----------
        param   : config attribute to vary, e.g. 'rebalance', 'train_years',
                  'transaction_cost', 'initial_capital'
        values  : list of values to test

        Returns
        -------
        DataFrame indexed by param value, columns = key metrics
        """
        records = []
        for v in values:
            cfg_dict = {k: getattr(config, k) for k in config.__dataclass_fields__}
            cfg_dict[param] = v
            cfg = BacktestConfig(**cfg_dict)
            try:
                result = self.run(cfg)
                m = result.metrics
                records.append({
                    param       : v,
                    "cagr"      : m.get("cagr"),
                    "sharpe"    : m.get("sharpe"),
                    "max_dd"    : m.get("max_drawdown"),
                    "sortino"   : m.get("sortino"),
                    "calmar"    : m.get("calmar"),
                    "final_val" : m.get("final_value"),
                })
                logger.info("Sensitivity | %s=%s | CAGR=%.2f%% Sharpe=%.2f",
                            param, v, (m.get("cagr") or 0) * 100, m.get("sharpe") or 0)
            except Exception as exc:
                logger.warning("Sensitivity param=%s value=%s failed: %s", param, v, exc)
        return pd.DataFrame(records).set_index(param)

    def load_history(self) -> pd.DataFrame:
        """Load all past backtest summary rows from the DB."""
        with self.engine.connect() as conn:
            return pd.read_sql(
                text("SELECT * FROM backtest_runs ORDER BY run_timestamp DESC"), conn
            )

    # -----------------------------------------------------------------------
    # Walk-Forward Helpers
    # -----------------------------------------------------------------------

    def _optimise_window(
        self,
        train_returns : pd.DataFrame,
        strategy      : str,
        constraints   : OptimisationConstraints,
        tickers       : list[str],
    ) -> Optional[dict[str, float]]:
        """
        Re-optimise on a train window by injecting custom mu/Sigma
        computed from the training data directly (avoids DB round-trip).
        """
        try:
            available = [t for t in tickers if t in train_returns.columns]
            if len(available) < 2:
                return None

            df = train_returns[available].dropna(how="all", axis=1)
            available = df.columns.tolist()
            if len(available) < 2:
                return None

            # Compute mu and Sigma from train window directly
            mu_daily = df.mean().values
            mu_ann   = mu_daily * TRADING_DAYS_YEAR

            cov_daily = df.cov().values
            Sigma_ann = cov_daily * TRADING_DAYS_YEAR
            Sigma_ann += np.eye(len(available)) * 1e-8

            # Patch the optimizer's _prepare_inputs to use these directly
            self.optimizer._last_tickers = available
            weights_raw, status = self._dispatch_strategy(
                strategy, mu_ann, Sigma_ann, len(available), constraints
            )
            weights = self.optimizer._clean_weights(weights_raw, constraints)
            return dict(zip(available, weights))

        except Exception as exc:
            logger.warning("Window optimisation failed: %s", exc)
            return None

    def _dispatch_strategy(
        self,
        strategy    : str,
        mu          : np.ndarray,
        Sigma       : np.ndarray,
        n           : int,
        constraints : OptimisationConstraints,
    ) -> tuple[np.ndarray, str]:
        """Route to the correct strategy method on PortfolioOptimizer."""
        opt = self.optimizer
        dispatch = {
            "max_sharpe"  : lambda: opt._max_sharpe(mu, Sigma, n, constraints),
            "min_variance": lambda: opt._min_variance(Sigma, n, constraints),
            "mean_variance": lambda: opt._mean_variance(mu, Sigma, n, constraints),
            "risk_parity" : lambda: opt._risk_parity(Sigma, n, constraints),
        }
        fn = dispatch.get(strategy, dispatch["max_sharpe"])
        return fn()

    def _apply_weights(
        self,
        test_returns : pd.DataFrame,
        weights      : dict[str, float],
    ) -> pd.Series:
        """
        Compute daily portfolio returns for the test window.
        Handles missing tickers by reallocating proportionally.
        """
        available = {t: w for t, w in weights.items() if t in test_returns.columns}
        if not available:
            return pd.Series(0.0, index=test_returns.index)

        total_wt = sum(available.values())
        norm_wt  = {t: w / total_wt for t, w in available.items()}

        port_ret = pd.Series(0.0, index=test_returns.index)
        for ticker, wt in norm_wt.items():
            port_ret += test_returns[ticker].fillna(0) * wt

        return port_ret

    def _compute_turnover(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> float:
        """
        Compute one-way turnover = 0.5 * Σ |w_new - w_old|.
        A turnover of 0.1 means 10% of the portfolio was traded.
        """
        all_tickers = set(old_weights) | set(new_weights)
        turnover = sum(
            abs(new_weights.get(t, 0) - old_weights.get(t, 0))
            for t in all_tickers
        )
        return turnover * 0.5

    def _equal_weights(self, tickers: list[str]) -> dict[str, float]:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    # -----------------------------------------------------------------------
    # Data Loading
    # -----------------------------------------------------------------------

    def _load_returns_matrix(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Load daily returns for all tickers as a wide pivot table (date × ticker).
        Falls back to computing from OHLCV if features table is empty.
        """
        placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
        query = f"""
            SELECT date, ticker, daily_return
            FROM features
            WHERE ticker IN ({placeholders})
              AND date >= :start AND date <= :end
            ORDER BY date, ticker
        """
        params = {f"t{i}": t for i, t in enumerate(tickers)}
        params["start"] = start_date
        params["end"]   = end_date

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params, parse_dates=["date"])

        if df.empty:
            # Fallback: compute from raw OHLCV
            logger.warning("Features table empty — computing returns from raw OHLCV")
            df = self._returns_from_ohlcv(tickers, start_date, end_date)

        pivot = df.pivot(index="date", columns="ticker", values="daily_return")
        pivot = pivot.sort_index()
        logger.info(
            "Return matrix: %d days × %d tickers | %s → %s",
            len(pivot), pivot.shape[1],
            pivot.index.min().date(), pivot.index.max().date(),
        )
        return pivot

    def _returns_from_ohlcv(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Compute pct_change returns directly from the ohlcv_data table."""
        placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
        query = f"""
            SELECT date, ticker, adj_close, close
            FROM ohlcv_data
            WHERE ticker IN ({placeholders})
              AND date >= :start AND date <= :end
            ORDER BY ticker, date
        """
        params = {f"t{i}": t for i, t in enumerate(tickers)}
        params["start"] = start_date
        params["end"]   = end_date

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params, parse_dates=["date"])

        if df.empty:
            return pd.DataFrame()

        df["price"] = df["adj_close"].fillna(df["close"])
        df["daily_return"] = df.groupby("ticker")["price"].pct_change()
        return df[["date", "ticker", "daily_return"]].dropna()

    def _load_benchmark_returns(
        self, benchmark: str, start_date: str, end_date: str
    ) -> pd.Series:
        """Load benchmark daily returns as a Series indexed by date."""
        df = self._load_returns_matrix([benchmark], start_date, end_date)
        if df.empty or benchmark not in df.columns:
            logger.warning("Benchmark %s not found — using zeros", benchmark)
            return pd.Series(dtype=float)
        return df[benchmark].fillna(0)

    # -----------------------------------------------------------------------
    # Performance Metrics
    # -----------------------------------------------------------------------

    def _compute_metrics(
        self,
        returns           : pd.Series,
        equity_curve      : pd.Series,
        benchmark_returns : pd.Series,
        drawdown          : pd.Series,
        rebalance_log     : list[RebalanceEvent],
        total_tx_cost     : float,
        initial_capital   : float,
    ) -> dict:
        """
        Compute the full set of performance metrics from the equity curve
        and return series.
        """
        r   = returns.dropna()
        n   = len(r)
        rf  = RISK_FREE_RATE / TRADING_DAYS_YEAR

        if n < 2:
            return {}

        # ── Returns ─────────────────────────────────────────────────────
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        n_years      = n / TRADING_DAYS_YEAR
        cagr         = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        # Annual returns by calendar year
        annual_rets = (1 + r).resample("YE").prod() - 1
        best_year   = float(annual_rets.max()) if len(annual_rets) > 0 else None
        worst_year  = float(annual_rets.min()) if len(annual_rets) > 0 else None

        monthly_rets    = (1 + r).resample("ME").prod() - 1
        avg_monthly_ret = float(monthly_rets.mean()) if len(monthly_rets) > 0 else None

        win_rate     = float((r > 0).mean())
        avg_win      = float(r[r > 0].mean()) if (r > 0).any() else 0.0
        avg_loss     = float(r[r < 0].mean()) if (r < 0).any() else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        # ── Volatility & Drawdown ────────────────────────────────────────
        annual_vol   = float(r.std() * np.sqrt(TRADING_DAYS_YEAR))
        max_drawdown = float(drawdown.min())
        dd_duration  = self._max_drawdown_duration(drawdown)

        # ── Risk-Adjusted ────────────────────────────────────────────────
        excess_r = r - rf
        sharpe   = float(excess_r.mean() / r.std() * np.sqrt(TRADING_DAYS_YEAR)) if r.std() > 0 else 0.0

        downside = r.copy()
        downside[downside > 0] = 0
        sortino = (
            float(excess_r.mean() / downside.std() * np.sqrt(TRADING_DAYS_YEAR))
            if downside.std() > 0 else 0.0
        )

        calmar = float(cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        # VaR / CVaR
        var_95  = float(r.quantile(0.05))
        var_99  = float(r.quantile(0.01))
        cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95

        # Ulcer Index (RMS of drawdown)
        ulcer_index = float(np.sqrt((drawdown ** 2).mean()))

        # Omega Ratio (probability-weighted ratio of gains to losses vs threshold)
        threshold = rf
        gains  = (r - threshold).clip(lower=0)
        losses = (threshold - r).clip(lower=0)
        omega  = float(gains.sum() / losses.sum()) if losses.sum() > 0 else np.inf

        recovery_factor = float(abs(total_return / max_drawdown)) if max_drawdown != 0 else np.inf

        # ── Benchmark-Relative ───────────────────────────────────────────
        bench = benchmark_returns.reindex(r.index).fillna(0)
        bench_total   = float((1 + bench).prod() - 1)
        bench_n_years = len(bench) / TRADING_DAYS_YEAR
        bench_cagr    = float((1 + bench_total) ** (1 / bench_n_years) - 1) if bench_n_years > 0 else 0.0

        active_ret     = r - bench
        tracking_error = float(active_ret.std() * np.sqrt(TRADING_DAYS_YEAR))
        info_ratio     = float(active_ret.mean() / active_ret.std() * np.sqrt(TRADING_DAYS_YEAR)) \
                         if active_ret.std() > 0 else 0.0

        # Beta & Alpha (OLS)
        if bench.std() > 0:
            cov_mat = np.cov(r.values, bench.values)
            beta    = float(cov_mat[0, 1] / cov_mat[1, 1])
            alpha   = float(cagr - (RISK_FREE_RATE + beta * (bench_cagr - RISK_FREE_RATE)))
        else:
            beta, alpha = 1.0, 0.0

        # Up/Down capture ratios
        up_mask   = bench > 0
        down_mask = bench < 0
        up_capture   = float(r[up_mask].mean()   / bench[up_mask].mean())   if up_mask.any()   else 1.0
        down_capture = float(r[down_mask].mean()  / bench[down_mask].mean()) if down_mask.any() else 1.0

        # ── Trading ──────────────────────────────────────────────────────
        avg_turnover = float(
            np.mean([e.turnover for e in rebalance_log]) if rebalance_log else 0
        )
        final_value = float(equity_curve.iloc[-1])

        return {
            # Returns
            "total_return"    : total_return,
            "cagr"            : cagr,
            "best_year"       : best_year,
            "worst_year"      : worst_year,
            "avg_monthly_return": avg_monthly_ret,
            "win_rate"        : win_rate,
            "avg_win"         : avg_win,
            "avg_loss"        : avg_loss,
            "profit_factor"   : profit_factor,
            # Risk
            "annual_vol"      : annual_vol,
            "max_drawdown"    : max_drawdown,
            "max_dd_duration" : dd_duration,
            "var_95"          : var_95,
            "var_99"          : var_99,
            "cvar_95"         : cvar_95,
            "ulcer_index"     : ulcer_index,
            # Risk-adjusted
            "sharpe"          : sharpe,
            "sortino"         : sortino,
            "calmar"          : calmar,
            "omega"           : omega,
            "recovery_factor" : recovery_factor,
            # Benchmark
            "benchmark_cagr"  : bench_cagr,
            "alpha"           : alpha,
            "beta"            : beta,
            "information_ratio": info_ratio,
            "tracking_error"  : tracking_error,
            "up_capture"      : up_capture,
            "down_capture"    : down_capture,
            # Trading
            "n_rebalances"    : len(rebalance_log),
            "avg_turnover"    : avg_turnover,
            "total_tx_cost"   : total_tx_cost,
            # Summary
            "final_value"     : final_value,
            "total_pnl"       : final_value - self.optimizer.fe.ingester.engine.__class__.__name__ and
                                float(equity_curve.iloc[-1] - equity_curve.iloc[0]),
        }

    def _compute_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Compute the drawdown series: (current - peak) / peak."""
        rolling_max = equity_curve.cummax()
        return (equity_curve - rolling_max) / rolling_max

    def _max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """
        Compute the longest period (in days) the portfolio spent underwater.
        """
        is_underwater  = drawdown < 0
        max_duration   = 0
        current_streak = 0
        for val in is_underwater:
            if val:
                current_streak += 1
                max_duration = max(max_duration, current_streak)
            else:
                current_streak = 0
        return max_duration

    # -----------------------------------------------------------------------
    # Date Utilities
    # -----------------------------------------------------------------------

    def _get_rebalance_dates(
        self,
        index       : pd.DatetimeIndex,
        start_date  : str,
        end_date    : str,
        frequency   : str,
    ) -> list[pd.Timestamp]:
        """
        Generate rebalance timestamps aligned to actual trading days.
        """
        months        = REBALANCE_FREQ.get(frequency, 3)
        start_ts      = pd.Timestamp(start_date)
        end_ts        = pd.Timestamp(end_date)
        trading_days  = index[(index >= start_ts) & (index <= end_ts)]

        if len(trading_days) == 0:
            raise RuntimeError(f"No trading days found between {start_date} and {end_date}")

        dates = [trading_days[0]]
        cursor = dates[0]
        while True:
            next_dt = cursor + pd.DateOffset(months=months)
            future  = trading_days[trading_days >= next_dt]
            if future.empty:
                break
            dates.append(future[0])
            cursor = future[0]

        if trading_days[-1] not in dates:
            dates.append(trading_days[-1])

        return dates

    @staticmethod
    def _shift_date(date_str: str, years: int) -> str:
        """Shift an ISO date string by N years."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        shifted = dt + relativedelta(years=years)
        return shifted.strftime("%Y-%m-%d")

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _ensure_backtest_table(self) -> None:
        create_sql = """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id          TEXT PRIMARY KEY,
            strategy        TEXT,
            tickers         TEXT,
            start_date      TEXT,
            end_date        TEXT,
            rebalance       TEXT,
            train_years     INTEGER,
            cagr            REAL,
            annual_vol      REAL,
            sharpe          REAL,
            sortino         REAL,
            max_drawdown    REAL,
            calmar          REAL,
            alpha           REAL,
            beta            REAL,
            information_ratio REAL,
            win_rate        REAL,
            total_tx_cost   REAL,
            final_value     REAL,
            n_rebalances    INTEGER,
            run_timestamp   TEXT
        )
        """
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()

    def _save_backtest(self, result: BacktestResult) -> None:
        """Save backtest summary to DB and equity curve + weights to CSV."""
        m   = result.metrics
        cfg = result.config

        record = {
            "run_id"           : result.run_id,
            "strategy"         : cfg.strategy,
            "tickers"          : json.dumps(cfg.tickers),
            "start_date"       : cfg.start_date,
            "end_date"         : cfg.end_date,
            "rebalance"        : cfg.rebalance,
            "train_years"      : cfg.train_years,
            "cagr"             : m.get("cagr"),
            "annual_vol"       : m.get("annual_vol"),
            "sharpe"           : m.get("sharpe"),
            "sortino"          : m.get("sortino"),
            "max_drawdown"     : m.get("max_drawdown"),
            "calmar"           : m.get("calmar"),
            "alpha"            : m.get("alpha"),
            "beta"             : m.get("beta"),
            "information_ratio": m.get("information_ratio"),
            "win_rate"         : m.get("win_rate"),
            "total_tx_cost"    : m.get("total_tx_cost"),
            "final_value"      : m.get("final_value"),
            "n_rebalances"     : m.get("n_rebalances"),
            "run_timestamp"    : datetime.utcnow().isoformat(),
        }

        sql = """
            INSERT OR REPLACE INTO backtest_runs
            (run_id, strategy, tickers, start_date, end_date, rebalance, train_years,
             cagr, annual_vol, sharpe, sortino, max_drawdown, calmar, alpha, beta,
             information_ratio, win_rate, total_tx_cost, final_value, n_rebalances, run_timestamp)
            VALUES
            (:run_id, :strategy, :tickers, :start_date, :end_date, :rebalance, :train_years,
             :cagr, :annual_vol, :sharpe, :sortino, :max_drawdown, :calmar, :alpha, :beta,
             :information_ratio, :win_rate, :total_tx_cost, :final_value, :n_rebalances, :run_timestamp)
        """
        with self.engine.connect() as conn:
            conn.execute(text(sql), record)
            conn.commit()

        # Equity curve CSV
        eq_path = BACKTEST_DIR / f"{result.run_id}_equity.csv"
        pd.DataFrame({
            "date"           : result.equity_curve.index,
            "portfolio_value": result.equity_curve.values,
            "benchmark_value": result.benchmark_curve.reindex(result.equity_curve.index).values,
            "daily_return"   : result.returns.reindex(result.equity_curve.index).values,
            "drawdown"       : result.drawdown_series.reindex(result.equity_curve.index).values,
        }).to_csv(eq_path, index=False)

        # Weights history CSV
        if not result.weights_history.empty:
            wt_path = BACKTEST_DIR / f"{result.run_id}_weights.csv"
            result.weights_history.to_csv(wt_path)

        logger.info("Backtest saved | run_id=%s | equity_csv=%s", result.run_id, eq_path)


# ---------------------------------------------------------------------------
# Convenience Entry Point
# ---------------------------------------------------------------------------

def run_backtest(
    optimizer    : PortfolioOptimizer,
    tickers      : list[str],
    strategy     : str   = "max_sharpe",
    start_date   : str   = "2020-01-01",
    end_date     : str   = datetime.today().strftime("%Y-%m-%d"),
    rebalance    : str   = "quarterly",
    train_years  : int   = 1,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    benchmark    : str   = BENCHMARK_TICKER,
    constraints  : Optional[OptimisationConstraints] = None,
) -> BacktestResult:
    """
    One-line convenience wrapper around Backtester.run().
    Creates a BacktestConfig, instantiates Backtester, and runs.
    """
    config = BacktestConfig(
        tickers          = tickers,
        strategy         = strategy,
        start_date       = start_date,
        end_date         = end_date,
        train_years      = train_years,
        rebalance        = rebalance,
        initial_capital  = initial_capital,
        transaction_cost = transaction_cost,
        benchmark        = benchmark,
        constraints      = constraints or OptimisationConstraints(),
    )
    backtester = Backtester(optimizer)
    return backtester.run(config)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingester  = MarketDataIngester(start_date="2018-01-01")
    fe        = FeatureEngineer(ingester)
    optimizer = PortfolioOptimizer(fe)

    universe = (
        ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM"]
        + ["SPY", "QQQ", "GLD", "BND"]
        + ["BTC-USD", "ETH-USD"]
    )

    # ── Single strategy run ──────────────────────────────────────────────
    result = run_backtest(
        optimizer       = optimizer,
        tickers         = universe,
        strategy        = "max_sharpe",
        start_date      = "2020-01-01",
        rebalance       = "quarterly",
        train_years     = 1,
        initial_capital = 100_000,
        transaction_cost= 0.001,
    )
    print(result.report())

    # ── Strategy comparison ──────────────────────────────────────────────
    backtester = Backtester(optimizer)
    config     = BacktestConfig(tickers=universe, start_date="2020-01-01")
    comparison = backtester.compare_strategies(config)
    print("\nStrategy Comparison:\n", comparison.to_string())

    # ── Sensitivity: rebalance frequency ────────────────────────────────
    sensitivity = backtester.sensitivity_analysis(
        config = config,
        param  = "rebalance",
        values = ["monthly", "quarterly", "semi-annual", "annual"],
    )
    print("\nRebalance Frequency Sensitivity:\n", sensitivity.to_string())
