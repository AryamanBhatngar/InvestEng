"""
api.py — InvestEng FastAPI Backend
====================================
REST API that exposes the full InvestEng pipeline as HTTP endpoints.
Designed to be consumed by the Streamlit dashboard and any external client.

Endpoints
---------
  POST /ingest                — trigger data ingestion
  POST /features/build        — run feature engineering
  GET  /features/risk-summary — per-ticker risk scorecard
  GET  /features/correlation  — correlation matrix
  POST /portfolio/optimise    — run portfolio optimisation
  POST /portfolio/compare     — compare all strategies
  GET  /portfolio/frontier    — efficient frontier curve
  POST /backtest/run          — run walk-forward backtest
  POST /backtest/compare      — multi-strategy backtest comparison
  POST /backtest/sensitivity  — sensitivity analysis
  GET  /backtest/history      — past backtest runs
  GET  /universe              — available tickers per asset class
  GET  /health                — liveness check

Run with:
    uvicorn api:app --reload --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from market_data import MarketDataIngester, ASSET_UNIVERSE
from feature_engineering import FeatureEngineer
from portfolio_optimizer import (
    PortfolioOptimizer,
    OptimisationConstraints,
    RiskProfiler,
)
from backtesting import Backtester, BacktestConfig, run_backtest

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("InvestEng.API")


# ---------------------------------------------------------------------------
# App-wide singletons (initialised once at startup)
# ---------------------------------------------------------------------------

_ingester  : Optional[MarketDataIngester] = None
_fe        : Optional[FeatureEngineer]    = None
_optimizer : Optional[PortfolioOptimizer] = None
_backtester: Optional[Backtester]         = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy singletons once at startup, tear down on shutdown."""
    global _ingester, _fe, _optimizer, _backtester
    logger.info("InvestEng API starting up...")
    _ingester   = MarketDataIngester(start_date="2019-01-01")
    _fe         = FeatureEngineer(_ingester)
    _optimizer  = PortfolioOptimizer(_fe)
    _backtester = Backtester(_optimizer)
    logger.info("All engine components initialised.")
    yield
    logger.info("InvestEng API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "InvestEng API",
    description = "Intelligent Investment Engine — Data → Features → Optimisation → Backtest",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    asset_types : list[str] = Field(
        default=["us_stocks", "etfs", "crypto", "indices"],
        description="Asset classes to ingest",
    )
    start_date  : str = Field(default="2019-01-01", description="ISO date string")
    interval    : str = Field(default="1d", description="yfinance interval")

class FeaturesRequest(BaseModel):
    asset_types: list[str] = Field(default=["us_stocks", "etfs", "crypto", "indices"])

class RiskSummaryRequest(BaseModel):
    asset_type: Optional[str] = None

class CorrelationRequest(BaseModel):
    tickers: Optional[list[str]] = None
    window : int = Field(default=252, ge=20, le=1260)
    method : str = Field(default="pearson", pattern="^(pearson|spearman|kendall)$")

class ConstraintsModel(BaseModel):
    min_weight : float = Field(default=0.01, ge=0.0, le=0.5)
    max_weight : float = Field(default=0.40, ge=0.05, le=1.0)
    long_only  : bool  = True
    leverage   : float = Field(default=1.0,  ge=0.5, le=2.0)

class OptimiseRequest(BaseModel):
    tickers       : list[str] = Field(min_length=2)
    strategy      : str = Field(
        default="max_sharpe",
        pattern="^(max_sharpe|min_variance|mean_variance|risk_parity|black_litterman)$",
    )
    constraints   : ConstraintsModel = Field(default_factory=ConstraintsModel)
    target_return : Optional[float]  = None
    views         : Optional[dict[str, float]] = None

class ProfileOptimiseRequest(BaseModel):
    tickers         : list[str] = Field(min_length=2)
    risk_score      : int = Field(ge=0, le=100, description="Risk score 0–100")
    views           : Optional[dict[str, float]] = None

class FrontierRequest(BaseModel):
    tickers     : list[str] = Field(min_length=2)
    n_points    : int = Field(default=40, ge=10, le=100)
    constraints : ConstraintsModel = Field(default_factory=ConstraintsModel)

class BacktestRequest(BaseModel):
    tickers          : list[str] = Field(min_length=2)
    strategy         : str  = Field(
        default="max_sharpe",
        pattern="^(max_sharpe|min_variance|mean_variance|risk_parity)$",
    )
    start_date       : str  = Field(default="2020-01-01")
    end_date         : str  = Field(default=datetime.today().strftime("%Y-%m-%d"))
    rebalance        : str  = Field(
        default="quarterly",
        pattern="^(monthly|quarterly|semi-annual|annual)$",
    )
    train_years      : int  = Field(default=1, ge=1, le=5)
    initial_capital  : float= Field(default=100_000, ge=1_000)
    transaction_cost : float= Field(default=0.001, ge=0.0, le=0.05)
    slippage         : float= Field(default=0.0005, ge=0.0, le=0.01)
    benchmark        : str  = Field(default="^GSPC")
    constraints      : ConstraintsModel = Field(default_factory=ConstraintsModel)

class SensitivityRequest(BaseModel):
    base_config : BacktestRequest
    param       : str
    values      : list[Any] = Field(min_length=2)


# ---------------------------------------------------------------------------
# Helper: clean numpy / NaN for JSON serialisation
# ---------------------------------------------------------------------------

def _clean(obj: Any) -> Any:
    """Recursively make an object JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return _clean(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    df = df.copy().reset_index()
    df.columns = [str(c) for c in df.columns]
    return _clean(df.where(pd.notna(df), None).to_dict(orient="records"))


def _constraints_from_model(c: ConstraintsModel) -> OptimisationConstraints:
    return OptimisationConstraints(
        min_weight = c.min_weight,
        max_weight = c.max_weight,
        long_only  = c.long_only,
        leverage   = c.leverage,
    )


# ---------------------------------------------------------------------------
# Routes — Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    return {
        "status"   : "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version"  : "1.0.0",
        "components": {
            "ingester"  : _ingester  is not None,
            "features"  : _fe        is not None,
            "optimizer" : _optimizer is not None,
            "backtester": _backtester is not None,
        },
    }


@app.get("/universe", tags=["System"])
async def get_universe():
    """Return the full asset universe by class."""
    return {"universe": ASSET_UNIVERSE, "total_tickers": sum(len(v) for v in ASSET_UNIVERSE.values())}


# ---------------------------------------------------------------------------
# Routes — Data Ingestion
# ---------------------------------------------------------------------------

@app.post("/ingest", tags=["Ingestion"])
async def ingest_data(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger market data ingestion for specified asset classes.
    Runs in the background — returns immediately with a job ID.
    """
    job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def _run_ingest():
        ingester = MarketDataIngester(
            start_date=req.start_date, interval=req.interval
        )
        for at in req.asset_types:
            tickers = ASSET_UNIVERSE.get(at, [])
            if tickers:
                ingester.ingest_asset_class(at, tickers)
        logger.info("Background ingestion complete | job_id=%s", job_id)

    background_tasks.add_task(_run_ingest)
    return {
        "job_id"     : job_id,
        "status"     : "accepted",
        "asset_types": req.asset_types,
        "message"    : "Ingestion running in background. Check /health for status.",
    }


@app.get("/ingest/summary", tags=["Ingestion"])
async def ingestion_summary():
    """Return what's currently in the OHLCV database."""
    try:
        from market_data import get_ingestion_summary
        summary = get_ingestion_summary(_ingester)
        return {"summary": _df_to_records(summary)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Routes — Features
# ---------------------------------------------------------------------------

@app.post("/features/build", tags=["Features"])
async def build_features(req: FeaturesRequest, background_tasks: BackgroundTasks):
    """Trigger full feature engineering pipeline (background task)."""
    job_id = datetime.utcnow().strftime("feat_%Y%m%d_%H%M%S")

    def _run():
        _fe.build_all(asset_types=req.asset_types)
        logger.info("Feature build complete | job_id=%s", job_id)

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "accepted",
            "message": "Feature engineering running in background."}


@app.post("/features/risk-summary", tags=["Features"])
async def risk_summary(req: RiskSummaryRequest):
    """Per-ticker risk scorecard sorted by Sharpe ratio."""
    try:
        df = _fe.get_risk_summary(asset_type=req.asset_type)
        if df.empty:
            raise HTTPException(status_code=404, detail="No feature data found. Run /features/build first.")
        return {"risk_summary": _df_to_records(df), "count": len(df)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/correlation", tags=["Features"])
async def correlation_matrix(req: CorrelationRequest):
    """Rolling correlation matrix of daily returns."""
    try:
        corr = _fe.get_correlation_matrix(
            tickers=req.tickers, window=req.window, method=req.method
        )
        if corr.empty:
            raise HTTPException(status_code=404, detail="No data for correlation matrix.")
        return {
            "tickers"    : corr.columns.tolist(),
            "matrix"     : _clean(corr.values.tolist()),
            "window_days": req.window,
            "method"     : req.method,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Routes — Portfolio Optimisation
# ---------------------------------------------------------------------------

@app.post("/portfolio/optimise", tags=["Optimisation"])
async def optimise_portfolio(req: OptimiseRequest):
    """
    Run a single portfolio optimisation.
    Returns weights, metrics, allocation breakdown, and explanation.
    """
    try:
        result = _optimizer.optimise(
            tickers       = req.tickers,
            strategy      = req.strategy,
            constraints   = _constraints_from_model(req.constraints),
            target_return = req.target_return,
            views         = req.views,
        )
        return {
            "strategy"        : result.strategy,
            "solver_status"   : result.solver_status,
            "timestamp"       : result.timestamp.isoformat(),
            "metrics": {
                "expected_return" : _clean(result.expected_return),
                "volatility"      : _clean(result.volatility),
                "sharpe_ratio"    : _clean(result.sharpe_ratio),
                "diversification" : _clean(result.diversification),
            },
            "allocation"      : _df_to_records(result.allocation_df),
            "weights"         : _clean(dict(zip(result.tickers, result.weights.tolist()))),
            "risk_contributions": _clean(dict(zip(result.tickers, result.risk_contributions.tolist()))),
            "explanation"     : result.explanation,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/optimise-profile", tags=["Optimisation"])
async def optimise_for_profile(req: ProfileOptimiseRequest):
    """
    Optimise using a risk score (0–100) to auto-select strategy and constraints.
    """
    try:
        profile = RiskProfiler.from_score(req.risk_score)
        result  = _optimizer.optimise_for_profile(
            tickers=req.tickers, profile=profile, views=req.views
        )
        return {
            "risk_profile": {
                "score"      : profile.score,
                "label"      : profile.label,
                "strategy"   : profile.strategy,
                "target_vol" : profile.target_vol,
                "max_crypto" : profile.max_crypto,
            },
            "strategy"   : result.strategy,
            "metrics": {
                "expected_return": _clean(result.expected_return),
                "volatility"     : _clean(result.volatility),
                "sharpe_ratio"   : _clean(result.sharpe_ratio),
                "diversification": _clean(result.diversification),
            },
            "allocation" : _df_to_records(result.allocation_df),
            "weights"    : _clean(dict(zip(result.tickers, result.weights.tolist()))),
            "explanation": result.explanation,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/compare", tags=["Optimisation"])
async def compare_strategies(req: OptimiseRequest):
    """Run all strategies on the same universe and return a comparison table."""
    try:
        df = _optimizer.compare_strategies(
            tickers     = req.tickers,
            constraints = _constraints_from_model(req.constraints),
        )
        return {"comparison": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/frontier", tags=["Optimisation"])
async def efficient_frontier(req: FrontierRequest):
    """Trace the efficient frontier (risk/return curve)."""
    try:
        df = _optimizer.efficient_frontier(
            tickers     = req.tickers,
            n_points    = req.n_points,
            constraints = _constraints_from_model(req.constraints),
        )
        if df.empty:
            raise HTTPException(status_code=404, detail="Could not compute efficient frontier.")
        return {
            "frontier"       : _df_to_records(df),
            "n_points"       : len(df),
            "return_range"   : [_clean(df["portfolio_return"].min()), _clean(df["portfolio_return"].max())],
            "vol_range"      : [_clean(df["volatility"].min()), _clean(df["volatility"].max())],
            "max_sharpe_point": _clean(df.loc[df["sharpe"].idxmax()].to_dict()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Routes — Backtesting
# ---------------------------------------------------------------------------

@app.post("/backtest/run", tags=["Backtesting"])
async def run_backtest_endpoint(req: BacktestRequest):
    """
    Execute a full walk-forward backtest.
    Returns equity curve, metrics, drawdown series, and rebalance log.
    """
    try:
        config = BacktestConfig(
            tickers          = req.tickers,
            strategy         = req.strategy,
            start_date       = req.start_date,
            end_date         = req.end_date,
            rebalance        = req.rebalance,
            train_years      = req.train_years,
            initial_capital  = req.initial_capital,
            transaction_cost = req.transaction_cost,
            slippage         = req.slippage,
            benchmark        = req.benchmark,
            constraints      = _constraints_from_model(req.constraints),
        )
        result = _backtester.run(config)

        # Equity curve as {date: value} records
        eq_records = [
            {
                "date"            : str(d.date() if hasattr(d, "date") else d),
                "portfolio_value" : _clean(v),
                "benchmark_value" : _clean(result.benchmark_curve.get(d)),
                "drawdown"        : _clean(result.drawdown_series.get(d)),
                "daily_return"    : _clean(result.returns.get(d)),
            }
            for d, v in result.equity_curve.items()
        ]

        # Rebalance log
        reb_log = [
            {
                "date"            : str(e.date),
                "turnover"        : _clean(e.turnover),
                "transaction_cost": _clean(e.transaction_cost),
                "tickers_used"    : e.tickers_used,
                "weights_after"   : _clean(e.weights_after),
            }
            for e in result.rebalance_log
        ]

        return {
            "run_id"         : result.run_id,
            "strategy"       : config.strategy,
            "metrics"        : _clean(result.metrics),
            "equity_curve"   : eq_records,
            "rebalance_log"  : reb_log,
            "report_text"    : result.report(),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest/compare", tags=["Backtesting"])
async def compare_backtests(req: BacktestRequest):
    """Run max_sharpe, min_variance, and risk_parity backtests and compare."""
    try:
        config = BacktestConfig(
            tickers          = req.tickers,
            start_date       = req.start_date,
            end_date         = req.end_date,
            rebalance        = req.rebalance,
            train_years      = req.train_years,
            initial_capital  = req.initial_capital,
            transaction_cost = req.transaction_cost,
            constraints      = _constraints_from_model(req.constraints),
        )
        df = _backtester.compare_strategies(config)
        return {"comparison": _df_to_records(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest/sensitivity", tags=["Backtesting"])
async def sensitivity_analysis(req: SensitivityRequest):
    """Vary one config param and measure impact on key metrics."""
    try:
        base = req.base_config
        config = BacktestConfig(
            tickers          = base.tickers,
            strategy         = base.strategy,
            start_date       = base.start_date,
            end_date         = base.end_date,
            rebalance        = base.rebalance,
            train_years      = base.train_years,
            initial_capital  = base.initial_capital,
            transaction_cost = base.transaction_cost,
            constraints      = _constraints_from_model(base.constraints),
        )
        df = _backtester.sensitivity_analysis(
            config=config, param=req.param, values=req.values
        )
        return {"sensitivity": _df_to_records(df), "param": req.param}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/history", tags=["Backtesting"])
async def backtest_history():
    """Return all historical backtest runs from the database."""
    try:
        df = _backtester.load_history()
        return {"history": _df_to_records(df), "count": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
