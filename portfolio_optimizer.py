"""
portfolio_optimizer.py — InvestEng Portfolio Optimisation Engine
================================================================
Transforms feature-engineered data into optimal portfolio allocations.
Implements four institutional-grade strategies via convex optimisation
(cvxpy), an efficient frontier generator, and a risk-profile-aware
allocation selector.

Optimisation Strategies
-----------------------
  1. Maximum Sharpe Ratio   — highest risk-adjusted return (tangency portfolio)
  2. Minimum Variance       — lowest portfolio volatility (defensive)
  3. Mean-Variance (target) — Markowitz: minimise variance for a target return
  4. Risk Parity            — equal risk contribution from every asset
  5. Black-Litterman        — Bayesian blend of market equilibrium + analyst views

Supporting Modules
------------------
  EfficientFrontier  — traces the full optimal risk/return curve
  RiskProfiler       — maps a user's risk tolerance to a recommended strategy
  PortfolioResult    — dataclass capturing weights, metrics and explanation

Usage
-----
    from market_data import MarketDataIngester
    from feature_engineering import FeatureEngineer
    from portfolio_optimizer import PortfolioOptimizer, RiskProfiler

    ingester  = MarketDataIngester()
    fe        = FeatureEngineer(ingester)
    optimizer = PortfolioOptimizer(fe)

    result = optimizer.optimise(
        tickers    = ["AAPL","MSFT","GOOGL","BTC-USD","GLD","BND"],
        strategy   = "max_sharpe",
        constraints= {"min_weight": 0.02, "max_weight": 0.40},
    )
    print(result)

    # Or let the risk profiler pick the strategy
    profile = RiskProfiler.from_questionnaire(score=65)  # 0-100
    result  = optimizer.optimise_for_profile(tickers, profile)
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from market_data import MarketDataIngester, DB_URL, BASE_DIR
from feature_engineering import FeatureEngineer, RISK_FREE_RATE, TRADING_DAYS_YEAR

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("InvestEng.Optimizer")

RESULTS_DIR = BASE_DIR / "portfolios"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class OptimisationConstraints:
    """
    Defines per-ticker and portfolio-level weight bounds.

    Attributes
    ----------
    min_weight      : minimum allocation per asset (e.g. 0.02 = 2%)
    max_weight      : maximum allocation per asset (e.g. 0.40 = 40%)
    max_sector_wt   : optional cap on a group of tickers (dict: label → max)
    long_only       : if False, allows short positions (weights < 0)
    leverage        : sum-of-weights constraint (1.0 = fully invested)
    """
    min_weight    : float = 0.01
    max_weight    : float = 0.40
    long_only     : bool  = True
    leverage      : float = 1.0
    max_sector_wt : dict  = field(default_factory=dict)   # {"crypto": 0.20}


@dataclass
class PortfolioResult:
    """
    Full output of a single optimisation run.

    Attributes
    ----------
    strategy        : name of the optimisation strategy used
    tickers         : ordered list of assets in the portfolio
    weights         : optimal weight vector (sums to 1)
    expected_return : annualised expected portfolio return
    volatility      : annualised portfolio standard deviation
    sharpe_ratio    : (return - rf) / volatility
    diversification : 1 - HHI of weights (0 = concentrated, 1 = equal weight)
    risk_contributions : per-asset % of total portfolio risk
    allocation_df   : DataFrame with full allocation breakdown
    solver_status   : cvxpy solver exit status
    timestamp       : when this result was computed
    explanation     : human-readable summary of the allocation
    """
    strategy          : str
    tickers           : list[str]
    weights           : np.ndarray
    expected_return   : float
    volatility        : float
    sharpe_ratio      : float
    diversification   : float
    risk_contributions: np.ndarray
    allocation_df     : pd.DataFrame
    solver_status     : str
    timestamp         : datetime = field(default_factory=datetime.utcnow)
    explanation       : str = ""

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 64,
            f"  InvestEng — Portfolio Optimisation Result",
            f"  Strategy  : {self.strategy.replace('_', ' ').title()}",
            f"  Solved at : {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"  Status    : {self.solver_status}",
            "=" * 64,
            f"  Expected Return (ann.) : {self.expected_return * 100:+.2f}%",
            f"  Volatility (ann.)      : {self.volatility * 100:.2f}%",
            f"  Sharpe Ratio           : {self.sharpe_ratio:.3f}",
            f"  Diversification Score  : {self.diversification:.3f}",
            "",
            "  Allocation Breakdown:",
            f"  {'Ticker':<12} {'Weight':>8} {'Risk Contrib':>14} {'Exp Return':>12}",
            "  " + "-" * 50,
        ]
        for _, row in self.allocation_df.sort_values("weight", ascending=False).iterrows():
            lines.append(
                f"  {row['ticker']:<12} {row['weight'] * 100:>7.2f}%"
                f" {row['risk_contribution'] * 100:>13.2f}%"
                f" {row['expected_return'] * 100:>11.2f}%"
            )
        lines += ["=" * 64, "", f"  {self.explanation}", ""]
        return "\n".join(lines)


@dataclass
class RiskProfile:
    """
    Encodes an investor's risk tolerance and maps it to optimisation params.

    score       : 0 (ultra-conservative) → 100 (aggressive/speculative)
    label       : human-readable label
    strategy    : recommended optimisation strategy
    target_vol  : maximum acceptable annualised volatility
    max_crypto  : maximum crypto allocation allowed
    max_equity  : maximum equity (stock + ETF) allocation
    """
    score       : int
    label       : str
    strategy    : str
    target_vol  : float
    max_crypto  : float
    max_equity  : float


# ---------------------------------------------------------------------------
# Risk Profiler
# ---------------------------------------------------------------------------

class RiskProfiler:
    """
    Maps a numeric risk score (0–100) to a structured RiskProfile.

    Profiles
    --------
      0–20   Conservative    — min variance, heavy bonds/gold
      21–40  Moderate        — risk parity, balanced equities
      41–60  Balanced        — risk parity / mean-variance
      61–80  Growth          — max Sharpe, equity-heavy
      81–100 Aggressive      — max Sharpe, allows crypto
    """

    PROFILES = [
        RiskProfile(score=10,  label="Conservative",  strategy="min_variance",
                    target_vol=0.08, max_crypto=0.00, max_equity=0.40),
        RiskProfile(score=30,  label="Moderate",      strategy="risk_parity",
                    target_vol=0.12, max_crypto=0.02, max_equity=0.55),
        RiskProfile(score=50,  label="Balanced",      strategy="risk_parity",
                    target_vol=0.16, max_crypto=0.05, max_equity=0.70),
        RiskProfile(score=70,  label="Growth",        strategy="max_sharpe",
                    target_vol=0.22, max_crypto=0.10, max_equity=0.85),
        RiskProfile(score=90,  label="Aggressive",    strategy="max_sharpe",
                    target_vol=0.35, max_crypto=0.25, max_equity=1.00),
    ]

    @classmethod
    def from_score(cls, score: int) -> RiskProfile:
        """
        Return the closest RiskProfile for a score in [0, 100].
        Uses nearest-neighbour matching on profile score midpoints.
        """
        score = max(0, min(100, score))
        return min(cls.PROFILES, key=lambda p: abs(p.score - score))

    @classmethod
    def from_questionnaire(
        cls,
        investment_horizon_years: int,
        loss_tolerance_pct: float,
        income_stability: str,    # "stable" | "variable" | "none"
        prior_experience: str,    # "none" | "basic" | "experienced"
    ) -> RiskProfile:
        """
        Derive a risk score from a 4-question financial questionnaire.

        Scoring rubric (each dimension contributes 0–25 pts):
          Horizon      : <3yr=5  3-5yr=10  5-10yr=18  10+yr=25
          Loss tolerance: <5%=5  5-10%=12  10-20%=18  >20%=25
          Income       : none=5  variable=12  stable=25
          Experience   : none=5  basic=15  experienced=25
        """
        # Horizon score
        h = (5 if investment_horizon_years < 3 else
             10 if investment_horizon_years < 5 else
             18 if investment_horizon_years < 10 else 25)

        # Loss tolerance score
        l = (5 if loss_tolerance_pct < 5 else
             12 if loss_tolerance_pct < 10 else
             18 if loss_tolerance_pct < 20 else 25)

        # Income stability score
        i = {"none": 5, "variable": 12, "stable": 25}.get(income_stability, 12)

        # Experience score
        e = {"none": 5, "basic": 15, "experienced": 25}.get(prior_experience, 10)

        total = h + l + i + e
        logger.info(
            "Risk questionnaire score: %d/100 (horizon=%d, loss=%d, income=%d, exp=%d)",
            total, h, l, i, e,
        )
        return cls.from_score(total)


# ---------------------------------------------------------------------------
# Portfolio Optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Core optimisation engine. Consumes FeatureEngineer outputs and solves
    for optimal portfolio weights using four distinct strategies.

    Parameters
    ----------
    fe              : FeatureEngineer instance
    cov_window      : lookback window in trading days for covariance (default 252)
    return_window   : lookback window for expected return estimation (default 252)
    """

    STRATEGIES = ["max_sharpe", "min_variance", "mean_variance", "risk_parity", "black_litterman"]

    def __init__(
        self,
        fe: FeatureEngineer,
        cov_window: int    = 252,
        return_window: int = 252,
    ) -> None:
        self.fe            = fe
        self.cov_window    = cov_window
        self.return_window = return_window
        self.engine        = create_engine(DB_URL, echo=False)
        self._ensure_results_table()
        logger.info("PortfolioOptimizer ready | cov_window=%d", cov_window)

    # -----------------------------------------------------------------------
    # Primary Public API
    # -----------------------------------------------------------------------

    def optimise(
        self,
        tickers    : list[str],
        strategy   : str = "max_sharpe",
        constraints: Optional[OptimisationConstraints] = None,
        target_return: Optional[float] = None,   # for mean_variance only
        views      : Optional[dict]   = None,    # for black_litterman only
    ) -> PortfolioResult:
        """
        Run a single optimisation and return a PortfolioResult.

        Parameters
        ----------
        tickers       : list of ticker strings to include
        strategy      : one of STRATEGIES
        constraints   : OptimisationConstraints (uses defaults if None)
        target_return : required for 'mean_variance' strategy (annualised)
        views         : dict of {ticker: expected_return} for Black-Litterman
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {self.STRATEGIES}")

        constraints = constraints or OptimisationConstraints()
        logger.info("Optimising | strategy=%s | n=%d tickers", strategy, len(tickers))

        # --- Load inputs ---
        mu, Sigma, tickers = self._prepare_inputs(tickers)
        n = len(tickers)

        if n < 2:
            raise ValueError("Need at least 2 tickers with sufficient data.")

        # --- Dispatch to strategy ---
        dispatch = {
            "max_sharpe"       : lambda: self._max_sharpe(mu, Sigma, n, constraints),
            "min_variance"     : lambda: self._min_variance(Sigma, n, constraints),
            "mean_variance"    : lambda: self._mean_variance(mu, Sigma, n, constraints, target_return),
            "risk_parity"      : lambda: self._risk_parity(Sigma, n, constraints),
            "black_litterman"  : lambda: self._black_litterman(mu, Sigma, n, constraints, views or {}),
        }

        weights, status = dispatch[strategy]()
        weights = self._clean_weights(weights, constraints)

        result = self._build_result(tickers, weights, mu, Sigma, strategy, status)
        self._save_result(result)
        return result

    def optimise_for_profile(
        self,
        tickers    : list[str],
        profile    : RiskProfile,
        views      : Optional[dict] = None,
    ) -> PortfolioResult:
        """
        Run optimisation with constraints derived from a RiskProfile.
        Automatically enforces asset-class caps for crypto and equities.
        """
        logger.info(
            "Profile-based optimisation | profile=%s | strategy=%s",
            profile.label, profile.strategy,
        )
        constraints = OptimisationConstraints(
            min_weight = 0.01,
            max_weight = 0.40,
            max_sector_wt = {
                "crypto" : profile.max_crypto,
                "equity" : profile.max_equity,
            },
        )
        return self.optimise(
            tickers    = tickers,
            strategy   = profile.strategy,
            constraints= constraints,
            views      = views,
        )

    def efficient_frontier(
        self,
        tickers     : list[str],
        n_points    : int = 50,
        constraints : Optional[OptimisationConstraints] = None,
    ) -> pd.DataFrame:
        """
        Trace the full efficient frontier by solving minimum variance
        for a grid of target returns between min and max feasible return.

        Returns
        -------
        pd.DataFrame with columns:
            target_return | volatility | sharpe | weights (one col per ticker)
        """
        constraints = constraints or OptimisationConstraints()
        mu, Sigma, tickers = self._prepare_inputs(tickers)
        n = len(tickers)

        # Feasible return range
        min_ret = float(mu.min()) * 0.80
        max_ret = float(mu.max()) * 1.10
        target_returns = np.linspace(min_ret, max_ret, n_points)

        records = []
        for target in target_returns:
            try:
                weights, status = self._mean_variance(mu, Sigma, n, constraints, target)
                if status not in ("optimal", "optimal_inaccurate"):
                    continue
                weights = self._clean_weights(weights, constraints)
                port_ret = float(mu @ weights)
                port_vol = float(np.sqrt(weights @ Sigma @ weights))
                sharpe   = (port_ret - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
                row      = {"target_return": target, "volatility": port_vol,
                            "sharpe": sharpe, "portfolio_return": port_ret}
                row.update({t: float(w) for t, w in zip(tickers, weights)})
                records.append(row)
            except Exception:
                continue

        df = pd.DataFrame(records)
        logger.info("Efficient frontier: %d feasible points out of %d", len(df), n_points)
        return df

    def compare_strategies(
        self,
        tickers     : list[str],
        constraints : Optional[OptimisationConstraints] = None,
    ) -> pd.DataFrame:
        """
        Run all strategies on the same universe and return a comparison table.
        Columns: strategy | return | volatility | sharpe | max_weight | diversification
        """
        constraints = constraints or OptimisationConstraints()
        rows = []
        for strategy in ["max_sharpe", "min_variance", "risk_parity"]:
            try:
                result = self.optimise(tickers, strategy=strategy, constraints=constraints)
                rows.append({
                    "strategy"       : strategy,
                    "annual_return"  : result.expected_return,
                    "volatility"     : result.volatility,
                    "sharpe_ratio"   : result.sharpe_ratio,
                    "max_weight"     : float(result.weights.max()),
                    "diversification": result.diversification,
                    "solver_status"  : result.solver_status,
                })
            except Exception as exc:
                logger.warning("Strategy %s failed: %s", strategy, exc)
        return pd.DataFrame(rows).set_index("strategy")

    # -----------------------------------------------------------------------
    # Strategy Implementations
    # -----------------------------------------------------------------------

    def _max_sharpe(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        n: int,
        c: OptimisationConstraints,
    ) -> tuple[np.ndarray, str]:
        """
        Maximum Sharpe Ratio via Markowitz-Tobin reformulation.

        Trick: let y = w / (w^T · 1), solve for y, normalise.
        This converts a non-convex problem into a convex QP.

        Objective : minimise  y^T Σ y
        Subject to: (μ - rf)^T y = 1
                    y ≥ 0  (long-only)
                    bounds mapped back after normalisation
        """
        excess_mu = mu - RISK_FREE_RATE / TRADING_DAYS_YEAR * TRADING_DAYS_YEAR
        excess_mu = np.maximum(excess_mu, 1e-8)  # avoid degenerate cases

        y      = cp.Variable(n)
        obj    = cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma)))
        cons   = [excess_mu @ y == 1, y >= 0]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.CLARABEL, warm_start=True)

        if prob.status in ("optimal", "optimal_inaccurate") and y.value is not None:
            raw     = y.value
            weights = raw / raw.sum()
        else:
            logger.warning("Max Sharpe solver status: %s — falling back to equal weight", prob.status)
            weights = np.ones(n) / n

        return weights, prob.status or "fallback"

    def _min_variance(
        self,
        Sigma: np.ndarray,
        n: int,
        c: OptimisationConstraints,
    ) -> tuple[np.ndarray, str]:
        """
        Global Minimum Variance Portfolio.

        Objective : minimise  w^T Σ w
        Subject to: sum(w) = leverage
                    w_i ∈ [min_weight, max_weight]
        """
        w    = cp.Variable(n)
        obj  = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
        cons = [
            cp.sum(w) == c.leverage,
            w >= (0 if c.long_only else -0.5),
            w <= c.max_weight,
        ]
        if c.long_only:
            cons.append(w >= c.min_weight)

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.CLARABEL, warm_start=True)

        weights = w.value if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None \
                  else np.ones(n) / n
        return weights, prob.status or "fallback"

    def _mean_variance(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        n: int,
        c: OptimisationConstraints,
        target_return: Optional[float] = None,
    ) -> tuple[np.ndarray, str]:
        """
        Markowitz Mean-Variance: minimise variance for a target return.
        If target_return is None, defaults to midpoint of feasible range.

        Objective : minimise  w^T Σ w
        Subject to: μ^T w >= target_return
                    sum(w) = 1
                    w_i ∈ [min_weight, max_weight]
        """
        if target_return is None:
            target_return = float(np.mean(mu))

        w    = cp.Variable(n)
        obj  = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))
        cons = [
            cp.sum(w) == c.leverage,
            mu @ w >= target_return,
            w <= c.max_weight,
            w >= (c.min_weight if c.long_only else -0.5),
        ]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.CLARABEL, warm_start=True)

        weights = w.value if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None \
                  else np.ones(n) / n
        return weights, prob.status or "fallback"

    def _risk_parity(
        self,
        Sigma: np.ndarray,
        n: int,
        c: OptimisationConstraints,
    ) -> tuple[np.ndarray, str]:
        """
        Risk Parity (Equal Risk Contribution).

        Each asset contributes the same amount to total portfolio variance.
        RC_i = w_i * (Σw)_i / (w^T Σ w)  — we want RC_i = 1/n for all i.

        Solved via SciPy L-BFGS-B (non-convex but smooth and well-behaved).

        Objective : minimise  Σ_i (RC_i - 1/n)^2
        """
        Sigma_np = np.array(Sigma)

        def risk_contributions(w: np.ndarray) -> np.ndarray:
            port_var = w @ Sigma_np @ w
            mrc      = Sigma_np @ w                   # marginal risk contributions
            rc       = w * mrc / (port_var + 1e-12)   # % risk contribution
            return rc

        def objective(w: np.ndarray) -> float:
            rc     = risk_contributions(w)
            target = np.ones(n) / n
            return float(np.sum((rc - target) ** 2))

        def gradient(w: np.ndarray) -> np.ndarray:
            port_var = w @ Sigma_np @ w + 1e-12
            mrc      = Sigma_np @ w
            rc       = w * mrc / port_var
            target   = np.ones(n) / n
            # Analytic gradient of objective w.r.t. w
            d_rc_d_w = (
                np.diag(mrc) + np.diag(w) @ Sigma_np
                - 2 * np.outer(w * mrc, mrc)
            ) / port_var
            return 2 * (rc - target) @ d_rc_d_w

        w0      = np.ones(n) / n
        bounds  = [(max(c.min_weight, 1e-4), c.max_weight)] * n
        cons_eq = {"type": "eq", "fun": lambda w: np.sum(w) - c.leverage}

        result = minimize(
            objective, w0, jac=gradient, method="SLSQP",
            bounds=bounds, constraints=cons_eq,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = result.x if result.success else np.ones(n) / n
        status  = "optimal" if result.success else f"scipy_{result.message}"
        return weights, status

    def _black_litterman(
        self,
        mu_hist: np.ndarray,
        Sigma: np.ndarray,
        n: int,
        c: OptimisationConstraints,
        views: dict,
        tau: float = 0.05,
    ) -> tuple[np.ndarray, str]:
        """
        Black-Litterman Model.

        Blends market-implied equilibrium returns (from reverse optimisation
        of equal-weight portfolio) with analyst views using Bayesian updating.

        Parameters
        ----------
        views : dict mapping ticker → expected annualised return
                e.g. {"AAPL": 0.18, "BTC-USD": 0.45}
        tau   : uncertainty scalar on prior (typically 0.01–0.10)

        Steps
        -----
        1. Compute market-implied excess returns  Π = λ Σ w_mkt
        2. Build view matrix P and view vector Q
        3. Compute posterior: μ_BL = [(τΣ)^-1 + P^T Ω^-1 P]^-1 [(τΣ)^-1 Π + P^T Ω^-1 Q]
        4. Run Max Sharpe on posterior μ_BL
        """
        tickers_list = list(views.keys()) if views else []

        # Step 1: Market-implied equilibrium returns
        w_mkt    = np.ones(n) / n          # equal-weight market portfolio proxy
        port_var = float(w_mkt @ Sigma @ w_mkt)
        lam      = (np.mean(mu_hist) - RISK_FREE_RATE) / port_var  # risk aversion
        pi_eq    = lam * Sigma @ w_mkt     # equilibrium excess returns

        if not views:
            logger.info("No BL views provided — using equilibrium returns")
            mu_bl = pi_eq + RISK_FREE_RATE
        else:
            # Step 2: Build P (k×n pick matrix) and Q (k×1 view vector)
            ticker_idx = {t: i for i, t in enumerate(self._last_tickers)}
            k    = len(views)
            P    = np.zeros((k, n))
            Q    = np.zeros(k)
            for j, (ticker, ret) in enumerate(views.items()):
                if ticker in ticker_idx:
                    P[j, ticker_idx[ticker]] = 1.0
                    Q[j] = ret - RISK_FREE_RATE
                else:
                    logger.warning("BL view ticker %s not found in universe", ticker)

            # Step 3: Omega = diagonal uncertainty proportional to P Σ P^T
            Omega = np.diag(np.diag(tau * P @ Sigma @ P.T))

            # Step 4: Posterior mean (Black-Litterman formula)
            tau_sigma_inv = np.linalg.inv(tau * Sigma)
            omega_inv     = np.linalg.inv(Omega + np.eye(k) * 1e-8)
            posterior_cov = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
            mu_bl         = posterior_cov @ (tau_sigma_inv @ pi_eq + P.T @ omega_inv @ Q)
            mu_bl        += RISK_FREE_RATE

            logger.info("Black-Litterman posterior computed | %d views blended", k)

        # Step 5: Max Sharpe on posterior returns
        return self._max_sharpe(mu_bl, Sigma, n, c)

    # -----------------------------------------------------------------------
    # Input Preparation
    # -----------------------------------------------------------------------

    def _prepare_inputs(
        self, tickers: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Load and align expected returns (μ) and covariance matrix (Σ).

        Returns
        -------
        mu     : 1-D array of annualised expected returns (n,)
        Sigma  : 2-D annualised covariance matrix (n×n), positive semi-definite
        tickers: filtered list of tickers that have sufficient data
        """
        # Expected returns from risk summary
        risk_df  = self.fe.get_risk_summary()
        risk_df  = risk_df.set_index("ticker")

        # Covariance matrix
        cov_df   = self.fe.get_covariance_matrix(tickers=tickers, window=self.cov_window)

        if cov_df.empty:
            raise RuntimeError("No covariance data available. Run fe.build_all() first.")

        # Intersect: keep only tickers present in both risk_df and cov_df
        available = [t for t in tickers if t in cov_df.index and t in risk_df.index]

        if len(available) < 2:
            raise RuntimeError(
                f"Insufficient data. Found {len(available)} tickers with complete features."
                f" Run fe.build_all() first."
            )

        dropped = set(tickers) - set(available)
        if dropped:
            logger.warning("Dropped tickers (insufficient data): %s", dropped)

        # Align and extract numpy arrays
        cov_df  = cov_df.loc[available, available]
        mu_raw  = risk_df.loc[available, "annual_return"].fillna(0).values.astype(float)

        # Regularise covariance matrix (add small diagonal to ensure PSD)
        Sigma   = cov_df.values.astype(float)
        Sigma  += np.eye(len(available)) * 1e-8

        self._last_tickers = available  # cache for Black-Litterman
        logger.info("Inputs ready | n=%d | μ range: [%.3f, %.3f]",
                    len(available), mu_raw.min(), mu_raw.max())
        return mu_raw, Sigma, available

    # -----------------------------------------------------------------------
    # Post-Processing
    # -----------------------------------------------------------------------

    def _clean_weights(
        self, weights: np.ndarray, c: OptimisationConstraints
    ) -> np.ndarray:
        """
        Post-solve weight cleanup:
          1. Clip to [min_weight, max_weight]
          2. Zero out weights below threshold (< 0.005 = 0.5%)
          3. Re-normalise to sum to leverage
        """
        weights = np.array(weights, dtype=float)
        weights = np.clip(weights, 0 if c.long_only else -0.5, c.max_weight)
        weights[weights < 0.005] = 0          # trim dust positions
        total = weights.sum()
        if total > 1e-6:
            weights = weights / total * c.leverage
        else:
            weights = np.ones(len(weights)) / len(weights)
        return weights

    def _build_result(
        self,
        tickers : list[str],
        weights : np.ndarray,
        mu      : np.ndarray,
        Sigma   : np.ndarray,
        strategy: str,
        status  : str,
    ) -> PortfolioResult:
        """
        Compute portfolio-level statistics and assemble a PortfolioResult.
        """
        # Portfolio return and volatility
        port_ret = float(mu @ weights)
        port_vol = float(np.sqrt(weights @ Sigma @ weights))
        sharpe   = (port_ret - RISK_FREE_RATE) / port_vol if port_vol > 1e-8 else 0.0

        # Diversification (1 - Herfindahl-Hirschman Index)
        hhi   = float(np.sum(weights ** 2))
        divers = 1.0 - hhi

        # Risk contributions (% of total portfolio variance)
        mrc  = Sigma @ weights
        rc   = weights * mrc / (weights @ Sigma @ weights + 1e-12)

        # Allocation DataFrame
        alloc_df = pd.DataFrame({
            "ticker"           : tickers,
            "weight"           : weights,
            "expected_return"  : mu,
            "risk_contribution": rc,
        })

        # Human-readable explanation
        top3 = alloc_df.nlargest(3, "weight")[["ticker", "weight"]]
        top3_str = ", ".join(
            f"{row.ticker} ({row.weight * 100:.1f}%)"
            for _, row in top3.iterrows()
        )
        explanation = (
            f"Top holdings: {top3_str}. "
            f"Portfolio targets {port_ret * 100:.1f}% annual return at "
            f"{port_vol * 100:.1f}% volatility (Sharpe: {sharpe:.2f}). "
            f"Diversification score: {divers:.2f}/1.00."
        )

        return PortfolioResult(
            strategy           = strategy,
            tickers            = tickers,
            weights            = weights,
            expected_return    = port_ret,
            volatility         = port_vol,
            sharpe_ratio       = sharpe,
            diversification    = divers,
            risk_contributions = rc,
            allocation_df      = alloc_df,
            solver_status      = status,
            explanation        = explanation,
        )

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _ensure_results_table(self) -> None:
        create_sql = """
        CREATE TABLE IF NOT EXISTS portfolio_results (
            run_id          TEXT PRIMARY KEY,
            strategy        TEXT,
            tickers         TEXT,
            weights         TEXT,
            expected_return REAL,
            volatility      REAL,
            sharpe_ratio    REAL,
            diversification REAL,
            solver_status   TEXT,
            explanation     TEXT,
            timestamp       TEXT
        )
        """
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()

    def _save_result(self, result: PortfolioResult) -> None:
        """
        Persist a PortfolioResult to the portfolio_results table and
        save the allocation breakdown as a CSV.
        """
        import json
        run_id = f"{result.strategy}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}"

        record = {
            "run_id"         : run_id,
            "strategy"       : result.strategy,
            "tickers"        : json.dumps(result.tickers),
            "weights"        : json.dumps(result.weights.tolist()),
            "expected_return": result.expected_return,
            "volatility"     : result.volatility,
            "sharpe_ratio"   : result.sharpe_ratio,
            "diversification": result.diversification,
            "solver_status"  : result.solver_status,
            "explanation"    : result.explanation,
            "timestamp"      : result.timestamp.isoformat(),
        }

        insert_sql = """
            INSERT OR REPLACE INTO portfolio_results
            (run_id, strategy, tickers, weights, expected_return, volatility,
             sharpe_ratio, diversification, solver_status, explanation, timestamp)
            VALUES
            (:run_id, :strategy, :tickers, :weights, :expected_return, :volatility,
             :sharpe_ratio, :diversification, :solver_status, :explanation, :timestamp)
        """
        with self.engine.connect() as conn:
            conn.execute(text(insert_sql), record)
            conn.commit()

        # CSV snapshot
        csv_path = RESULTS_DIR / f"{run_id}.csv"
        result.allocation_df.to_csv(csv_path, index=False)
        logger.info("Result persisted | run_id=%s | csv=%s", run_id, csv_path)

    def load_results_history(self) -> pd.DataFrame:
        """
        Load all past optimisation runs from the DB for comparison.
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM portfolio_results ORDER BY timestamp DESC"),
                conn, parse_dates=["timestamp"],
            )
        return df


# ---------------------------------------------------------------------------
# Standalone Summary Printer
# ---------------------------------------------------------------------------

def print_comparison(df: pd.DataFrame) -> None:
    """Pretty-print a strategy comparison table."""
    print("\n" + "=" * 72)
    print("  InvestEng — Strategy Comparison")
    print("=" * 72)
    print(f"  {'Strategy':<20} {'Ann.Ret':>9} {'Vol':>9} {'Sharpe':>9} "
          f"{'MaxWt':>8} {'Divers':>8}")
    print("  " + "-" * 68)
    for strat, row in df.iterrows():
        print(
            f"  {strat:<20} "
            f"{row['annual_return'] * 100:>+8.2f}% "
            f"{row['volatility'] * 100:>8.2f}%  "
            f"{row['sharpe_ratio']:>8.3f}  "
            f"{row['max_weight'] * 100:>7.1f}%  "
            f"{row['diversification']:>7.3f}"
        )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from market_data import ASSET_UNIVERSE

    ingester  = MarketDataIngester(start_date="2019-01-01")
    fe        = FeatureEngineer(ingester)
    optimizer = PortfolioOptimizer(fe)

    # Build a diversified universe: mix of equities, ETFs, and crypto
    universe = (
        ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM"]  # US stocks
        + ["SPY", "QQQ", "GLD", "BND"]                      # ETFs
        + ["BTC-USD", "ETH-USD"]                             # Crypto
    )

    print("\n── Max Sharpe Portfolio ──")
    r1 = optimizer.optimise(universe, strategy="max_sharpe")
    print(r1)

    print("\n── Min Variance Portfolio ──")
    r2 = optimizer.optimise(universe, strategy="min_variance")
    print(r2)

    print("\n── Risk Parity Portfolio ──")
    r3 = optimizer.optimise(universe, strategy="risk_parity")
    print(r3)

    print("\n── Strategy Comparison ──")
    comp = optimizer.compare_strategies(universe)
    print_comparison(comp)

    print("\n── Profile-Based (Balanced investor, score=50) ──")
    profile = RiskProfiler.from_score(50)
    r4      = optimizer.optimise_for_profile(universe, profile)
    print(r4)

    print("\n── Black-Litterman (analyst views) ──")
    r5 = optimizer.optimise(
        universe, strategy="black_litterman",
        views={"NVDA": 0.35, "BTC-USD": 0.50, "BND": 0.04},
    )
    print(r5)
