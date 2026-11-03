"""Dixon-Coles match goal model for the WC 2026 Predictor.

Implements the model from Dixon & Coles (1997) *"Modelling Association
Football Scores and Inefficiencies in the Football Betting Market"*.

Performance optimisations over the naive implementation
-------------------------------------------------------
- **Analytical gradient** for ~50× faster convergence vs. finite-difference.
- **L-BFGS-B** with penalty term instead of constrained SLSQP.
- **Team filtering** — drop teams with <N appearances to shrink param space.
- **Vectorised goal matrix** — numpy outer product, no Python loops.
- **Matchup cache** — pre-compute all pairwise CDFs before Monte Carlo.

Parameters
----------
For *T* teams the model has:
    alpha_i  (i=1..T)  — attack strength of team *i*
    beta_i   (i=1..T)  — defense strength of team *i*
    gamma              — home-advantage parameter
    rho                — low-score dependence parameter

Identifiability: sum(alpha) = 0, enforced via penalty + post-hoc centring.

Point-in-time guarantee
-----------------------
``fit()`` accepts an ``as_of`` date.  Only matches with ``date < as_of``
are used for fitting.  No post-cutoff data leaks into parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

logger = logging.getLogger(__name__)

# Maximum goals per team considered in the probability matrix.
_MAX_GOALS: int = 7


# ---------------------------------------------------------------------------
# Tau (low-score correction)
# ---------------------------------------------------------------------------

def tau(
    x: int,
    y: int,
    lambda_: float,
    mu: float,
    rho: float,
) -> float:
    """Dixon-Coles low-score correction factor.

    Returns 1.0 for all scorelines beyond (0,0)/(0,1)/(1,0)/(1,1).
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_ * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lambda_ * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ---------------------------------------------------------------------------
# Vectorised goal matrix
# ---------------------------------------------------------------------------

def _build_goal_matrix_fast(
    lambda_: float,
    mu: float,
    rho: float,
    max_goals: int = _MAX_GOALS,
) -> np.ndarray:
    """Build goal probability matrix using vectorised outer product."""
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals, lambda_)
    p_away = poisson.pmf(goals, mu)

    # Outer product = independent Poisson joint probabilities
    matrix = np.outer(p_home, p_away)

    # Apply tau correction to the four low-score cells
    matrix[0, 0] *= (1.0 - lambda_ * mu * rho)
    matrix[0, 1] *= (1.0 + lambda_ * rho)
    matrix[1, 0] *= (1.0 + mu * rho)
    matrix[1, 1] *= (1.0 - rho)

    # Normalise
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


# ---------------------------------------------------------------------------
# NLL + analytical gradient
# ---------------------------------------------------------------------------

def _neg_log_likelihood_and_grad(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    neutral: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    penalty_strength: float,
) -> tuple[float, np.ndarray]:
    """NLL with analytical gradient and identifiability penalty.

    Uses L-BFGS-B-compatible penalty ``penalty_strength * sum(alpha)^2``
    instead of an equality constraint.
    """
    alphas = params[:n_teams]
    betas = params[n_teams: 2 * n_teams]
    gamma = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    # Expected goals
    home_venue = gamma * (1.0 - neutral)
    log_lambda = alphas[home_idx] - betas[away_idx] + home_venue
    log_mu = alphas[away_idx] - betas[home_idx]
    lambda_ = np.exp(log_lambda)
    mu_ = np.exp(log_mu)

    # Poisson log-probabilities (manual = faster than scipy for arrays)
    log_p_home = home_goals * log_lambda - lambda_ - gammaln(home_goals + 1)
    log_p_away = away_goals * log_mu - mu_ - gammaln(away_goals + 1)

    # Tau correction
    tau_vals = np.ones(len(home_goals), dtype=np.float64)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)

    tau_vals[m00] = 1.0 - lambda_[m00] * mu_[m00] * rho
    tau_vals[m01] = 1.0 + lambda_[m01] * rho
    tau_vals[m10] = 1.0 + mu_[m10] * rho
    tau_vals[m11] = 1.0 - rho

    tau_vals = np.clip(tau_vals, 1e-10, None)
    log_tau = np.log(tau_vals)

    # Weighted NLL
    match_ll = weights * (log_p_home + log_p_away + log_tau)
    nll = -match_ll.sum()

    # Penalty
    alpha_sum = alphas.sum()
    nll += penalty_strength * alpha_sum ** 2

    # --- Gradient ---
    grad = np.zeros_like(params)

    # Derivatives of Poisson log-prob w.r.t. log-rates
    dll_dloglam = home_goals - lambda_
    dll_dlogmu = away_goals - mu_

    # Derivatives of log(tau) w.r.t. lambda_, mu_, rho
    dtau_dlam = np.zeros(len(home_goals))
    dtau_dmu = np.zeros(len(home_goals))
    dtau_drho = np.zeros(len(home_goals))

    dtau_dlam[m00] = -mu_[m00] * rho
    dtau_dmu[m00] = -lambda_[m00] * rho
    dtau_drho[m00] = -lambda_[m00] * mu_[m00]

    dtau_dlam[m01] = rho
    dtau_drho[m01] = lambda_[m01]

    dtau_dmu[m10] = rho
    dtau_drho[m10] = mu_[m10]

    dtau_drho[m11] = -1.0

    dlogtau_dlam = dtau_dlam / tau_vals
    dlogtau_dmu = dtau_dmu / tau_vals
    dlogtau_drho = dtau_drho / tau_vals

    # Chain: d(log_tau)/d(log_lambda) = d(log_tau)/d(lambda) * lambda_
    d_loglam = weights * (dll_dloglam + dlogtau_dlam * lambda_)
    d_logmu = weights * (dll_dlogmu + dlogtau_dmu * mu_)

    # Alpha gradients (attack)
    np.add.at(grad[:n_teams], home_idx, -d_loglam)
    np.add.at(grad[:n_teams], away_idx, -d_logmu)

    # Beta gradients (defense)
    np.add.at(grad[n_teams:2 * n_teams], away_idx, d_loglam)
    np.add.at(grad[n_teams:2 * n_teams], home_idx, d_logmu)

    # Gamma gradient
    grad[2 * n_teams] = -np.sum(d_loglam * (1.0 - neutral))

    # Rho gradient
    grad[2 * n_teams + 1] = -np.sum(weights * dlogtau_drho)

    # Penalty gradient
    grad[:n_teams] += 2 * penalty_strength * alpha_sum

    return nll, grad


# Legacy interface for tests
def _neg_log_likelihood(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    neutral: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """Compute NLL without gradient (backward-compatible)."""
    nll, _ = _neg_log_likelihood_and_grad(
        params, home_idx, away_idx, home_goals, away_goals,
        neutral, weights, n_teams, penalty_strength=0.0,
    )
    return nll


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

@dataclass
class DixonColesModel:
    """Fitted Dixon-Coles model."""

    teams: list[str]
    attack: np.ndarray
    defense: np.ndarray
    home_advantage: float
    rho: float
    _team_to_idx: dict[str, int] = field(init=False, repr=False)
    _matchup_cache: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._team_to_idx = {t: i for i, t in enumerate(self.teams)}
        self._matchup_cache = {}

    def predict(
        self,
        home: str,
        away: str,
        neutral: bool = True,
        max_goals: int = _MAX_GOALS,
    ) -> dict:
        """Predict match outcome probabilities.

        Returns dict with keys: home_win, draw, away_win,
        expected_home_goals, expected_away_goals, goal_matrix.
        """
        hi = self._team_to_idx[home]
        ai = self._team_to_idx[away]

        gamma_adj = 0.0 if neutral else self.home_advantage
        lambda_ = np.exp(self.attack[hi] - self.defense[ai] + gamma_adj)
        mu_ = np.exp(self.attack[ai] - self.defense[hi])

        goal_matrix = _build_goal_matrix_fast(lambda_, mu_, self.rho, max_goals)

        home_win = float(np.sum(np.tril(goal_matrix, k=-1)))
        draw = float(np.sum(np.diag(goal_matrix)))
        away_win = float(np.sum(np.triu(goal_matrix, k=1)))

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "expected_home_goals": float(lambda_),
            "expected_away_goals": float(mu_),
            "goal_matrix": goal_matrix,
        }

    def simulate_score(
        self,
        home: str,
        away: str,
        neutral: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[int, int]:
        """Draw a random scoreline.  Uses cached CDF for speed."""
        if rng is None:
            rng = np.random.default_rng()

        cache_key = (home, away, neutral)
        if cache_key not in self._matchup_cache:
            pred = self.predict(home, away, neutral=neutral)
            self._matchup_cache[cache_key] = pred["goal_matrix"].ravel()

        flat = self._matchup_cache[cache_key]
        idx = rng.choice(len(flat), p=flat)
        n_cols = _MAX_GOALS + 1
        home_goals, away_goals = divmod(idx, n_cols)
        return int(home_goals), int(away_goals)

    def precompute_matchups(self, teams: list[str], neutral: bool = True) -> None:
        """Pre-compute all pairwise CDFs for Monte Carlo speed.

        For 48 teams: 48×47 = 2,256 matchups cached in ~1 second.
        """
        n = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                key = (home, away, neutral)
                if key not in self._matchup_cache:
                    pred = self.predict(home, away, neutral=neutral)
                    self._matchup_cache[key] = pred["goal_matrix"].ravel()
                    n += 1
        logger.info("precompute_matchups: cached %d matchup CDFs", n)

    def get_params(self) -> pd.DataFrame:
        """Return DataFrame of team attack/defense params, sorted by attack."""
        df = pd.DataFrame({
            "team": self.teams,
            "attack": self.attack,
            "defense": self.defense,
        })
        return df.sort_values("attack", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit(
    matches: pd.DataFrame,
    as_of: Optional[str] = None,
    half_life_days: float = 730.0,
    max_goals_filter: int = 10,
    min_team_appearances: int = 10,
    required_teams: Optional[list[str]] = None,
) -> DixonColesModel:
    """Fit a Dixon-Coles model on international match results.

    Point-in-time guarantee: only matches with ``date < as_of`` are used.

    Parameters
    ----------
    matches : pd.DataFrame
        Required columns: date, home_team, away_team, home_score,
        away_score, neutral.
    as_of : str or None
        ISO date cutoff.
    half_life_days : float
        Exponential decay half-life (default 730 = 2 years).
    max_goals_filter : int
        Drop matches with more goals than this.
    min_team_appearances : int
        Drop teams with fewer appearances (reduces param space).
    required_teams : list[str] or None
        Teams kept regardless of appearance count (e.g. 48 WC teams).

    Returns
    -------
    DixonColesModel
    """
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])

    # --- Point-in-time filter ---
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        df = df[df["date"] < cutoff]
        logger.info("fit: filtered to %d matches before %s", len(df), as_of)

    if len(df) == 0:
        raise ValueError("No matches remain after filtering.")

    # --- Outlier filter ---
    mask = (df["home_score"] <= max_goals_filter) & (
        df["away_score"] <= max_goals_filter
    )
    n_dropped = (~mask).sum()
    if n_dropped:
        logger.info("fit: dropped %d matches with >%d goals", n_dropped, max_goals_filter)
    df = df[mask].reset_index(drop=True)

    # --- Neutral venue ---
    if "neutral" not in df.columns:
        df["neutral"] = 0.0
    df["neutral"] = df["neutral"].astype(float)

    # --- Filter low-appearance teams ---
    if min_team_appearances > 0:
        appearances = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        keep_teams = set(appearances[appearances >= min_team_appearances].index)
        if required_teams:
            keep_teams |= set(required_teams)
        mask = df["home_team"].isin(keep_teams) & df["away_team"].isin(keep_teams)
        n_before = len(df)
        df = df[mask].reset_index(drop=True)
        all_teams_before = len(appearances)
        logger.info(
            "fit: team filter (min=%d) kept %d/%d matches, %d->%d teams",
            min_team_appearances, len(df), n_before,
            all_teams_before, len(keep_teams),
        )

    # --- Team index mapping ---
    all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)
    logger.info("fit: %d teams, %d matches", n_teams, len(df))

    home_idx = df["home_team"].map(team_to_idx).values.astype(int)
    away_idx = df["away_team"].map(team_to_idx).values.astype(int)
    home_goals = df["home_score"].values.astype(int)
    away_goals = df["away_score"].values.astype(int)
    neutral_arr = df["neutral"].values.astype(float)

    # --- Time-decay weights ---
    ref_date = df["date"].max()
    days_ago = (ref_date - df["date"]).dt.days.values.astype(float)
    decay = np.log(2) / half_life_days
    weights = np.exp(-decay * days_ago)

    # --- Initial parameters ---
    x0 = np.zeros(2 * n_teams + 2)
    x0[2 * n_teams] = 0.1
    x0[2 * n_teams + 1] = -0.05

    # --- Penalty for identifiability ---
    penalty_strength = float(len(df))

    # --- Bounds ---
    bounds = [(None, None)] * (2 * n_teams)
    bounds.append((None, None))      # gamma
    bounds.append((-0.5, 0.5))       # rho

    logger.info("fit: starting L-BFGS-B (%d params, %d matches)…", len(x0), len(df))
    result = minimize(
        _neg_log_likelihood_and_grad,
        x0,
        args=(home_idx, away_idx, home_goals, away_goals,
              neutral_arr, weights, n_teams, penalty_strength),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-8},
    )

    if not result.success:
        logger.warning("fit: optimiser note — %s", result.message)

    # --- Extract and centre parameters ---
    alphas = result.x[:n_teams]
    betas = result.x[n_teams: 2 * n_teams]
    gamma = float(result.x[2 * n_teams])
    rho = float(result.x[2 * n_teams + 1])

    # Post-hoc centring for exact identifiability
    alphas = alphas - alphas.mean()

    logger.info(
        "fit: done — gamma=%.4f, rho=%.4f, nll=%.1f, iters=%d, "
        "alpha_range=[%.3f, %.3f]",
        gamma, rho, result.fun, result.nit, alphas.min(), alphas.max(),
    )

    return DixonColesModel(
        teams=all_teams,
        attack=alphas,
        defense=betas,
        home_advantage=gamma,
        rho=rho,
    )
