"""Backtesting framework for the WC 2026 Predictor.

Evaluates the Dixon-Coles model on historical World Cups (2014, 2018, 2022)
using strict point-in-time data.

Point-in-time guarantee
-----------------------
The model is fitted on all international results with ``date < tournament_start``.
No post-tournament data is used in parameter estimation.  This guarantee is
enforced by passing ``as_of=tournament_start`` to ``models.dixon_coles.fit()``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from evaluation.metrics import brier_score, log_loss
from models.dixon_coles import DixonColesModel, fit

logger = logging.getLogger(__name__)

# World Cup start dates (first match of the tournament).
_WC_START_DATES: dict[int, str] = {
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
    2026: "2026-06-11",
}


def get_tournament_matches(
    results: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Extract World Cup matches for a given year.

    Parameters
    ----------
    results : pd.DataFrame
        Full international results (from ``data.loaders.load_international_results``).
    year : int
        World Cup year (2014, 2018, or 2022).

    Returns
    -------
    pd.DataFrame
        Subset of results that are World Cup matches in the given year.
    """
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Match on year and tournament containing "FIFA World Cup" (not qualifiers)
    mask = (
        (df["date"].dt.year == year)
        & (df["tournament"].str.contains("FIFA World Cup", case=False, na=False))
        & (~df["tournament"].str.contains("qualif", case=False, na=False))
    )
    wc_matches = df[mask].reset_index(drop=True)
    logger.info(
        "get_tournament_matches: %d matches for WC %d", len(wc_matches), year
    )
    return wc_matches


def _outcome_to_onehot(home_score: int, away_score: int) -> np.ndarray:
    """Convert a scoreline to one-hot [home_win, draw, away_win]."""
    if home_score > away_score:
        return np.array([1.0, 0.0, 0.0])
    elif home_score == away_score:
        return np.array([0.0, 1.0, 0.0])
    else:
        return np.array([0.0, 0.0, 1.0])


def backtest_tournament(
    results: pd.DataFrame,
    year: int,
    half_life_days: float = 730.0,
    start_year: int = 1990,
) -> dict:
    """Backtest the Dixon-Coles model on a single World Cup.

    Point-in-time guarantee: the model is fitted on all results before the
    tournament start date.  No post-tournament data leaks into parameters.

    Parameters
    ----------
    results : pd.DataFrame
        Full international results.
    year : int
        World Cup year to evaluate (2014, 2018, or 2022).
    half_life_days : float
        Time-decay half-life for the Dixon-Coles fit.
    start_year : int
        Earliest year of training data.

    Returns
    -------
    dict
        Keys:
        - ``year`` : int
        - ``n_matches`` : int — number of WC matches evaluated
        - ``brier_score`` : float
        - ``log_loss`` : float
        - ``predictions`` : pd.DataFrame — per-match predictions + outcomes
        - ``model`` : DixonColesModel — the fitted model
    """
    if year not in _WC_START_DATES:
        raise ValueError(f"Unknown WC year {year}. Valid: {list(_WC_START_DATES)}")

    as_of = _WC_START_DATES[year]
    logger.info("backtest_tournament: WC %d, as_of=%s", year, as_of)

    # --- Fit model on pre-tournament data ---
    train = results[pd.to_datetime(results["date"]).dt.year >= start_year].copy()
    model = fit(train, as_of=as_of, half_life_days=half_life_days)

    # --- Get tournament matches ---
    wc_matches = get_tournament_matches(results, year)
    if len(wc_matches) == 0:
        raise ValueError(f"No WC matches found for {year}.")

    # --- Predict each match ---
    predictions_rows = []
    predicted_probs = []
    actual_outcomes = []

    for _, row in wc_matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Skip teams not in the training data
        if home not in model._team_to_idx or away not in model._team_to_idx:
            logger.warning(
                "backtest_tournament: skipping %s vs %s — team(s) not in model",
                home, away,
            )
            continue

        pred = model.predict(home, away, neutral=True)
        prob = np.array([pred["home_win"], pred["draw"], pred["away_win"]])
        outcome = _outcome_to_onehot(row["home_score"], row["away_score"])

        predicted_probs.append(prob)
        actual_outcomes.append(outcome)

        predictions_rows.append({
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "p_home_win": prob[0],
            "p_draw": prob[1],
            "p_away_win": prob[2],
            "actual_outcome": "H" if outcome[0] else ("D" if outcome[1] else "A"),
        })

    if not predicted_probs:
        raise ValueError(f"No predictable matches for WC {year}.")

    pred_arr = np.array(predicted_probs)
    out_arr = np.array(actual_outcomes)

    bs = brier_score(pred_arr, out_arr)
    ll = log_loss(pred_arr, out_arr)

    predictions_df = pd.DataFrame(predictions_rows)
    logger.info(
        "backtest_tournament: WC %d — %d matches, Brier=%.4f, LogLoss=%.4f",
        year, len(predictions_df), bs, ll,
    )

    return {
        "year": year,
        "n_matches": len(predictions_df),
        "brier_score": bs,
        "log_loss": ll,
        "predictions": predictions_df,
        "model": model,
    }


def backtest_all(
    results: pd.DataFrame,
    years: Optional[list[int]] = None,
    half_life_days: float = 730.0,
    start_year: int = 1990,
) -> pd.DataFrame:
    """Run backtests on multiple World Cups and return a summary table.

    Parameters
    ----------
    results : pd.DataFrame
        Full international results.
    years : list[int] or None
        WC years to evaluate.  Defaults to [2014, 2018, 2022].
    half_life_days : float
        Time-decay half-life.
    start_year : int
        Earliest training year.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: ``year, n_matches, brier_score, log_loss``.
    """
    if years is None:
        years = [2014, 2018, 2022]

    rows = []
    for yr in years:
        try:
            result = backtest_tournament(
                results, yr, half_life_days=half_life_days, start_year=start_year
            )
            rows.append({
                "year": result["year"],
                "n_matches": result["n_matches"],
                "brier_score": result["brier_score"],
                "log_loss": result["log_loss"],
            })
        except ValueError as e:
            logger.warning("backtest_all: skipping %d — %s", yr, e)

    summary = pd.DataFrame(rows)
    if len(summary):
        avg_brier = summary["brier_score"].mean()
        avg_ll = summary["log_loss"].mean()
        logger.info(
            "backtest_all: avg Brier=%.4f, avg LogLoss=%.4f over %d tournaments",
            avg_brier, avg_ll, len(summary),
        )
    return summary
