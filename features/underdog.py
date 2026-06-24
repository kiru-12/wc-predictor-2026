"""Underdog feature pipeline for the WC 2026 Predictor.

PURPOSE — FEATURE COLUMNS ONLY
--------------------------------
This module produces the ``upset_potential_index`` (UPI) and its four
sub-metric columns as **input features** for the logistic upset model in
``models/``.  These columns must NOT be used as a win-probability multiplier
or blended with any other score at this stage.  The upset model learns the
weight of UPI from backtested data.  Do NOT re-add a hardcoded multiplier
(e.g. the 0.65/0.35 blend shown in the compute_metrics integration guide —
that is for reference only and is explicitly forbidden by project rules).

Pipeline
--------
1. Load ``context_csv`` (manually curated per-team data).
2. Left-join onto ``team_baseline_df`` so team names are consistent.
3. Warn and emit NaN for any team missing a required input column.
4. Call ``compute_metrics`` from ``world_cup_predictor_metrics``.
5. Select output columns and save to ``data/processed/underdog_features.csv``.

Output columns
--------------
``team, upset_potential_index, resource_gap_ratio,
leadership_stability_score, cohesion_pressure_score,
motivational_momentum_score``
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from features.world_cup_predictor_metrics import compute_metrics

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "processed"

# Columns that compute_metrics reads from the context CSV.
# A missing column → warning + NaN for the affected sub-metric(s).
_REQUIRED_CONTEXT_COLS: list[str] = [
    "squad_market_value_eur",
    "avg_group_opponent_value_eur",
    "pre_tournament_odds",
    "manager_tenure_months",
    "historic_first_time",
    "core_retention_score",
    "manager_stability_score",
    "rival_top_team_underperformed",
]

# Columns to keep in the final output.
_OUTPUT_COLS: list[str] = [
    "team",
    "upset_potential_index",
    "resource_gap_ratio",
    "leadership_stability_score",
    "cohesion_pressure_score",
    "motivational_momentum_score",
]


def _load_context(context_csv: Path) -> pd.DataFrame:
    """Load and lightly validate the team context CSV.

    Parameters
    ----------
    context_csv : Path
        Path to the manually curated CSV (default: data/external/team_context_2026.csv).

    Returns
    -------
    pd.DataFrame
        Context table with at least a ``team`` column.  Missing input
        columns are logged as warnings; compute_metrics handles them
        gracefully downstream.
    """
    if not context_csv.exists():
        logger.warning(
            "_load_context: %s not found — returning empty DataFrame", context_csv
        )
        return pd.DataFrame(columns=["team"])

    ctx = pd.read_csv(context_csv)

    if "team" not in ctx.columns:
        logger.warning("_load_context: 'team' column missing from %s", context_csv)
        return pd.DataFrame(columns=["team"])

    for col in _REQUIRED_CONTEXT_COLS:
        if col not in ctx.columns:
            logger.warning(
                "_load_context: required column '%s' missing from context CSV — "
                "affected sub-metric(s) will be NaN",
                col,
            )

    # Coerce boolean columns — CSV stores them as strings (True/False).
    for bool_col in ("historic_first_time", "rival_top_team_underperformed"):
        if bool_col in ctx.columns:
            ctx[bool_col] = (
                ctx[bool_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(float)
            )

    return ctx


def _join_context_to_baseline(
    team_baseline_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join context columns onto the team baseline, logging unmatched teams.

    Parameters
    ----------
    team_baseline_df : pd.DataFrame
        Output of ``features.team_baseline.build_team_baseline``.
        Must contain a ``team`` column.
    context_df : pd.DataFrame
        Loaded context CSV.

    Returns
    -------
    pd.DataFrame
        Joined DataFrame indexed by team.  Teams in baseline but absent from
        context will have NaN for all context columns (no crash).
    """
    if "team" not in team_baseline_df.columns:
        logger.warning(
            "_join_context_to_baseline: 'team' column missing from team_baseline_df"
        )
        return context_df.copy()

    merged = team_baseline_df[["team"]].merge(context_df, on="team", how="left")

    n_missing = merged[_REQUIRED_CONTEXT_COLS[0]].isna().sum() if _REQUIRED_CONTEXT_COLS[0] in merged.columns else 0
    if n_missing:
        missing_teams = merged.loc[merged[_REQUIRED_CONTEXT_COLS[0]].isna(), "team"].tolist()
        logger.warning(
            "_join_context_to_baseline: %d team(s) have no context data and will "
            "receive NaN sub-metrics: %s",
            n_missing,
            missing_teams,
        )

    return merged


def build_underdog_features(
    team_baseline_df: pd.DataFrame,
    context_csv: Union[str, Path] = "data/external/team_context_2026.csv",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build UPI and sub-metric feature columns for every team.

    .. warning::
        The output columns are **features only**.  Do NOT apply UPI as a
        win-probability multiplier here.  Pass these columns into the logistic
        upset model (``models/upset_model.py``) and let it learn the weights
        from backtested data.

    Parameters
    ----------
    team_baseline_df : pd.DataFrame
        Output of ``features.team_baseline.build_team_baseline``.
        Used to establish the canonical team list and name mapping.
    context_csv : str or Path
        Path to the manually curated team context CSV.
        Defaults to ``data/external/team_context_2026.csv``.
    output_path : Path, optional
        Override default save path (``data/processed/underdog_features.csv``).

    Returns
    -------
    pd.DataFrame
        Columns: ``team, upset_potential_index, resource_gap_ratio,
        leadership_stability_score, cohesion_pressure_score,
        motivational_momentum_score``.
        Teams missing context data have NaN sub-metrics and UPI = 50.0
        (the compute_metrics fallback for insufficient data).
    """
    context_path = Path(context_csv)
    ctx = _load_context(context_path)

    if ctx.empty:
        logger.warning("build_underdog_features: context DataFrame empty — returning empty result")
        return pd.DataFrame(columns=_OUTPUT_COLS)

    joined = _join_context_to_baseline(team_baseline_df, ctx)

    logger.info("build_underdog_features: computing metrics for %d teams", len(joined))
    metrics_df = compute_metrics(joined)

    # Ensure output columns exist even if compute_metrics couldn't compute them.
    for col in _OUTPUT_COLS:
        if col not in metrics_df.columns:
            logger.warning(
                "build_underdog_features: output column '%s' not produced by "
                "compute_metrics — filling with NaN",
                col,
            )
            metrics_df[col] = np.nan

    result = metrics_df[_OUTPUT_COLS].copy()

    if output_path is None:
        output_path = PROCESSED_DIR / "underdog_features.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("build_underdog_features: saved %d rows to %s", len(result), output_path)

    return result
