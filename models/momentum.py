"""Bayesian momentum adjustment for Dixon-Coles parameters.

Updates pre-tournament attack/defense estimates based on actual WC 2026
performance.  Uses the ratio of actual goals to model-expected goals as
a multiplicative adjustment on the log-scale parameters.

    alpha'_i = alpha_i + eta * log(actual_goals / expected_goals)
    beta'_i  = beta_i  + eta * log(expected_conceded / actual_conceded)

Where eta is a learning rate (0.3-0.5) that controls how much weight
to give tournament performance vs. the pre-tournament estimate.

This is principled: teams outperforming their expected output get a
boost; teams underperforming get lowered.  The opponent-quality
adjustment is inherent since expected goals already account for
the opposition's defense/attack strength.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)


def compute_momentum_adjustment(
    model: DixonColesModel,
    wc_results: pd.DataFrame,
    eta: float = 0.4,
    min_matches: int = 2,
) -> DixonColesModel:
    """Adjust model parameters based on actual WC tournament performance.

    Parameters
    ----------
    model : DixonColesModel
        Pre-tournament fitted model.
    wc_results : pd.DataFrame
        Actual WC 2026 results (from wc2026_results.csv).
        Columns: date, home_team, away_team, home_score, away_score, group.
    eta : float
        Learning rate (0.0 = no adjustment, 1.0 = full adjustment).
        Default 0.4 balances pre-tournament strength with current form.
    min_matches : int
        Minimum matches played to apply adjustment (avoid noise from 1 game).

    Returns
    -------
    DixonColesModel
        New model with adjusted attack/defense parameters.
    """
    teams_in_model = set(model._team_to_idx.keys())

    # Compute expected vs actual for each team
    team_stats: dict[str, dict] = {}

    for _, match in wc_results.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        if home not in teams_in_model or away not in teams_in_model:
            continue

        # Get model's expected goals for this matchup
        pred = model.predict(home, away, neutral=True)
        exp_home = pred["expected_home_goals"]
        exp_away = pred["expected_away_goals"]

        # Accumulate actual vs expected
        for team, actual_scored, actual_conceded, exp_scored, exp_conceded in [
            (home, match["home_score"], match["away_score"], exp_home, exp_away),
            (away, match["away_score"], match["home_score"], exp_away, exp_home),
        ]:
            if team not in team_stats:
                team_stats[team] = {
                    "actual_scored": 0,
                    "actual_conceded": 0,
                    "expected_scored": 0.0,
                    "expected_conceded": 0.0,
                    "matches": 0,
                }
            team_stats[team]["actual_scored"] += actual_scored
            team_stats[team]["actual_conceded"] += actual_conceded
            team_stats[team]["expected_scored"] += exp_scored
            team_stats[team]["expected_conceded"] += exp_conceded
            team_stats[team]["matches"] += 1

    # Compute adjustments
    new_attack = model.attack.copy()
    new_defense = model.defense.copy()
    adjustments = []

    for team, stats in team_stats.items():
        if stats["matches"] < min_matches:
            continue

        idx = model._team_to_idx[team]

        # Attack adjustment: log(actual_scored / expected_scored)
        # Clamp to avoid log(0)
        actual_s = max(stats["actual_scored"], 0.5)
        expected_s = max(stats["expected_scored"], 0.5)
        attack_adj = eta * np.log(actual_s / expected_s)

        # Defense adjustment: log(expected_conceded / actual_conceded)
        # Positive when conceding less than expected (better defense)
        actual_c = max(stats["actual_conceded"], 0.5)
        expected_c = max(stats["expected_conceded"], 0.5)
        defense_adj = eta * np.log(expected_c / actual_c)

        new_attack[idx] += attack_adj
        new_defense[idx] += defense_adj

        adjustments.append({
            "team": team,
            "matches": stats["matches"],
            "actual_scored": stats["actual_scored"],
            "expected_scored": round(stats["expected_scored"], 2),
            "actual_conceded": stats["actual_conceded"],
            "expected_conceded": round(stats["expected_conceded"], 2),
            "attack_adj": round(attack_adj, 4),
            "defense_adj": round(defense_adj, 4),
        })

    # Re-centre attacks
    new_attack = new_attack - new_attack.mean()

    # Log top adjustments
    adj_df = pd.DataFrame(adjustments).sort_values("attack_adj", ascending=False)
    logger.info(
        "momentum: adjusted %d teams (eta=%.2f). Top attack boosts:\n%s",
        len(adj_df), eta,
        adj_df.head(10)[["team", "matches", "actual_scored", "expected_scored", "attack_adj"]].to_string(index=False),
    )

    # Create new model with adjusted params
    adjusted_model = DixonColesModel(
        teams=model.teams,
        attack=new_attack,
        defense=new_defense,
        home_advantage=model.home_advantage,
        rho=model.rho,
    )

    return adjusted_model
