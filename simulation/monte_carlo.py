"""Monte Carlo simulation engine for the WC 2026 Predictor.

Optimisations
-------------
- Pre-computes all pairwise matchup CDFs before the loop (2,256 for 48 teams).
- Uses SeedSequence for reproducible independent streams.
- Single loop with minimal allocations per iteration.
- tqdm progress bar with ETA.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.dixon_coles import DixonColesModel
from simulation.tournament import simulate_tournament

logger = logging.getLogger(__name__)


def run_simulations(
    model: DixonColesModel,
    groups_df: pd.DataFrame,
    n: int = 100_000,
    seed: int = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run Monte Carlo tournament simulations.

    Parameters
    ----------
    model : DixonColesModel
        Fitted model (call ``model.precompute_matchups(teams)`` before
        this for maximum speed — done automatically if not already cached).
    groups_df : pd.DataFrame
        Group assignments (columns: team, group).
    n : int
        Number of simulations.
    seed : int
        Master random seed.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        Columns: team, win_count, win_pct, knockout_count, knockout_pct,
        group_exit_count, group_exit_pct.  Sorted by win_pct descending.
    """
    all_teams = groups_df["team"].tolist()

    # Verify all teams are in the model
    missing = [t for t in all_teams if t not in model._team_to_idx]
    if missing:
        raise ValueError(
            f"{len(missing)} teams not in model: {missing[:5]}... "
            "Fit the model on data that includes these teams."
        )

    # Pre-compute all pairwise CDFs (the key optimisation)
    model.precompute_matchups(all_teams, neutral=True)

    logger.info("run_simulations: %d simulations, seed=%d", n, seed)

    # Counters
    win_counts = np.zeros(len(all_teams), dtype=int)
    knockout_counts = np.zeros(len(all_teams), dtype=int)
    team_to_local_idx = {t: i for i, t in enumerate(all_teams)}

    # Reproducible sub-streams
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n)

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, desc="Simulating", unit="sim", miniters=100)

    for i in iterator:
        rng = np.random.default_rng(child_seeds[i])
        result = simulate_tournament(model, groups_df, rng=rng)

        win_counts[team_to_local_idx[result["winner"]]] += 1
        for team in result["knockout_teams"]:
            knockout_counts[team_to_local_idx[team]] += 1

    # Build results
    rows = []
    for i, team in enumerate(all_teams):
        wins = int(win_counts[i])
        ko = int(knockout_counts[i])
        rows.append({
            "team": team,
            "win_count": wins,
            "win_pct": 100.0 * wins / n,
            "knockout_count": ko,
            "knockout_pct": 100.0 * ko / n,
            "group_exit_count": n - ko,
            "group_exit_pct": 100.0 * (n - ko) / n,
        })

    results_df = pd.DataFrame(rows).sort_values("win_pct", ascending=False)
    results_df = results_df.reset_index(drop=True)

    logger.info(
        "run_simulations: done. Top-5: %s",
        results_df.head(5)[["team", "win_pct"]].to_dict("records"),
    )

    return results_df
