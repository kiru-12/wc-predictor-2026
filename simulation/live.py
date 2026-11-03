"""Live tournament simulation conditioned on actual WC 2026 results.

Instead of simulating the entire tournament from scratch, this module:
1. Loads actual match results played so far.
2. Computes current group standings from real data.
3. For groups with all 3 matchdays complete → locks final standings.
4. For groups with remaining matches → simulates only those.
5. Determines the 32 knockout qualifiers (actual + simulated).
6. Simulates the knockout bracket from the round of 32.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

# Each group has 4 teams playing 3 matches each = 6 matches total
_MATCHES_PER_GROUP = 6


def load_wc2026_results(path: str = "data/external/wc2026_results.csv") -> pd.DataFrame:
    """Load actual WC 2026 match results."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_standings_from_results(results: pd.DataFrame) -> pd.DataFrame:
    """Compute group standings from actual match results.

    Returns DataFrame with: team, group, pts, gd, gf, matches_played.
    """
    rows = []
    for group, matches in results.groupby("group"):
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        tally = {t: {"pts": 0, "gf": 0, "ga": 0, "mp": 0} for t in teams}

        for _, m in matches.iterrows():
            h, a = m["home_team"], m["away_team"]
            hg, ag = int(m["home_score"]), int(m["away_score"])
            tally[h]["gf"] += hg
            tally[h]["ga"] += ag
            tally[h]["mp"] += 1
            tally[a]["gf"] += ag
            tally[a]["ga"] += hg
            tally[a]["mp"] += 1

            if hg > ag:
                tally[h]["pts"] += 3
            elif hg == ag:
                tally[h]["pts"] += 1
                tally[a]["pts"] += 1
            else:
                tally[a]["pts"] += 3

        for t in teams:
            rows.append({
                "team": t,
                "group": group,
                "pts": tally[t]["pts"],
                "gf": tally[t]["gf"],
                "ga": tally[t]["ga"],
                "gd": tally[t]["gf"] - tally[t]["ga"],
                "matches_played": tally[t]["mp"],
            })

    return pd.DataFrame(rows)


def get_remaining_matches(
    groups_df: pd.DataFrame,
    played_results: pd.DataFrame,
) -> dict[str, list[tuple[str, str]]]:
    """Determine which matches still need to be played in each group.

    Returns dict: group_letter -> list of (home, away) tuples not yet played.
    """
    remaining = {}
    for group, teams_in_group in groups_df.groupby("group"):
        teams = teams_in_group["team"].tolist()
        group_results = played_results[played_results["group"] == group]

        # All possible pairs (order doesn't matter for matching)
        played_pairs = set()
        for _, m in group_results.iterrows():
            played_pairs.add(frozenset([m["home_team"], m["away_team"]]))

        all_pairs = [frozenset(pair) for pair in combinations(teams, 2)]
        unplayed = [pair for pair in all_pairs if pair not in played_pairs]

        if unplayed:
            remaining[group] = [(list(p)[0], list(p)[1]) for p in unplayed]

    return remaining


def simulate_live_group_stage(
    model: DixonColesModel,
    groups_df: pd.DataFrame,
    played_results: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Simulate group stage conditioned on actual results.

    Groups that are complete use actual standings.
    Groups with remaining matches simulate only those matches.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Start from actual standings
    standings = compute_standings_from_results(played_results)
    remaining = get_remaining_matches(groups_df, played_results)

    # Simulate remaining matches
    for group, matches in remaining.items():
        for home, away in matches:
            hg, ag = model.simulate_score(home, away, neutral=True, rng=rng)

            # Update standings
            h_mask = (standings["team"] == home) & (standings["group"] == group)
            a_mask = (standings["team"] == away) & (standings["group"] == group)

            standings.loc[h_mask, "gf"] += hg
            standings.loc[h_mask, "ga"] += ag
            standings.loc[a_mask, "gf"] += ag
            standings.loc[a_mask, "ga"] += hg

            if hg > ag:
                standings.loc[h_mask, "pts"] += 3
            elif hg == ag:
                standings.loc[h_mask, "pts"] += 1
                standings.loc[a_mask, "pts"] += 1
            else:
                standings.loc[a_mask, "pts"] += 3

    # Recompute GD
    standings["gd"] = standings["gf"] - standings["ga"]
    all_standings = []
    for group, grp_df in standings.groupby("group"):
        grp = grp_df.copy()
        grp["_tiebreak"] = rng.random(len(grp))
        grp = grp.sort_values(
            ["pts", "gd", "gf", "_tiebreak"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        grp["group_rank"] = range(1, len(grp) + 1)
        grp = grp.drop(columns=["_tiebreak"])
        all_standings.append(grp)

    return pd.concat(all_standings, ignore_index=True)


def simulate_live_tournament(
    model: DixonColesModel,
    groups_df: pd.DataFrame,
    played_results: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Simulate the remainder of the WC 2026 from current state."""
    if rng is None:
        rng = np.random.default_rng()

    from simulation.tournament import build_knockout_bracket, simulate_knockout

    standings = simulate_live_group_stage(model, groups_df, played_results, rng=rng)
    matchups = build_knockout_bracket(standings, rng=rng)
    knockout_teams = [t for pair in matchups for t in pair]
    winner = simulate_knockout(model, matchups, rng=rng)

    return {
        "winner": winner,
        "standings": standings,
        "knockout_teams": knockout_teams,
    }


def run_live_simulations(
    model: DixonColesModel,
    groups_df: pd.DataFrame,
    played_results: pd.DataFrame,
    n: int = 100_000,
    seed: int = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run Monte Carlo on the remaining tournament.

    Much faster than full simulation since groups A, B, C are locked.
    """
    from tqdm import tqdm

    all_teams = groups_df["team"].tolist()
    win_counts = np.zeros(len(all_teams), dtype=int)
    knockout_counts = np.zeros(len(all_teams), dtype=int)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    # Pre-compute matchup CDFs
    model.precompute_matchups(all_teams, neutral=True)

    logger.info("run_live_simulations: %d sims, seed=%d", n, seed)

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n)

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, desc="Live simulating", unit="sim", miniters=100)

    for i in iterator:
        rng = np.random.default_rng(child_seeds[i])
        result = simulate_live_tournament(model, groups_df, played_results, rng=rng)

        win_counts[team_to_idx[result["winner"]]] += 1
        for team in result["knockout_teams"]:
            knockout_counts[team_to_idx[team]] += 1

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
    return results_df.reset_index(drop=True)
