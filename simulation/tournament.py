"""Tournament simulation engine for the WC 2026 Predictor.

Implements the FIFA 2026 World Cup format:
    - 48 teams in 12 groups of 4
    - Group stage: round-robin within each group (6 matches per group)
    - Advancement: top 2 per group (24) + 8 best 3rd-place teams = 32
    - Knockout: single elimination from round of 32 to final

Optimisations
-------------
- Group stage uses numpy arrays for tallies instead of dicts.
- Knockout avoids repeated predict() calls (CDFs are cached in model).
- No DataFrame construction during simulation hot path.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from models.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)


def simulate_group_stage(
    model: DixonColesModel,
    groups_df: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Simulate the group stage for all 12 groups.

    Returns DataFrame with: team, group, pts, gd, gf, group_rank.
    """
    if rng is None:
        rng = np.random.default_rng()

    standings_rows = []

    for group_name, group_teams in groups_df.groupby("group"):
        teams = group_teams["team"].tolist()
        n = len(teams)
        pts = np.zeros(n, dtype=int)
        gf = np.zeros(n, dtype=int)
        ga = np.zeros(n, dtype=int)

        # Round-robin: every pair plays once
        for i in range(n):
            for j in range(i + 1, n):
                hg, ag = model.simulate_score(teams[i], teams[j], neutral=True, rng=rng)
                gf[i] += hg
                ga[i] += ag
                gf[j] += ag
                ga[j] += hg

                if hg > ag:
                    pts[i] += 3
                elif hg == ag:
                    pts[i] += 1
                    pts[j] += 1
                else:
                    pts[j] += 3

        # Sort: pts desc, gd desc, gf desc, random tiebreak
        gd = gf - ga
        tiebreak = rng.random(n)
        # Create sort key (negate for descending)
        order = np.lexsort((tiebreak, gf, gd, pts))[::-1]

        for rank, idx in enumerate(order, 1):
            standings_rows.append({
                "team": teams[idx],
                "group": group_name,
                "pts": int(pts[idx]),
                "gd": int(gd[idx]),
                "gf": int(gf[idx]),
                "group_rank": rank,
            })

    return pd.DataFrame(standings_rows)


def build_knockout_bracket(
    standings: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> list[tuple[str, str]]:
    """Build the round-of-32 bracket from group standings.

    24 auto-qualifiers (top 2) + 8 best 3rd-place = 32 teams.
    Returns 16 matchups for the round of 32.
    """
    if rng is None:
        rng = np.random.default_rng()

    first = standings[standings["group_rank"] == 1]["team"].tolist()
    second = standings[standings["group_rank"] == 2]["team"].tolist()

    # Best 8 third-place teams
    thirds = standings[standings["group_rank"] == 3].copy()
    thirds = thirds.sort_values(
        ["pts", "gd", "gf"], ascending=[False, False, False]
    ).reset_index(drop=True)
    best_third = thirds.head(8)["team"].tolist()

    # Shuffle for variety within pools
    first_arr = np.array(first)
    second_arr = np.array(second)
    third_arr = np.array(best_third)
    rng.shuffle(first_arr)
    rng.shuffle(second_arr)
    rng.shuffle(third_arr)

    matchups = []
    # 8 first-place vs 8 best-third
    for i in range(8):
        matchups.append((first_arr[i], third_arr[i]))
    # 4 first-place vs 4 second-place
    for i in range(4):
        matchups.append((first_arr[8 + i], second_arr[i]))
    # 4 pairs of remaining second-place teams
    for i in range(4):
        matchups.append((second_arr[4 + i], second_arr[8 + i]))

    return matchups


def simulate_knockout(
    model: DixonColesModel,
    matchups: list[tuple[str, str]],
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Simulate single-elimination knockout. Returns winner name."""
    if rng is None:
        rng = np.random.default_rng()

    current_round = list(matchups)

    while len(current_round) > 1:
        next_round = []
        for home, away in current_round:
            hg, ag = model.simulate_score(home, away, neutral=True, rng=rng)
            if hg > ag:
                next_round.append(home)
            elif ag > hg:
                next_round.append(away)
            else:
                # Penalty shootout = coin flip
                next_round.append(home if rng.random() < 0.5 else away)
        current_round = [
            (next_round[i], next_round[i + 1])
            for i in range(0, len(next_round), 2)
        ]

    # Final
    home, away = current_round[0]
    hg, ag = model.simulate_score(home, away, neutral=True, rng=rng)
    if hg > ag:
        return home
    elif ag > hg:
        return away
    else:
        return home if rng.random() < 0.5 else away


def simulate_tournament(
    model: DixonColesModel,
    groups_df: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Simulate a complete World Cup (group + knockout).

    Returns dict with: winner, standings, knockout_teams.
    """
    if rng is None:
        rng = np.random.default_rng()

    standings = simulate_group_stage(model, groups_df, rng=rng)
    matchups = build_knockout_bracket(standings, rng=rng)
    knockout_teams = [t for pair in matchups for t in pair]
    winner = simulate_knockout(model, matchups, rng=rng)

    return {
        "winner": winner,
        "standings": standings,
        "knockout_teams": knockout_teams,
    }
