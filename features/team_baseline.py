"""Team baseline features for the WC 2026 Predictor.

Computes a per-team feature set from international match results:

  1. ``elo_national``       – National-team ELO iteratively computed from results.
  2. ``form_score``         – Weighted average form over the last *n* matches.
  3. ``continental_winner`` – 1 if the team won a continental championship in the
                              current WC cycle, else 0.
  4. ``unbeaten_streak``    – Consecutive matches without a loss (backwards from
                              *as_of*).
  5. ``wc_qualifier_gd``    – Net goal difference across all WC-qualifier matches
                              in the cycle.

Combined score
--------------
``baseline_score`` is the **equal-weight mean of z-scored sub-features**.
This is a deliberately transparent, assumption-free aggregation intended
for exploratory use and as an initial ranking signal.  All feature weights
will be replaced by values learned / calibrated during backtesting (see
``models/`` layer).  No hardcoded blend constants are used here.

Point-in-time guarantee
-----------------------
Every function accepts an ``as_of`` cutoff and excludes any match played
on or after that date.  Callers must pass the correct pre-tournament date
to avoid data leakage.  This guarantee is stated explicitly in each
function's docstring.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Tournament classification
# ---------------------------------------------------------------------------

#: Canonical tournament categories used for both ELO K-factors and form weights.
_WC_QUALIFIER = "wc_qualifier"
_CONTINENTAL = "continental_cup"
_NATIONS_LEAGUE = "nations_league"
_FRIENDLY = "friendly"
_OTHER = "other"

#: Match-result weight applied when computing ``form_score``.
TOURNAMENT_FORM_WEIGHTS: dict[str, float] = {
    _WC_QUALIFIER: 1.5,
    _CONTINENTAL: 2.0,
    _NATIONS_LEAGUE: 1.2,
    _FRIENDLY: 0.4,
    _OTHER: 1.0,
}

#: ELO K-factor per tournament category.
ELO_K: dict[str, float] = {
    _WC_QUALIFIER: 40.0,
    _CONTINENTAL: 50.0,
    _NATIONS_LEAGUE: 35.0,
    _FRIENDLY: 20.0,
    _OTHER: 30.0,
}

# Substrings that identify each category (lower-cased).
_WC_QUAL_KEYWORDS = ("world cup qual",)
_CONTINENTAL_KEYWORDS = (
    "copa am",          # Copa América
    "uefa euro",        # UEFA European Championship
    "african cup of nations",
    "afc asian cup",
    "gold cup",
    "ofc nations cup",
    "oceania nations cup",
    "concacaf championship",
    "aff championship",
    "confederations cup",
)
_NATIONS_LEAGUE_KEYWORDS = ("nations league",)
_FRIENDLY_KEYWORDS = ("friendly",)


def classify_tournament(tournament: str) -> str:
    """Map a raw tournament name to a canonical category string.

    Parameters
    ----------
    tournament : str
        Raw ``tournament`` value from the international results CSV.

    Returns
    -------
    str
        One of ``'wc_qualifier'``, ``'continental_cup'``, ``'nations_league'``,
        ``'friendly'``, ``'other'``.
    """
    t = tournament.lower()
    if any(k in t for k in _WC_QUAL_KEYWORDS):
        return _WC_QUALIFIER
    if any(k in t for k in _CONTINENTAL_KEYWORDS):
        return _CONTINENTAL
    if any(k in t for k in _NATIONS_LEAGUE_KEYWORDS):
        return _NATIONS_LEAGUE
    if any(k in t for k in _FRIENDLY_KEYWORDS):
        return _FRIENDLY
    return _OTHER


# ---------------------------------------------------------------------------
# Internal helper: team-centric long format
# ---------------------------------------------------------------------------

def _to_team_centric(results: pd.DataFrame) -> pd.DataFrame:
    """Reshape results from match-row to team-row format.

    Each match produces two rows — one for the home side, one for the away
    side — with columns:
    ``date, team, opponent, goals_for, goals_against, is_home,
    tournament, category, points, goal_diff``

    Missing score values are coerced to NaN with a warning rather than
    crashing.

    Parameters
    ----------
    results : pd.DataFrame
        Raw international results with columns ``date, home_team, away_team,
        home_score, away_score, tournament``.

    Returns
    -------
    pd.DataFrame
        Long-format team-centric DataFrame sorted by date ascending.
    """
    needed = {"date", "home_team", "away_team", "home_score", "away_score", "tournament"}
    missing = needed - set(results.columns)
    if missing:
        logger.warning("_to_team_centric: missing columns %s — returning empty frame", missing)
        return pd.DataFrame()

    home = results[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].copy()
    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "goals_for",
            "away_score": "goals_against",
        }
    )
    home["is_home"] = True

    away = results[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].copy()
    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "goals_for",
            "home_score": "goals_against",
        }
    )
    away["is_home"] = False

    tc = pd.concat([home, away], ignore_index=True)

    # Coerce scores to numeric — bad rows become NaN rather than crashing.
    for col in ("goals_for", "goals_against"):
        tc[col] = pd.to_numeric(tc[col], errors="coerce")
        n_bad = tc[col].isna().sum()
        if n_bad:
            logger.warning("_to_team_centric: %d NaN values in '%s' after coercion", n_bad, col)

    tc["category"] = tc["tournament"].map(classify_tournament)
    tc["goal_diff"] = tc["goals_for"] - tc["goals_against"]

    # Result points: win=1, draw=0.5, loss=0, NaN stays NaN.
    conditions = [
        tc["goal_diff"] > 0,
        tc["goal_diff"] == 0,
        tc["goal_diff"] < 0,
    ]
    tc["points"] = np.select(conditions, [1.0, 0.5, 0.0], default=np.nan)

    tc["date"] = pd.to_datetime(tc["date"])
    # Sort by [date, team, opponent] for a fully deterministic row order so that
    # iterative computations (e.g. ELO) are stable regardless of input ordering.
    return tc.sort_values(["date", "team", "opponent"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# National-team ELO
# ---------------------------------------------------------------------------

def compute_national_elo(
    results: pd.DataFrame,
    as_of: pd.Timestamp,
    initial: float = 1500.0,
) -> pd.Series:
    """Compute national-team ELO ratings iteratively from match results.

    Point-in-time guarantee: only matches with ``date < as_of`` are
    used.  Pass the tournament start date to avoid data leakage.

    Uses the standard ELO update formula::

        E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
        R_A_new = R_A + K * (actual_A - E_A)

    where ``K`` is tournament-category-dependent (see ``ELO_K``).

    Parameters
    ----------
    results : pd.DataFrame
        International results as returned by ``load_international_results``.
    as_of : pd.Timestamp
        Exclusive upper bound on match dates.
    initial : float
        Starting ELO for any team before its first match.

    Returns
    -------
    pd.Series
        Indexed by team name, values are float ELO ratings.  Name is
        ``'elo_national'``.
    """
    tc = _to_team_centric(results)
    if tc.empty:
        return pd.Series(dtype=float, name="elo_national")

    tc = tc[tc["date"] < as_of].dropna(subset=["points"])

    ratings: dict[str, float] = {}

    for _, row in tc.iterrows():
        team = row["team"]
        opp = row["opponent"]
        r_team = ratings.get(team, initial)
        r_opp = ratings.get(opp, initial)

        k = ELO_K[row["category"]]
        expected = 1.0 / (1.0 + 10.0 ** ((r_opp - r_team) / 400.0))
        # Opponent expected is symmetric; update only this team's row since
        # the opponent row will be processed in its own iteration.
        ratings[team] = r_team + k * (row["points"] - expected)
        # Seed opponent if unseen so it doesn't start from initial on next match
        if opp not in ratings:
            ratings[opp] = initial

    series = pd.Series(ratings, name="elo_national").rename_axis("team")
    logger.info("compute_national_elo: %d teams rated as of %s", len(series), as_of.date())
    return series


# ---------------------------------------------------------------------------
# Recent weighted form
# ---------------------------------------------------------------------------

def compute_recent_form(
    tc: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp,
    n: int = 20,
) -> float:
    """Compute weighted-average form score for a team over their last *n* matches.

    Point-in-time guarantee: only matches with ``date < as_of`` are used.

    Result points (win=1, draw=0.5, loss=0) are multiplied by the match's
    tournament-category weight (see ``TOURNAMENT_FORM_WEIGHTS``), then the
    weighted average is computed::

        form_score = sum(points_i * w_i) / sum(w_i)   over last n matches

    Parameters
    ----------
    tc : pd.DataFrame
        Team-centric long format from ``_to_team_centric``.
    team : str
        Team name exactly as it appears in the data.
    as_of : pd.Timestamp
        Exclusive upper bound on match dates.
    n : int
        Number of most-recent matches to include.

    Returns
    -------
    float
        Weighted form score in [0, 1], or ``nan`` if no matches found.
    """
    subset = tc[(tc["team"] == team) & (tc["date"] < as_of)].dropna(subset=["points"])
    if subset.empty:
        logger.warning("compute_recent_form: no matches for '%s' before %s", team, as_of.date())
        return float("nan")

    last_n = subset.tail(n)
    weights = last_n["category"].map(TOURNAMENT_FORM_WEIGHTS)
    w_sum = weights.sum()
    if w_sum == 0:
        return float("nan")

    return float((last_n["points"] * weights).sum() / w_sum)


# ---------------------------------------------------------------------------
# Unbeaten streak
# ---------------------------------------------------------------------------

def compute_unbeaten_streak(
    tc: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp,
) -> int:
    """Count consecutive matches without a loss going backwards from *as_of*.

    Point-in-time guarantee: only matches with ``date < as_of`` are used.

    Parameters
    ----------
    tc : pd.DataFrame
        Team-centric long format from ``_to_team_centric``.
    team : str
        Team name.
    as_of : pd.Timestamp
        Exclusive upper bound on match dates.

    Returns
    -------
    int
        Number of consecutive unbeaten games (0 if last match was a loss or
        no matches found).
    """
    subset = (
        tc[(tc["team"] == team) & (tc["date"] < as_of)]
        .dropna(subset=["points"])
        .sort_values("date", ascending=False)
    )
    if subset.empty:
        return 0

    streak = 0
    for pts in subset["points"]:
        if pts == 0.0:  # loss
            break
        streak += 1
    return streak


# ---------------------------------------------------------------------------
# WC qualifier goal difference
# ---------------------------------------------------------------------------

def compute_wc_qualifier_gd(
    tc: pd.DataFrame,
    team: str,
    cycle_start: int,
    cycle_end: int,
) -> float:
    """Sum goal difference in WC-qualification matches within a tournament cycle.

    Point-in-time guarantee: ``cycle_end`` must be set to the calendar year
    *before* the World Cup (e.g., 2021 for WC 2022) so no WC matches are
    included.

    Parameters
    ----------
    tc : pd.DataFrame
        Team-centric long format from ``_to_team_centric``.
    team : str
        Team name.
    cycle_start : int
        First year of the qualifying cycle (inclusive).
    cycle_end : int
        Last year of the qualifying cycle (inclusive).

    Returns
    -------
    float
        Cumulative goal difference; 0.0 if no qualifying matches found.
    """
    mask = (
        (tc["team"] == team)
        & (tc["category"] == _WC_QUALIFIER)
        & (tc["date"].dt.year >= cycle_start)
        & (tc["date"].dt.year <= cycle_end)
    )
    subset = tc[mask].dropna(subset=["goal_diff"])
    if subset.empty:
        return 0.0
    return float(subset["goal_diff"].sum())


# ---------------------------------------------------------------------------
# Continental cup winner flag
# ---------------------------------------------------------------------------

# Tournaments considered continental championships for flagging purposes.
_CONTINENTAL_TOURNAMENT_NAMES = {
    "Copa América",
    "UEFA Euro",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "OFC Nations Cup",
    "Oceania Nations Cup",
    "CONCACAF Championship",
    "AFF Championship",
    "Confederations Cup",
}


def flag_continental_winner(
    results: pd.DataFrame,
    team: str,
    cycle_start: int,
    cycle_end: int,
) -> int:
    """Return 1 if *team* won a continental championship within the cycle.

    Point-in-time guarantee: only tournaments that *finished* on or before
    ``cycle_end``-12-31 are considered.

    The winner is identified as the team that won the *final* match of a
    tournament edition (latest date within the tournament-year group).
    If the final ended in a draw (penalty result not in data), a warning
    is logged and that tournament is skipped.

    Parameters
    ----------
    results : pd.DataFrame
        Raw international results.
    team : str
        Team name.
    cycle_start : int
        First calendar year to search (inclusive).
    cycle_end : int
        Last calendar year to search (inclusive).

    Returns
    -------
    int
        1 if the team won at least one continental title in the cycle, else 0.
    """
    needed = {"date", "home_team", "away_team", "home_score", "away_score", "tournament"}
    missing = needed - set(results.columns)
    if missing:
        logger.warning("flag_continental_winner: missing columns %s", missing)
        return 0

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Only continental tournaments within the cycle.
    mask = (
        df["tournament"].isin(_CONTINENTAL_TOURNAMENT_NAMES)
        & (df["year"] >= cycle_start)
        & (df["year"] <= cycle_end)
    )
    cont = df[mask]
    if cont.empty:
        return 0

    for (tourn, yr), grp in cont.groupby(["tournament", "year"]):
        final = grp.sort_values("date").iloc[-1]
        h_score = pd.to_numeric(final["home_score"], errors="coerce")
        a_score = pd.to_numeric(final["away_score"], errors="coerce")

        if pd.isna(h_score) or pd.isna(a_score) or h_score == a_score:
            logger.warning(
                "flag_continental_winner: %s %d final ended in a draw or "
                "has no score — cannot determine winner from results data",
                tourn,
                yr,
            )
            continue

        winner = final["home_team"] if h_score > a_score else final["away_team"]
        if winner == team:
            logger.info("flag_continental_winner: '%s' won %s %d", team, tourn, yr)
            return 1

    return 0


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_team_baseline(
    results: pd.DataFrame,
    teams: list[str],
    as_of: str,
    wc_cycle_start: int,
    wc_cycle_end: int,
    n_form_matches: int = 20,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build a one-row-per-team baseline feature DataFrame.

    Combined score
    --------------
    ``baseline_score`` is the **equal-weight mean of z-scored sub-features**:
    ``elo_national``, ``form_score``, ``unbeaten_streak``,
    ``wc_qualifier_gd``, ``continental_winner``.  Columns with NaN for a
    specific team contribute 0 to that team's z-score mean (i.e. NaN
    sub-features are skipped, not propagated).  This temporary aggregation
    should be superseded by the learned-weight model in ``models/``.

    Point-in-time guarantee: ``as_of`` is applied as an exclusive upper-
    bound cutoff to every sub-feature computation, so no match played on
    or after ``as_of`` influences the output.

    Parameters
    ----------
    results : pd.DataFrame
        Output of ``load_international_results``.
    teams : list[str]
        Teams to score (must match names in the results data).
    as_of : str
        ISO date string ``'YYYY-MM-DD'``.  Cutoff date — the tournament
        start date for the prediction target.
    wc_cycle_start : int
        First year of the WC qualifying cycle (e.g., 2018 for WC 2022).
    wc_cycle_end : int
        Last year of the WC qualifying cycle (e.g., 2021 for WC 2022).
    n_form_matches : int
        Number of most-recent matches for weighted form calculation.
    output_path : Path or None
        If provided, the DataFrame is saved to this path as CSV.  Defaults
        to ``data/processed/team_baseline.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``team, elo_national, form_score, unbeaten_streak,
        wc_qualifier_gd, continental_winner, baseline_score``.
    """
    as_of_ts = pd.Timestamp(as_of)

    logger.info(
        "build_team_baseline: %d teams, as_of=%s, cycle=%d–%d",
        len(teams),
        as_of,
        wc_cycle_start,
        wc_cycle_end,
    )

    # --- Pre-compute reusable artefacts ----------------------------------------
    tc = _to_team_centric(results)
    elo_series = compute_national_elo(results, as_of=as_of_ts)

    rows = []
    for team in teams:
        elo = elo_series.get(team, float("nan"))
        if pd.isna(elo):
            logger.warning("build_team_baseline: no ELO data for '%s'", team)

        form = compute_recent_form(tc, team, as_of=as_of_ts, n=n_form_matches)
        streak = compute_unbeaten_streak(tc, team, as_of=as_of_ts)
        qgd = compute_wc_qualifier_gd(tc, team, wc_cycle_start, wc_cycle_end)
        cont_win = flag_continental_winner(results, team, wc_cycle_start, wc_cycle_end)

        rows.append(
            {
                "team": team,
                "elo_national": elo,
                "form_score": form,
                "unbeaten_streak": float(streak),
                "wc_qualifier_gd": qgd,
                "continental_winner": float(cont_win),
            }
        )

    df = pd.DataFrame(rows)

    # --- Z-score each sub-feature and combine ----------------------------------
    feature_cols = ["elo_national", "form_score", "unbeaten_streak",
                    "wc_qualifier_gd", "continental_winner"]

    z_parts = []
    for col in feature_cols:
        mu = df[col].mean()
        sigma = df[col].std(ddof=0)
        if sigma == 0 or pd.isna(sigma):
            z = pd.Series(0.0, index=df.index)
        else:
            z = (df[col] - mu) / sigma
        z_parts.append(z.rename(col))

    z_df = pd.concat(z_parts, axis=1)
    # skipna=True: NaN sub-features do not propagate; they are excluded
    # from the mean for that team.
    df["baseline_score"] = z_df.mean(axis=1, skipna=True)

    # --- Save ------------------------------------------------------------------
    if output_path is None:
        output_path = PROCESSED_DIR / "team_baseline.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("build_team_baseline: saved %d rows to %s", len(df), output_path)

    return df
