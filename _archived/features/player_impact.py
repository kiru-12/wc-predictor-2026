"""Player impact features for the WC 2026 Predictor.

Computes a per-team player impact score from club-season statistics.

Input DataFrame schema
----------------------
Every public function that takes ``players_df`` expects a flat DataFrame with
the following columns.  Missing optional columns are logged as warnings and the
corresponding sub-scores become NaN (they do NOT crash).

Required
~~~~~~~~
``player``         — unique player name / ID string
``national_team``  — FIFA country name (must match team names used elsewhere)
``club``           — club name
``minutes``        — total minutes played in the season (float, >= 0)
``starts``         — number of starts (int)
``matches``        — number of appearances (int)

Optional (NaN if absent)
~~~~~~~~~~~~~~~~~~~~~~~~~
``goals``          — season goals (raw count)
``assists``        — season assists (raw count)
``xg``             — season expected goals (raw)
``xa``             — season expected assists (raw)
``avg_rating``     — match-average rating on a 0–10 scale.
                     When sourced from SoFIFA: divide FIFA overall (0–99) by 10
                     before passing in.  No SofaScore data is used anywhere.
``is_ucl_uel_club``— bool; True when the player's club competed in UCL or UEL
                     during the season.

Pipeline summary
----------------
1. ``normalise_per90``          per-90 columns added (goals90, assists90, …)
2. ``flag_star_players``        is_star column; any of three criteria
3. ``compute_star_scores``      star_score column (with UCL/UEL 1.3× multiplier)
4. ``compute_squad_score``      top-11 mean + depth bonus from players 12-23
5. ``compute_star_dependency``  top_score / total_squad — flag if > 0.35
6. ``build_player_impact``      one row per national team; writes CSV

Combination formula
-------------------
``player_impact = 0.6 × squad_score + 0.4 × top_star_score``

These weights are specified in the project requirements.  They are not learned
from data and should be replaced by calibrated weights once backtesting is
implemented.

Depth bonus
-----------
``depth_bonus = mean(star_score, players ranked 12–23) × DEPTH_WEIGHT``
where ``DEPTH_WEIGHT = 0.30`` (design choice; replace with calibrated value
once backtesting is in place).
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soccerdata as sd

logger = logging.getLogger(__name__)

RAW_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum starts fraction to qualify as "regular starter" for UCL/UEL flag.
UCL_UEL_MIN_START_FRACTION: float = 0.60

#: avg_rating threshold for star flag.
AVG_RATING_STAR_THRESHOLD: float = 7.5

#: UCL/UEL minutes multiplier.
UCL_UEL_MULTIPLIER: float = 1.30

#: Weight for depth-squad players (12–23) in squad_score.
DEPTH_WEIGHT: float = 0.30

#: Star-dependency threshold above which a team is flagged as dependent.
STAR_DEPENDENCY_THRESHOLD: float = 0.35

# ---------------------------------------------------------------------------
# Per-90 normalisation
# ---------------------------------------------------------------------------

_PER90_COLS: list[str] = ["goals", "assists", "xg", "xa"]


def normalise_per90(players_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-90-minute versions of counting stats.

    Divides raw counting columns by ``minutes / 90``.  If ``minutes`` is zero
    or missing for a row, the per-90 value is NaN.  Missing source columns are
    logged as warnings and their per-90 equivalents will be NaN.

    Parameters
    ----------
    players_df : pd.DataFrame
        Player-level DataFrame with at least a ``minutes`` column.

    Returns
    -------
    pd.DataFrame
        A copy with extra columns ``goals90``, ``assists90``, ``xg90``,
        ``xa90``.
    """
    df = players_df.copy()

    if "minutes" not in df.columns:
        logger.warning("normalise_per90: 'minutes' column missing — all per-90 values NaN")
        for col in _PER90_COLS:
            df[f"{col}90"] = np.nan
        return df

    minutes = pd.to_numeric(df["minutes"], errors="coerce")
    ninety = minutes / 90.0
    # Avoid division by zero — treat 0 minutes as NaN.
    ninety = ninety.replace(0, np.nan)

    for col in _PER90_COLS:
        if col not in df.columns:
            logger.warning(
                "normalise_per90: '%s' column missing — '%s90' will be NaN", col, col
            )
            df[f"{col}90"] = np.nan
        else:
            raw = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}90"] = raw / ninety

    return df


# ---------------------------------------------------------------------------
# Star-player flag
# ---------------------------------------------------------------------------

def flag_star_players(
    players_df: pd.DataFrame,
    ucl_uel_clubs: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Add an ``is_star`` boolean column to the player DataFrame.

    A player is flagged as a star if *any* of the following conditions holds:

    1. **Top performer**: ranked in the top-3 goal scorers OR top-3 assisters
       within their league–season group (uses raw goals/assists).
    2. **High rating**: ``avg_rating >= 7.5``.  Source must not be SofaScore;
       pass SoFIFA overall rating ÷ 10 or leave as NaN to skip this criterion.
    3. **UCL/UEL regular starter**: ``is_ucl_uel_club`` is True AND
       ``starts / matches >= 0.60``.

    Missing columns are handled gracefully (criterion skipped, not a crash).

    Parameters
    ----------
    players_df : pd.DataFrame
        Must already contain ``normalise_per90`` output (or at minimum ``goals``
        and ``assists`` for criterion 1, ``avg_rating`` for criterion 2, and
        ``is_ucl_uel_club`` + ``starts`` + ``matches`` for criterion 3).
    ucl_uel_clubs : set[str], optional
        Override set of club names that play in UCL/UEL.  When provided,
        ``is_ucl_uel_club`` is (re)computed from this set.

    Returns
    -------
    pd.DataFrame
        Copy with ``is_star`` (bool) column added.
    """
    df = players_df.copy()

    # Optionally (re)compute is_ucl_uel_club from a provided club set.
    if ucl_uel_clubs is not None:
        if "club" not in df.columns:
            logger.warning("flag_star_players: 'club' column missing — is_ucl_uel_club=False")
            df["is_ucl_uel_club"] = False
        else:
            df["is_ucl_uel_club"] = df["club"].isin(ucl_uel_clubs)

    # --- Criterion 1: top-3 scorer or assister within league-season ----------
    c1 = pd.Series(False, index=df.index)
    group_cols = [c for c in ("league", "season") if c in df.columns]
    for col in ("goals", "assists"):
        if col in df.columns:
            raw = pd.to_numeric(df[col], errors="coerce")
            if group_cols:
                rank = raw.groupby([df[g] for g in group_cols]).rank(
                    method="min", ascending=False, na_option="bottom"
                )
            else:
                rank = raw.rank(method="min", ascending=False, na_option="bottom")
            c1 = c1 | (rank <= 3)
        else:
            logger.warning(
                "flag_star_players: '%s' column missing — top-performer criterion skipped",
                col,
            )

    # --- Criterion 2: avg_rating >= threshold --------------------------------
    if "avg_rating" in df.columns:
        rating = pd.to_numeric(df["avg_rating"], errors="coerce")
        c2 = rating >= AVG_RATING_STAR_THRESHOLD
    else:
        logger.warning(
            "flag_star_players: 'avg_rating' missing — rating criterion skipped "
            "(no SofaScore is used; pass SoFIFA overall÷10 or leave absent)"
        )
        c2 = pd.Series(False, index=df.index)

    # --- Criterion 3: UCL/UEL regular starter --------------------------------
    if "is_ucl_uel_club" in df.columns:
        is_uel = df["is_ucl_uel_club"].fillna(False).astype(bool)
        if "starts" in df.columns and "matches" in df.columns:
            starts = pd.to_numeric(df["starts"], errors="coerce")
            matches = pd.to_numeric(df["matches"], errors="coerce")
            start_frac = (starts / matches.replace(0, np.nan)).fillna(0)
            c3 = is_uel & (start_frac >= UCL_UEL_MIN_START_FRACTION)
        else:
            logger.warning(
                "flag_star_players: 'starts'/'matches' missing — UCL/UEL criterion "
                "uses club flag only (all UCL/UEL players flagged as stars)"
            )
            c3 = is_uel
    else:
        logger.warning(
            "flag_star_players: 'is_ucl_uel_club' column missing — UCL/UEL criterion skipped"
        )
        c3 = pd.Series(False, index=df.index)

    df["is_star"] = (c1 | c2 | c3).astype(bool)
    logger.info(
        "flag_star_players: %d / %d players flagged as stars",
        df["is_star"].sum(),
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Star score
# ---------------------------------------------------------------------------

_SCORE_WEIGHTS: dict[str, float] = {
    "goals90":   1.0,
    "assists90": 0.7,
    "xg90":      0.8,
    "xa90":      0.6,
    "avg_rating": 2.0,
}


def compute_star_scores(players_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``star_score`` for every player.

    Formula (as specified)::

        raw_score = 1.0×goals90 + 0.7×assists90 + 0.8×xg90 + 0.6×xa90
                    + 2.0×avg_rating

    Terms whose source column is NaN contribute 0 to that player's score
    (rather than propagating NaN), so a player with no xG data is not
    penalised — they simply receive no xG credit.

    UCL/UEL multiplier: if ``is_ucl_uel_club`` is True the raw score is
    multiplied by ``UCL_UEL_MULTIPLIER`` (1.30).

    Parameters
    ----------
    players_df : pd.DataFrame
        Must have been processed by ``normalise_per90``.  ``is_ucl_uel_club``
        should be present; if absent a warning is logged and no multiplier
        is applied.

    Returns
    -------
    pd.DataFrame
        Copy with ``star_score`` column added.
    """
    df = players_df.copy()

    score = pd.Series(0.0, index=df.index)
    for col, weight in _SCORE_WEIGHTS.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            score = score + weight * vals
        else:
            logger.warning(
                "compute_star_scores: '%s' missing — term omitted from star_score", col
            )

    # UCL/UEL multiplier
    if "is_ucl_uel_club" in df.columns:
        is_uel = df["is_ucl_uel_club"].fillna(False).astype(bool)
        score = score * np.where(is_uel, UCL_UEL_MULTIPLIER, 1.0)
    else:
        logger.warning(
            "compute_star_scores: 'is_ucl_uel_club' missing — UCL/UEL multiplier not applied"
        )

    df["star_score"] = score
    return df


# ---------------------------------------------------------------------------
# Squad-level aggregation
# ---------------------------------------------------------------------------

def compute_squad_score(
    team_scores: pd.Series,
    n_top: int = 11,
) -> float:
    """Compute squad score from a Series of per-player star_score values.

    Calculation::

        top_scores   = top n_top values (sorted descending)
        depth_scores = values ranked n_top+1 to n_top+12 (up to 12 bench players)
        squad_score  = mean(top_scores) + DEPTH_WEIGHT × mean(depth_scores)

    ``DEPTH_WEIGHT`` is 0.30 (see module-level constant).  If fewer than
    ``n_top`` players exist the available ones are used.  The depth term is
    zero if there are no bench-range players.

    Parameters
    ----------
    team_scores : pd.Series
        star_score values for one national team.
    n_top : int
        Number of best players considered the starting XI.

    Returns
    -------
    float
        Squad score; NaN if ``team_scores`` is empty.
    """
    if team_scores.empty:
        return float("nan")

    sorted_scores = team_scores.sort_values(ascending=False).reset_index(drop=True)
    top = sorted_scores.iloc[:n_top]
    depth = sorted_scores.iloc[n_top: n_top + 12]

    top_mean = float(top.mean()) if len(top) > 0 else 0.0
    depth_mean = float(depth.mean()) if len(depth) > 0 else 0.0

    return top_mean + DEPTH_WEIGHT * depth_mean


def compute_star_dependency(team_scores: pd.Series) -> float:
    """Compute star dependency as top-player fraction of total squad score.

    ``star_dependency = max(star_score) / sum(star_score)``

    A value > ``STAR_DEPENDENCY_THRESHOLD`` (0.35) flags excessive reliance
    on one player.

    Parameters
    ----------
    team_scores : pd.Series
        star_score values for one national team (must be non-negative).

    Returns
    -------
    float
        Dependency ratio in [0, 1]; NaN if total is 0 or series is empty.
    """
    if team_scores.empty:
        return float("nan")
    total = float(team_scores.sum())
    if total == 0:
        return float("nan")
    return float(team_scores.max()) / total


# ---------------------------------------------------------------------------
# FBref player stats loader
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column DataFrame to single-level snake_case names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c).strip() for c in col if c).lower()
                      for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df


def load_fbref_player_stats(
    leagues: list[str],
    seasons: list[int],
    fbref_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and cache FBref player season stats (standard + shooting + passing).

    Fetches ``standard``, ``shooting``, and ``passing`` stat tables, merges
    them on player/team/season, and returns a flat DataFrame with the columns
    documented in this module's input schema.

    Cached as a pickle at ``data/raw/fbref/player_stats_<hash>.pkl``.

    .. note::
        The returned DataFrame does **not** contain a ``national_team`` column.
        You must join it with a squad roster mapping
        ``(player, club) → national_team`` before passing to
        ``build_player_impact``.

    .. todo::
        Provide a squad-roster loader using StatsBomb open WC lineup data or
        an external squad list CSV (``data/external/squad_rosters.csv``).

    Parameters
    ----------
    leagues : list[str]
        soccerdata FBref league identifiers.
    seasons : list[int]
        Starting years of seasons.
    fbref_data_dir : Path, optional
        Override the soccerdata cache directory.

    Returns
    -------
    pd.DataFrame
        Flat player-season DataFrame.  Columns include: ``player``, ``club``,
        ``league``, ``season``, ``minutes``, ``starts``, ``matches``,
        ``goals``, ``assists``, ``xg``, ``xa``.
    """
    import hashlib, pickle as pkl

    raw_fbref_dir = RAW_DIR / "fbref"
    raw_fbref_dir.mkdir(parents=True, exist_ok=True)

    key = "player|" + ",".join(sorted(leagues)) + "|" + ",".join(str(s) for s in sorted(seasons))
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_path = raw_fbref_dir / f"player_stats_{digest}.pkl"

    if cache_path.exists():
        logger.info("load_fbref_player_stats: loading from cache %s", cache_path)
        with open(cache_path, "rb") as fh:
            return pkl.load(fh)

    sd_dir = fbref_data_dir or (RAW_DIR.parent.parent / "soccerdata" / "data" / "FBref")
    reader = sd.FBref(leagues=leagues, seasons=seasons)

    frames: dict[str, pd.DataFrame] = {}
    for stat in ("standard", "shooting", "passing"):
        logger.info("load_fbref_player_stats: fetching %s stats", stat)
        raw = reader.read_player_season_stats(stat_type=stat)
        frames[stat] = _flatten_columns(raw.reset_index())

    # Identify merge keys (they end up in the index before reset_index).
    # soccerdata typically indexes by (league, season, team, player).
    index_cols = ["league", "season", "team", "player"]
    available_keys = [c for c in index_cols if c in frames["standard"].columns]

    merged = frames["standard"]
    for stat in ("shooting", "passing"):
        extra_cols = [c for c in frames[stat].columns
                      if c not in merged.columns or c in available_keys]
        merged = merged.merge(
            frames[stat][extra_cols],
            on=available_keys,
            how="left",
        )

    # Normalise column names to the module schema.
    rename: dict[str, str] = {}
    col_lower = {c: c for c in merged.columns}

    # Minutes: look for 'min' or 'minutes' variants.
    for candidate in ("performance_min", "min", "minutes", "playing_time_min"):
        if candidate in col_lower:
            rename[candidate] = "minutes"
            break

    # Starts
    for candidate in ("performance_starts", "starts", "playing_time_starts"):
        if candidate in col_lower:
            rename[candidate] = "starts"
            break

    # Matches played
    for candidate in ("performance_mp", "mp", "matches", "playing_time_mp"):
        if candidate in col_lower:
            rename[candidate] = "matches"
            break

    # Goals
    for candidate in ("performance_gls", "gls", "goals", "standard_gls"):
        if candidate in col_lower:
            rename[candidate] = "goals"
            break

    # Assists
    for candidate in ("performance_ast", "ast", "assists", "standard_ast"):
        if candidate in col_lower:
            rename[candidate] = "assists"
            break

    # xG
    for candidate in ("expected_xg", "xg", "shooting_xg", "standard_xg", "npxg"):
        if candidate in col_lower:
            rename[candidate] = "xg"
            break

    # xA
    for candidate in ("expected_xa", "xa", "passing_xa", "expected_xag", "xag"):
        if candidate in col_lower:
            rename[candidate] = "xa"
            break

    # Squad → club
    for candidate in ("team", "squad"):
        if candidate in col_lower and candidate != "player":
            rename[candidate] = "club"
            break

    merged = merged.rename(columns=rename)

    # Log any schema columns that are still missing.
    for expected in ("minutes", "starts", "matches", "goals", "assists", "xg", "xa"):
        if expected not in merged.columns:
            logger.warning(
                "load_fbref_player_stats: could not map column '%s' — will be NaN", expected
            )
            merged[expected] = np.nan

    with open(cache_path, "wb") as fh:
        pkl.dump(merged, fh)
    logger.info(
        "load_fbref_player_stats: cached %d player-season rows to %s",
        len(merged),
        cache_path,
    )
    return merged


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_player_impact(
    players_df: pd.DataFrame,
    ucl_uel_clubs: Optional[set[str]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build a one-row-per-national-team player impact DataFrame.

    Pipeline
    --------
    1. Validate required columns (warn on missing, never crash).
    2. ``normalise_per90`` → adds ``goals90``, ``assists90``, ``xg90``, ``xa90``.
    3. ``flag_star_players`` → adds ``is_star``.
    4. ``compute_star_scores`` → adds ``star_score`` (with UCL/UEL multiplier).
    5. For each ``national_team``:
       - ``squad_score``        via ``compute_squad_score``
       - ``top_star_score``     = max(star_score) in squad
       - ``star_dependency``    via ``compute_star_dependency``
       - ``high_dependency``    = star_dependency > STAR_DEPENDENCY_THRESHOLD
       - ``player_impact``      = 0.6 × squad_score + 0.4 × top_star_score
    6. Save CSV to ``data/processed/player_impact.csv``.

    Parameters
    ----------
    players_df : pd.DataFrame
        One row per player.  Must include ``national_team``.  See module
        docstring for full column schema.
    ucl_uel_clubs : set[str], optional
        Club names competing in UCL/UEL.  Passed to ``flag_star_players``.
    output_path : Path, optional
        Override default output path.

    Returns
    -------
    pd.DataFrame
        Columns: ``national_team, squad_score, top_star_score,
        star_dependency, high_dependency, player_impact, n_stars, n_players``.
    """
    if "national_team" not in players_df.columns:
        logger.warning(
            "build_player_impact: 'national_team' column missing — cannot aggregate by team"
        )
        return pd.DataFrame()

    # Pipeline
    df = normalise_per90(players_df)
    df = flag_star_players(df, ucl_uel_clubs=ucl_uel_clubs)
    df = compute_star_scores(df)

    rows = []
    for team, grp in df.groupby("national_team"):
        scores = grp["star_score"]
        sq = compute_squad_score(scores)
        top = float(scores.max()) if not scores.empty else float("nan")
        dep = compute_star_dependency(scores)
        impact = 0.6 * sq + 0.4 * top if not (np.isnan(sq) or np.isnan(top)) else float("nan")

        rows.append(
            {
                "national_team": team,
                "squad_score": sq,
                "top_star_score": top,
                "star_dependency": dep,
                "high_dependency": dep > STAR_DEPENDENCY_THRESHOLD if not np.isnan(dep) else False,
                "player_impact": impact,
                "n_stars": int(grp["is_star"].sum()),
                "n_players": len(grp),
            }
        )

    result = pd.DataFrame(rows)

    if output_path is None:
        output_path = PROCESSED_DIR / "player_impact.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("build_player_impact: saved %d team rows to %s", len(result), output_path)

    return result
