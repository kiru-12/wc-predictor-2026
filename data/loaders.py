"""Data ingestion layer for the WC 2026 Predictor.

Sources
-------
- International results : martj42/international_results GitHub repo (CC0, public CSV,
                          downloaded via requests — no scraping, no ToS issues).
- Club ELO ratings      : soccerdata.ClubElo → http://api.clubelo.com (free, no key).
- Match xG / xGA        : soccerdata.FBref → fbref.com (free; soccerdata enforces
                          polite rate-limiting automatically).

No API keys are required for any of these sources.  See individual docstrings for
any manual-download steps.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import soccerdata as sd

logger = logging.getLogger(__name__)

# Project-relative raw-data directory (data/raw/ alongside this file).
RAW_DIR: Path = Path(__file__).resolve().parent / "raw"

# Public CSV maintained by Mart Jürisoo — updated regularly, CC0 licence.
_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


# ---------------------------------------------------------------------------
# International results
# ---------------------------------------------------------------------------


def load_international_results(
    start_year: int = 1990,
    end_year: int = 2022,
) -> pd.DataFrame:
    """Load historical international football results from a public GitHub CSV.

    Source
    ------
    https://github.com/martj42/international_results — CC0 licence, no API key.
    The full file (~50 k rows from 1872 onwards) is downloaded once and cached.

    Parameters
    ----------
    start_year : int
        First calendar year to include.
    end_year : int
        Last calendar year to include (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: date, home_team, away_team, home_score, away_score,
                 tournament, city, country, neutral.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / "results.csv"

    if cache_path.exists():
        logger.info("Loading international results from cache: %s", cache_path)
        df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        logger.info("Downloading international results from %s", _RESULTS_URL)
        df = pd.read_csv(_RESULTS_URL, parse_dates=["date"])
        df.to_csv(cache_path, index=False)
        logger.info("Cached %d rows to %s", len(df), cache_path)

    mask = (df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)
    result = df.loc[mask].reset_index(drop=True)
    logger.info(
        "International results %d–%d: %d rows returned", start_year, end_year, len(result)
    )
    return result


# ---------------------------------------------------------------------------
# Club ELO ratings
# ---------------------------------------------------------------------------


def load_elo_ratings(date: Optional[str] = None) -> pd.DataFrame:
    """Load club-level ELO ratings for a given date via soccerdata ClubElo.

    Source
    ------
    http://api.clubelo.com — free, no API key required.

    .. note::
        These are **club** ELO ratings, not national-team ratings.  They are
        used as a proxy for the quality of players assembled from their clubs.

    .. todo::
        Add a national-team ELO loader.  The best free source is
        https://www.eloratings.net but it has no public CSV endpoint.
        Options:
          (a) Manual download → place at ``data/raw/national_elo.csv``
          (b) Compute iteratively from international results using the
              standard Elo update formula (to be added in ``features/``).

    Parameters
    ----------
    date : str, optional
        ISO date string ``'YYYY-MM-DD'``.  Defaults to today if *None*.

    Returns
    -------
    pd.DataFrame
        Columns vary by soccerdata version; always includes ``Club`` and ``Elo``.
    """
    elo_dir = RAW_DIR / "clubelo"
    elo_dir.mkdir(parents=True, exist_ok=True)

    slug = date if date else "latest"
    cache_path = elo_dir / f"clubelo_{slug}.csv"

    if cache_path.exists():
        logger.info("Loading club ELO from cache: %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("Fetching club ELO for date=%s via soccerdata ClubElo", slug)
    reader = sd.ClubElo(data_dir=elo_dir)
    df = reader.read_by_date(date)
    df.to_csv(cache_path, index=False)
    logger.info("Cached %d club ELO rows to %s", len(df), cache_path)
    return df


# ---------------------------------------------------------------------------
# FBref xG / xGA (match-level shooting stats)
# ---------------------------------------------------------------------------


def _fbref_cache_path(leagues: list[str], seasons: list[int]) -> Path:
    """Return a deterministic cache path for a given leagues+seasons combination."""
    key = ",".join(sorted(leagues)) + "|" + ",".join(str(s) for s in sorted(seasons))
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return RAW_DIR / "fbref" / f"shooting_{digest}.pkl"


def load_fbref_shooting(
    leagues: list[str],
    seasons: list[int],
) -> pd.DataFrame:
    """Load per-match team shooting stats (includes xG, xGA) from FBref.

    Source
    ------
    https://fbref.com via ``soccerdata.FBref`` — free, no API key.
    soccerdata enforces polite request throttling automatically.

    Parameters
    ----------
    leagues : list[str]
        soccerdata league identifiers, e.g.
        ``['ENG-Premier League', 'ESP-La Liga']``.
        Use ``soccerdata.FBref.available_leagues()`` for the full list.
    seasons : list[int]
        Starting years of seasons, e.g. ``[2018, 2019, 2021]``.

    Returns
    -------
    pd.DataFrame
        Indexed by (league, season, team, game).  Columns include xG and xGA
        where FBref makes them available; missing columns are logged as warnings
        and will appear as NaN in downstream frames.
    """
    fbref_dir = RAW_DIR / "fbref"
    fbref_dir.mkdir(parents=True, exist_ok=True)

    cache_path = _fbref_cache_path(leagues, seasons)

    if cache_path.exists():
        logger.info("Loading FBref shooting stats from cache: %s", cache_path)
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    logger.info(
        "Fetching FBref shooting stats — leagues=%s seasons=%s", leagues, seasons
    )
    reader = sd.FBref(leagues=leagues, seasons=seasons, data_dir=fbref_dir)
    df: pd.DataFrame = reader.read_team_match_stats(stat_type="shooting")

    # Warn about missing xG columns so downstream code gets NaN rather than KeyError.
    col_labels = (
        list(df.columns.get_level_values(-1))
        if isinstance(df.columns, pd.MultiIndex)
        else list(df.columns)
    )
    col_lower = [str(c).lower() for c in col_labels]
    for expected in ("xg", "xga"):
        if not any(expected in c for c in col_lower):
            logger.warning(
                "Column '%s' not present in FBref shooting data — downstream will see NaN",
                expected,
            )

    with open(cache_path, "wb") as fh:
        pickle.dump(df, fh)
    logger.info("Cached %d rows to %s", len(df), cache_path)
    return df


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def load_all() -> dict[str, pd.DataFrame]:
    """Load every dataset and log row counts.

    Uses default parameters suitable for backtesting on 2014/2018/2022
    World Cups.  Call individual loaders directly for custom ranges.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``'international_results'``, ``'club_elo'``, ``'fbref_shooting'``.
    """
    data: dict[str, pd.DataFrame] = {}

    data["international_results"] = load_international_results(
        start_year=1990, end_year=2022
    )
    logger.info("international_results: %d rows", len(data["international_results"]))

    data["club_elo"] = load_elo_ratings()
    logger.info("club_elo: %d rows", len(data["club_elo"]))

    # Top-5 European leagues over the three WC cycles used for backtesting.
    data["fbref_shooting"] = load_fbref_shooting(
        leagues=[
            "ENG-Premier League",
            "ESP-La Liga",
            "GER-Bundesliga",
            "ITA-Serie A",
            "FRA-Ligue 1",
        ],
        seasons=list(range(2017, 2023)),
    )
    logger.info("fbref_shooting: %d rows", len(data["fbref_shooting"]))

    return data
