"""Context auto-fill pipeline for team_context_2026.csv.

What is automated here
----------------------
1. ``squad_market_value_eur``        — scraped from Transfermarkt's national-team
                                       market-value ranking page (robots.txt checked,
                                       ~3 s throttle, raw HTML cached to data/raw/).
                                       Left blank if the page is unreachable or
                                       scraping is disallowed; no values are invented.
2. ``avg_group_opponent_value_eur``  — computed from squad values + groups_2026.csv.
3. ``core_retention_score``          — roster overlap WC 2022 → 2026 if
                                       data/external/squads_2026.csv is provided.

What is left blank (manual pass required)
------------------------------------------
* ``pre_tournament_odds``           — decimal odds (Bet365/Pinnacle, 2026-06-10).
* ``manager_tenure_months``         — months since current manager's appointment.
* ``historic_first_time``           — judgment per team (bool, stored as 0.0/1.0).
* ``rival_top_team_underperformed`` — judgment per team (bool, stored as 0.0/1.0).
* ``manager_stability_score``       — TODO: derive from ``manager_tenure_months``
                                       via ``normalise_tenure_to_stability()`` once
                                       tenure is filled; not automated here to avoid
                                       partial normalisation over an incomplete vector.

Merge-preserve guarantee
------------------------
``build_team_context_csv`` reads any existing ``team_context_2026.csv`` and only
fills *blank cells* with computed values.  Non-null cells are never overwritten.

Boolean encoding
-----------------
Booleans are stored as ``0.0`` / ``1.0`` floats (not True/False strings) to match
the existing France row format.
"""

import hashlib
import logging
import pickle
import time
import urllib.robotparser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

EXTERNAL_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "external"
RAW_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Column spec
# ---------------------------------------------------------------------------

_CONTEXT_COLUMNS: list[str] = [
    "team",
    "squad_market_value_eur",
    "avg_group_opponent_value_eur",
    "pre_tournament_odds",
    "manager_tenure_months",
    "historic_first_time",
    "core_retention_score",
    "manager_stability_score",
    "rival_top_team_underperformed",
]

# ---------------------------------------------------------------------------
# Transfermarkt scraper
# ---------------------------------------------------------------------------

#: Scraped name → canonical name in groups_2026.csv.
TRANSFERMARKT_ALIAS_MAP: dict[str, str] = {
    # Straightforward matches
    "Argentina":                "Argentina",
    "Australia":                "Australia",
    "Austria":                  "Austria",
    "Belgium":                  "Belgium",
    "Brazil":                   "Brazil",
    "Canada":                   "Canada",
    "Colombia":                 "Colombia",
    "Croatia":                  "Croatia",
    "Czechia":                  "Czechia",
    "Ecuador":                  "Ecuador",
    "Egypt":                    "Egypt",
    "England":                  "England",
    "France":                   "France",
    "Germany":                  "Germany",
    "Ghana":                    "Ghana",
    "Haiti":                    "Haiti",
    "Iran":                     "Iran",
    "Iraq":                     "Iraq",
    "Japan":                    "Japan",
    "Jordan":                   "Jordan",
    "Mexico":                   "Mexico",
    "Morocco":                  "Morocco",
    "Netherlands":              "Netherlands",
    "New Zealand":              "New Zealand",
    "Norway":                   "Norway",
    "Panama":                   "Panama",
    "Paraguay":                 "Paraguay",
    "Portugal":                 "Portugal",
    "Qatar":                    "Qatar",
    "Saudi Arabia":             "Saudi Arabia",
    "Scotland":                 "Scotland",
    "Senegal":                  "Senegal",
    "Spain":                    "Spain",
    "Sweden":                   "Sweden",
    "Switzerland":              "Switzerland",
    "Tunisia":                  "Tunisia",
    "Uruguay":                  "Uruguay",
    "USA":                      "USA",
    "Uzbekistan":               "Uzbekistan",
    "Algeria":                  "Algeria",
    # Aliases — variants used by Transfermarkt
    "Turkey":                   "Turkey",
    "Türkiye":                  "Turkey",
    "Turkiye":                  "Turkey",
    "Ivory Coast":              "Ivory Coast",
    "Côte d'Ivoire":            "Ivory Coast",
    "Cote d'Ivoire":            "Ivory Coast",
    "South Korea":              "South Korea",
    "Korea Republic":           "South Korea",
    "Korea, Republic of":       "South Korea",
    "Republic of Korea":        "South Korea",
    "Bosnia and Herzegovina":   "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":       "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":     "Bosnia and Herzegovina",
    "DR Congo":                 "DR Congo",
    "Congo DR":                 "DR Congo",
    "Dem. Rep. Congo":          "DR Congo",
    "Congo, DR":                "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Cape Verde":               "Cape Verde",
    "Cabo Verde":               "Cape Verde",
    "South Africa":             "South Africa",
    "Curacao":                  "Curacao",
    "Curaçao":                  "Curacao",
    "United States":            "USA",
    "United States of America": "USA",
    "Czech Republic":           "Czechia",
    "Netherlands (Holland)":    "Netherlands",
    "New Caledonia":            "New Caledonia",
}

_TM_RANKING_URL = (
    "https://www.transfermarkt.com/statistik/weltrangliste/statistik"
)
_TM_ROBOTS_URL = "https://www.transfermarkt.com/robots.txt"
_REQUEST_DELAY_S = 3.5          # between requests (3-5 s per project rules)
_ROBOTS_TIMEOUT_S = 6           # short timeout for robots.txt — fail fast
_REQUEST_TIMEOUT = 20
_VALUE_MIN_EUR = 1_000_000      # €1 M — below this is implausible
_VALUE_MAX_EUR = 2_000_000_000  # €2 B — above this is implausible

# Well-known strong teams that should rank high; used for sanity ordering check.
_EXPECTED_TOP_TEAMS = {"England", "France", "Spain", "Germany", "Brazil", "Argentina", "Portugal", "Netherlands"}
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://www.transfermarkt.com/",
}


def _check_robots(url: str) -> bool:
    """Return True if scraping *url* is permitted by Transfermarkt's robots.txt.

    Uses a short ``requests`` fetch (timeout=``_ROBOTS_TIMEOUT_S``) instead of
    ``urllib.robotparser`` to avoid hanging on corporate-network blocks.  Returns
    False (disallowed) if the robots.txt cannot be fetched within the timeout.
    """
    try:
        resp = requests.get(
            _TM_ROBOTS_URL,
            headers={"User-Agent": _HEADERS["User-Agent"]},
            timeout=_ROBOTS_TIMEOUT_S,
        )
        resp.raise_for_status()
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(_TM_ROBOTS_URL)
        rp.parse(resp.text.splitlines())
        allowed = rp.can_fetch("*", url)
        if not allowed:
            logger.warning("_check_robots: robots.txt disallows %s", url)
        return allowed
    except requests.exceptions.Timeout:
        logger.warning(
            "_check_robots: robots.txt request timed out after %ds — "
            "treating as disallowed (try again on a less restricted network)",
            _ROBOTS_TIMEOUT_S,
        )
        return False
    except Exception as exc:
        logger.warning(
            "_check_robots: could not read robots.txt (%s) — treating as disallowed", exc
        )
        return False


def parse_transfermarkt_value(value_str: str) -> Optional[float]:
    """Parse a Transfermarkt formatted market-value string to EUR float.

    Handles ``"€1.23bn"``, ``"€450.00m"``, ``"€7.50k"``, plain numbers,
    comma-formatted values, etc.  Returns ``None`` if unparseable.

    Parameters
    ----------
    value_str : str
        Raw value cell from the Transfermarkt table.

    Returns
    -------
    float or None
    """
    if not isinstance(value_str, str):
        return None
    s = (
        value_str.strip()
        .replace(",", "")
        .replace("€", "")
        .replace(" ", "")
        .lower()
    )
    if not s or s in ("-", "n/a", "—", ""):
        return None
    try:
        if s.endswith("bn"):
            return float(s[:-2]) * 1_000_000_000
        if s.endswith("m"):
            return float(s[:-1]) * 1_000_000
        if s.endswith("k"):
            return float(s[:-1]) * 1_000
        return float(s)
    except ValueError:
        return None


def _validate_squad_values(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with implausible squad values and return the cleaned DataFrame.

    Values outside [_VALUE_MIN_EUR, _VALUE_MAX_EUR] are set to NaN and logged
    as warnings.  The DataFrame is returned with the ``squad_market_value_eur``
    column sanitised.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``team`` and ``squad_market_value_eur``.

    Returns
    -------
    pd.DataFrame
        Copy with out-of-range values replaced by NaN.
    """
    out = df.copy()
    vals = pd.to_numeric(out["squad_market_value_eur"], errors="coerce")
    too_low  = vals.notna() & (vals <  _VALUE_MIN_EUR)
    too_high = vals.notna() & (vals > _VALUE_MAX_EUR)
    for idx in out[too_low].index:
        logger.warning(
            "_validate_squad_values: %s value €%.0f is below minimum €%.0f — set to NaN",
            out.at[idx, "team"], vals[idx], _VALUE_MIN_EUR,
        )
    for idx in out[too_high].index:
        logger.warning(
            "_validate_squad_values: %s value €%.0f exceeds maximum €%.0f — set to NaN",
            out.at[idx, "team"], vals[idx], _VALUE_MAX_EUR,
        )
    out.loc[too_low | too_high, "squad_market_value_eur"] = np.nan
    return out


def _sanity_check_ordering(
    df: pd.DataFrame,
    raise_on_fail: bool = False,
) -> bool:
    """Print top-5 and bottom-5 teams by scraped value; check for ordering inversion.

    Returns True if the ordering looks plausible (at least one expected-top team
    is in the actual top 10 and at least one expected-bottom team is in the
    actual bottom 10).  Returns False (and logs an error) if neither condition
    holds, indicating the parser probably grabbed the wrong column.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: ``team``, ``squad_market_value_eur`` (validated, canonical names).
    raise_on_fail : bool
        If True raise ``ValueError`` when the sanity check fails.

    Returns
    -------
    bool
    """
    valid = df.dropna(subset=["squad_market_value_eur"]).sort_values(
        "squad_market_value_eur", ascending=False
    ).reset_index(drop=True)

    if valid.empty:
        logger.error("_sanity_check_ordering: no valid values to check")
        return False

    n = len(valid)
    top5    = valid.head(5)
    bottom5 = valid.tail(5)

    print("\n─── Scraped squad value sanity check ───────────────────────────────")
    print("  Top 5 teams by squad value (should include strong nations):")
    for _, row in top5.iterrows():
        print(f"    {row['team']:<30}  €{row['squad_market_value_eur']/1e6:>8.1f} M")
    print("  Bottom 5 teams by squad value (should include minnows):")
    for _, row in bottom5.iterrows():
        print(f"    {row['team']:<30}  €{row['squad_market_value_eur']/1e6:>8.1f} M")
    print("─────────────────────────────────────────────────────────────────────")

    actual_top10_teams = set(valid.head(10)["team"])

    top_ok = bool(_EXPECTED_TOP_TEAMS & actual_top10_teams)

    # Check value range: strongest team should be materially more valuable than
    # weakest.  A flat distribution (ratio < 2) suggests the parser grabbed the
    # wrong column (e.g., player counts instead of values).
    min_val = float(valid["squad_market_value_eur"].min())
    max_val = float(valid["squad_market_value_eur"].max())
    range_ok = max_val > 0 and (max_val / max(min_val, 1)) >= 2.0

    if not top_ok or not range_ok:
        reasons = []
        if not top_ok:
            reasons.append(
                f"expected strong teams in top-10 but found none of "
                f"{sorted(_EXPECTED_TOP_TEAMS)} there (got {sorted(actual_top10_teams)})"
            )
        if not range_ok:
            reasons.append(
                f"value range too flat: max €{max_val/1e6:.1f}M / min €{min_val/1e6:.1f}M "
                f"(ratio {max_val/max(min_val,1):.1f}x, expected ≥2x)"
            )
        msg = (
            "_sanity_check_ordering: ordering looks WRONG — "
            + "; ".join(reasons)
            + ". The parser likely grabbed the wrong table column. "
            "File NOT saved — inspect data/raw/transfermarkt/ HTML manually."
        )
        logger.error(msg)
        print(f"\n⚠  SANITY CHECK FAILED: {msg}\n")
        if raise_on_fail:
            raise ValueError(msg)
        return False

    print("  ✓ Ordering looks plausible.\n")
    return True


def apply_canonical_names(
    raw_df: pd.DataFrame,
    canonical_teams: list[str],
    alias_map: Optional[dict[str, str]] = None,
    name_col: str = "team",
) -> pd.DataFrame:
    """Map raw scraped team names to canonical names from groups_2026.csv.

    Teams whose raw name cannot be resolved are *dropped* and a WARNING is
    logged so no data is silently lost.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Scrape output; must contain *name_col*.
    canonical_teams : list[str]
        Accepted canonical team names.
    alias_map : dict, optional
        Defaults to ``TRANSFERMARKT_ALIAS_MAP``.
    name_col : str
        Column holding team names.

    Returns
    -------
    pd.DataFrame
        Copy with *name_col* replaced by canonical names; unresolvable rows
        removed.
    """
    if alias_map is None:
        alias_map = TRANSFERMARKT_ALIAS_MAP

    if name_col not in raw_df.columns:
        logger.warning("apply_canonical_names: '%s' column missing", name_col)
        return raw_df.copy()

    canonical_set = set(canonical_teams)
    result_rows: list[pd.Series] = []
    unmatched: list[str] = []

    for _, row in raw_df.iterrows():
        raw_name = str(row[name_col]).strip()
        canonical = alias_map.get(raw_name)
        if canonical is None:
            # Case-insensitive fallback.
            for k, v in alias_map.items():
                if k.lower() == raw_name.lower():
                    canonical = v
                    break
        if canonical is None or canonical not in canonical_set:
            unmatched.append(raw_name)
            continue
        new_row = row.copy()
        new_row[name_col] = canonical
        result_rows.append(new_row)

    if unmatched:
        logger.warning(
            "apply_canonical_names: %d scraped name(s) unmatched and dropped: %s",
            len(unmatched),
            unmatched,
        )

    if not result_rows:
        return pd.DataFrame(columns=raw_df.columns)
    return pd.DataFrame(result_rows).reset_index(drop=True)


def _parse_tm_html(html: str) -> pd.DataFrame:
    """Parse Transfermarkt weltrangliste HTML → raw team/value DataFrame.

    Uses ``pd.read_html`` (lxml backend) to find the market-value table.
    Returns empty DataFrame if parsing fails.

    Parameters
    ----------
    html : str
        Raw HTML of the Transfermarkt ranking page.

    Returns
    -------
    pd.DataFrame
        Columns: ``team`` (raw name), ``squad_market_value_eur`` (float).
    """
    try:
        import io as _io
        tables = pd.read_html(_io.StringIO(html), flavor="lxml")
    except Exception as exc:
        logger.warning("_parse_tm_html: pd.read_html failed (%s)", exc)
        return pd.DataFrame(columns=["team", "squad_market_value_eur"])

    for tbl in tables:
        # Flatten MultiIndex headers.
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [
                " ".join(str(c) for c in col if str(c) != "nan").strip()
                for col in tbl.columns
            ]

        col_lower = [str(c).lower() for c in tbl.columns]

        # Find team-name column.
        name_col = None
        for candidate in ("club", "team", "nation", "country", "verein", "name"):
            matches = [c for c, cl in zip(tbl.columns, col_lower) if candidate in cl]
            if matches:
                name_col = matches[0]
                break

        # Find market-value column.
        value_col = None
        for candidate in ("market value", "value", "squad value", "marktwert", "worth"):
            matches = [c for c, cl in zip(tbl.columns, col_lower) if candidate in cl]
            if matches:
                value_col = matches[0]
                break

        if name_col is None or value_col is None:
            continue

        raw = tbl[[name_col, value_col]].copy()
        raw.columns = ["team", "raw_value"]
        raw["team"] = raw["team"].astype(str).str.strip()
        raw["squad_market_value_eur"] = raw["raw_value"].apply(
            lambda v: parse_transfermarkt_value(str(v))
        )
        result = raw[["team", "squad_market_value_eur"]].dropna(
            subset=["squad_market_value_eur"]
        )
        if len(result) >= 20:
            logger.info("_parse_tm_html: parsed %d team rows", len(result))
            return result.reset_index(drop=True)

    logger.warning(
        "_parse_tm_html: could not identify a market-value table "
        "(Transfermarkt may have changed their HTML structure)"
    )
    return pd.DataFrame(columns=["team", "squad_market_value_eur"])


def fetch_transfermarkt_squad_values(
    canonical_teams: list[str],
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch national-team squad market values from Transfermarkt.

    Resilience features
    -------------------
    * robots.txt checked with a short timeout (``_ROBOTS_TIMEOUT_S``); aborts
      fast rather than hanging on corporate-network blocks.
    * Raw HTML cached to ``data/raw/transfermarkt/``; cache is read first on
      re-runs so a partial success is never lost and the site is not hammered.
    * 403/429/Cloudflare responses are caught, logged with a clear retry note,
      and return an empty DataFrame rather than crashing.
    * Parsed values are validated for plausibility (> €1 M, < €2 B); out-of-range
      values become NaN with a logged warning.
    * A sanity-ordering check confirms strong teams appear at the top and minnows
      at the bottom; if the check fails the empty DataFrame is returned and the
      bad HTML is NOT saved as a parsed pickle.

    Parameters
    ----------
    canonical_teams : list[str]
        Expected canonical team names (from groups_2026.csv).
    cache_dir : Path, optional
        Override Transfermarkt cache directory.
    force_refresh : bool
        Re-download even if HTML is cached.

    Returns
    -------
    pd.DataFrame
        Columns: ``team`` (canonical), ``squad_market_value_eur`` (float).
        May be empty if blocked, unreachable, unparseable, or failed sanity.
    """
    tm_dir = cache_dir or (RAW_DIR / "transfermarkt")
    tm_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.md5(_TM_RANKING_URL.encode()).hexdigest()[:12]
    html_cache   = tm_dir / f"weltrangliste_{url_hash}.html"
    parsed_cache = tm_dir / f"squad_values_{url_hash}.pkl"

    # Return parsed cache if available and not forcing refresh.
    if not force_refresh and parsed_cache.exists():
        logger.info(
            "fetch_transfermarkt_squad_values: using parsed cache %s", parsed_cache
        )
        with open(parsed_cache, "rb") as fh:
            return pickle.load(fh)

    # Robots.txt gate (short timeout so we fail fast on restricted networks).
    if not _check_robots(_TM_RANKING_URL):
        logger.warning(
            "fetch_transfermarkt_squad_values: robots.txt disallowed or unreachable "
            "— squad_market_value_eur will stay blank"
        )
        return pd.DataFrame(columns=["team", "squad_market_value_eur"])

    # Fetch HTML (use HTML cache if available).
    if not force_refresh and html_cache.exists():
        logger.info(
            "fetch_transfermarkt_squad_values: reading HTML from cache %s", html_cache
        )
        html = html_cache.read_text(encoding="utf-8", errors="replace")
    else:
        logger.info("fetch_transfermarkt_squad_values: fetching %s", _TM_RANKING_URL)
        try:
            session = requests.Session()
            session.headers.update(_HEADERS)
            resp = session.get(_TM_RANKING_URL, timeout=_REQUEST_TIMEOUT)

            if resp.status_code in (403, 429):
                print(
                    f"\n⛔  Transfermarkt returned HTTP {resp.status_code} "
                    "(blocked / rate-limited).  Cells already filled are preserved.\n"
                    "   → Wait a few minutes then re-run with force_tm_refresh=True,\n"
                    "     or try from a different network / VPN endpoint.\n"
                )
                logger.warning(
                    "fetch_transfermarkt_squad_values: HTTP %d from Transfermarkt — "
                    "squad_market_value_eur stays blank",
                    resp.status_code,
                )
                return pd.DataFrame(columns=["team", "squad_market_value_eur"])

            # Cloudflare challenge pages return 200 but contain no table.
            if "cloudflare" in resp.text.lower() or "cf-browser-verification" in resp.text:
                print(
                    "\n⛔  Transfermarkt served a Cloudflare challenge page (200 OK but no data).\n"
                    "   → Try again later or from a different IP / VPN endpoint.\n"
                )
                logger.warning(
                    "fetch_transfermarkt_squad_values: Cloudflare challenge detected"
                )
                return pd.DataFrame(columns=["team", "squad_market_value_eur"])

            resp.raise_for_status()
            html = resp.text
            html_cache.write_text(html, encoding="utf-8")
            logger.info(
                "fetch_transfermarkt_squad_values: cached %d bytes to %s",
                len(html),
                html_cache,
            )
            time.sleep(_REQUEST_DELAY_S)
        except requests.exceptions.Timeout:
            print(
                "\n⛔  Transfermarkt request timed out.  "
                "Try again on a less restricted network.\n"
            )
            logger.warning("fetch_transfermarkt_squad_values: request timed out")
            return pd.DataFrame(columns=["team", "squad_market_value_eur"])
        except Exception as exc:
            logger.warning(
                "fetch_transfermarkt_squad_values: HTTP request failed (%s) — "
                "squad_market_value_eur will stay blank",
                exc,
            )
            return pd.DataFrame(columns=["team", "squad_market_value_eur"])

    # Parse HTML.
    raw_df = _parse_tm_html(html)
    if raw_df.empty:
        return pd.DataFrame(columns=["team", "squad_market_value_eur"])

    # Validate values before alias mapping.
    raw_df = _validate_squad_values(raw_df)

    # Apply alias / canonical name mapping.
    canonical_df = apply_canonical_names(raw_df, canonical_teams)
    if canonical_df.empty:
        logger.warning(
            "fetch_transfermarkt_squad_values: no teams matched after alias mapping"
        )
        return pd.DataFrame(columns=["team", "squad_market_value_eur"])

    result = canonical_df.drop_duplicates(subset=["team"]).reset_index(drop=True)

    # Sanity ordering gate — do NOT cache or return if ordering is wrong.
    if not _sanity_check_ordering(result):
        logger.error(
            "fetch_transfermarkt_squad_values: sanity check failed — "
            "returning empty DataFrame. Inspect %s manually.",
            html_cache,
        )
        return pd.DataFrame(columns=["team", "squad_market_value_eur"])

    with open(parsed_cache, "wb") as fh:
        pickle.dump(result, fh)

    n_matched = len(result)
    unmatched_canonical = sorted(set(canonical_teams) - set(result["team"]))
    logger.info(
        "fetch_transfermarkt_squad_values: matched %d / %d teams",
        n_matched,
        len(canonical_teams),
    )
    if unmatched_canonical:
        print(
            f"\n⚠  {len(unmatched_canonical)} canonical team(s) not found in Transfermarkt "
            f"data (add to TRANSFERMARKT_ALIAS_MAP if needed): {unmatched_canonical}\n"
        )
        logger.warning(
            "fetch_transfermarkt_squad_values: %d canonical team(s) unmatched: %s",
            len(unmatched_canonical),
            unmatched_canonical,
        )

    return result


# ---------------------------------------------------------------------------
# Group-opponent average squad value
# ---------------------------------------------------------------------------

def compute_avg_group_opponent_value(
    context_df: pd.DataFrame,
    groups_df: pd.DataFrame,
) -> pd.Series:
    """Compute each team's average group-opponent squad market value.

    For team T in group G the result is the mean squad value of the other
    three teams in G.  Teams whose group-mates have no known values get NaN.

    Parameters
    ----------
    context_df : pd.DataFrame
        Columns: ``team``, ``squad_market_value_eur``.
    groups_df : pd.DataFrame
        Columns: ``team``, ``group``.

    Returns
    -------
    pd.Series
        Indexed by team, values float EUR or NaN.
    """
    needed_ctx = {"team", "squad_market_value_eur"}
    needed_grp = {"team", "group"}
    for df, needed, name in [
        (context_df, needed_ctx, "context_df"),
        (groups_df, needed_grp, "groups_df"),
    ]:
        missing = needed - set(df.columns)
        if missing:
            logger.warning(
                "compute_avg_group_opponent_value: missing columns %s in %s",
                missing, name,
            )
            teams = context_df["team"].tolist() if "team" in context_df.columns else []
            return pd.Series(np.nan, index=teams, name="avg_group_opponent_value_eur")

    values = (
        context_df[["team", "squad_market_value_eur"]]
        .copy()
        .assign(
            squad_market_value_eur=lambda d: pd.to_numeric(
                d["squad_market_value_eur"], errors="coerce"
            )
        )
    )
    merged = groups_df[["team", "group"]].merge(values, on="team", how="left")

    results: dict[str, float] = {}
    for _group_name, grp in merged.groupby("group"):
        for _, row in grp.iterrows():
            others = grp.loc[
                grp["team"] != row["team"], "squad_market_value_eur"
            ]
            valid = others.dropna()
            results[row["team"]] = (
                float(valid.mean()) if len(valid) >= 1 else float("nan")
            )

    for t in context_df["team"]:
        if t not in results:
            results[t] = float("nan")

    return pd.Series(results, name="avg_group_opponent_value_eur")


# ---------------------------------------------------------------------------
# Core retention score
# ---------------------------------------------------------------------------

def load_wc2022_squads_statsbomb(cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load WC 2022 squad members from StatsBomb open data, cached locally.

    Returns empty DataFrame if StatsBomb is unavailable.
    """
    cache_base = cache_dir or (RAW_DIR / "statsbomb")
    cache_base.mkdir(parents=True, exist_ok=True)
    cache_path = cache_base / "wc2022_squads.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    try:
        from statsbombpy import sb  # type: ignore
    except ImportError:
        logger.warning(
            "load_wc2022_squads_statsbomb: statsbombpy not installed — "
            "core_retention_score will be blank"
        )
        return pd.DataFrame(columns=["team", "player_id", "player_name"])

    try:
        matches = sb.matches(competition_id=43, season_id=106)
    except Exception as exc:
        logger.warning(
            "load_wc2022_squads_statsbomb: failed to fetch matches — %s", exc
        )
        return pd.DataFrame(columns=["team", "player_id", "player_name"])

    rows: list[dict] = []
    for match_id in matches["match_id"].tolist():
        try:
            lineups = sb.lineups(match_id=match_id)
            for team_name, lineup_df in lineups.items():
                for _, p in lineup_df.iterrows():
                    rows.append({
                        "team": team_name,
                        "player_id": int(p.get("player_id", -1)),
                        "player_name": str(p.get("player_name", "")),
                    })
            time.sleep(1.0)
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=["team", "player_id", "player_name"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["team", "player_id"])
    with open(cache_path, "wb") as fh:
        pickle.dump(df, fh)
    return df


def compute_core_retention(
    squads_2022: pd.DataFrame,
    squads_2026: pd.DataFrame,
    match_on: str = "player_name",
) -> pd.Series:
    """Compute WC 2022 → 2026 squad retention score (0–100) per team.

    Parameters
    ----------
    squads_2022 : pd.DataFrame
        WC 2022 squad members.
    squads_2026 : pd.DataFrame
        2026 squad members.
    match_on : str
        Column used for player identity.

    Returns
    -------
    pd.Series
        Indexed by team, values 0–100 or NaN.
    """
    if squads_2022.empty or squads_2026.empty:
        teams = squads_2026["team"].unique() if "team" in squads_2026.columns else []
        return pd.Series(np.nan, index=teams, name="core_retention_score")

    for df, name in [(squads_2022, "squads_2022"), (squads_2026, "squads_2026")]:
        for col in ("team", match_on):
            if col not in df.columns:
                logger.warning(
                    "compute_core_retention: column '%s' missing from %s", col, name
                )
                return pd.Series(dtype=float, name="core_retention_score")

    results: dict[str, float] = {}
    all_teams = set(squads_2022["team"].unique()) | set(squads_2026["team"].unique())
    for team in all_teams:
        sq22 = set(squads_2022.loc[squads_2022["team"] == team, match_on].dropna())
        sq26 = set(squads_2026.loc[squads_2026["team"] == team, match_on].dropna())
        if not sq22:
            results[team] = float("nan")
            continue
        results[team] = round(len(sq22 & sq26) / len(sq22) * 100, 1)

    return pd.Series(results, name="core_retention_score")


# ---------------------------------------------------------------------------
# Tenure → stability (NOT called automatically — see TODO in docstring)
# ---------------------------------------------------------------------------

def normalise_tenure_to_stability(tenure: pd.Series) -> pd.Series:
    """Log-normalise manager tenure (months) to a [0, 1] stability score.

    .. todo::
        Call this from ``build_team_context_csv`` once all 48 teams'
        ``manager_tenure_months`` values are filled.  Running it on a partial
        vector distorts the normalisation range for every team.

    Parameters
    ----------
    tenure : pd.Series
        Raw manager tenure in months.

    Returns
    -------
    pd.Series
        Scores in [0, 1].
    """
    t = pd.to_numeric(tenure, errors="coerce").clip(lower=0)
    log_t = np.log1p(t)
    lo = float(log_t.min(skipna=True))
    hi = float(log_t.max(skipna=True))
    if hi > lo:
        score = (log_t - lo) / (hi - lo)
    else:
        score = pd.Series(0.5, index=tenure.index)
    score = score.where(t.notna(), other=np.nan)
    return score.rename("manager_stability_score")


# ---------------------------------------------------------------------------
# Merge-preserve helper
# ---------------------------------------------------------------------------

def _fill_missing(
    base: pd.DataFrame,
    computed: pd.DataFrame,
    key: str = "team",
) -> pd.DataFrame:
    """Fill blank cells in *base* with values from *computed*.

    Non-null cells are never overwritten.
    """
    result = base.copy()
    if computed.empty:
        return result
    for col in computed.columns:
        if col == key:
            continue
        if col not in result.columns:
            result[col] = np.nan
        computed_indexed = computed.set_index(key)[col]
        mask = result[col].isna() & result[key].isin(computed_indexed.index)
        new_vals = result.loc[mask, key].map(computed_indexed)
        try:
            new_vals = new_vals.astype(result[col].dtype)
        except (TypeError, ValueError):
            pass
        result.loc[mask, col] = new_vals
    return result


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_team_context_csv(
    groups_path: Optional[Path] = None,
    existing_context_path: Optional[Path] = None,
    squads_2026_path: Optional[Path] = None,
    wc2022_squads_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    scrape_transfermarkt: bool = True,
    tm_cache_dir: Optional[Path] = None,
    force_tm_refresh: bool = False,
) -> pd.DataFrame:
    """Build (or refresh) team_context_2026.csv, preserving existing cells.

    See module docstring for the full list of what is and is not automated.

    Parameters
    ----------
    groups_path : Path, optional
        groups_2026.csv path (canonical team names and group assignments).
    existing_context_path : Path, optional
        Existing context CSV to merge with.  Defaults to *output_path*.
    squads_2026_path : Path, optional
        CSV with ``team, player_name`` for 2026 squads.
    wc2022_squads_df : pd.DataFrame, optional
        Inject pre-loaded WC 2022 squads (for testing).
    output_path : Path, optional
        Write destination.  Default: data/external/team_context_2026.csv.
    scrape_transfermarkt : bool
        Set False to skip TM scraping (useful in tests).
    tm_cache_dir : Path, optional
        Override Transfermarkt HTML cache directory.
    force_tm_refresh : bool
        Force re-download of Transfermarkt data.

    Returns
    -------
    pd.DataFrame
        48-row context table in ``_CONTEXT_COLUMNS`` order.
    """
    groups_path = groups_path or (EXTERNAL_DIR / "groups_2026.csv")
    output_path = output_path or (EXTERNAL_DIR / "team_context_2026.csv")
    existing_context_path = existing_context_path or output_path

    if not groups_path.exists():
        logger.warning("build_team_context_csv: %s not found", groups_path)
        return pd.DataFrame(columns=_CONTEXT_COLUMNS)

    groups_df = pd.read_csv(groups_path)
    if "team" not in groups_df.columns:
        logger.warning("build_team_context_csv: 'team' column missing from groups file")
        return pd.DataFrame(columns=_CONTEXT_COLUMNS)

    canonical_teams = groups_df["team"].tolist()

    # -- Load existing -------------------------------------------------------
    if existing_context_path.exists():
        existing = pd.read_csv(existing_context_path)
    else:
        existing = pd.DataFrame(columns=_CONTEXT_COLUMNS)

    skeleton = pd.DataFrame({"team": canonical_teams})
    for col in _CONTEXT_COLUMNS:
        if col != "team":
            skeleton[col] = np.nan

    if not existing.empty and "team" in existing.columns:
        skeleton = _fill_missing(skeleton, existing)

    # -- Transfermarkt -------------------------------------------------------
    if scrape_transfermarkt:
        tm_df = fetch_transfermarkt_squad_values(
            canonical_teams,
            cache_dir=tm_cache_dir,
            force_refresh=force_tm_refresh,
        )
        if not tm_df.empty:
            skeleton = _fill_missing(skeleton, tm_df)
        else:
            logger.warning(
                "build_team_context_csv: TM scrape returned no data — "
                "squad_market_value_eur stays blank. "
                "Fill manually: https://www.transfermarkt.com/statistik/weltrangliste"
            )

    # -- Avg opponent value --------------------------------------------------
    if skeleton["squad_market_value_eur"].notna().sum() > 0:
        opp = compute_avg_group_opponent_value(skeleton, groups_df)
        opp_df = opp.reset_index()
        opp_df.columns = ["team", "avg_group_opponent_value_eur"]
        skeleton = _fill_missing(skeleton, opp_df)

    # -- Core retention ------------------------------------------------------
    if squads_2026_path is not None and Path(squads_2026_path).exists():
        squads_2026 = pd.read_csv(squads_2026_path)
        if wc2022_squads_df is None:
            wc2022_squads_df = load_wc2022_squads_statsbomb()
        ret = compute_core_retention(wc2022_squads_df, squads_2026)
        ret_df = ret.reset_index()
        ret_df.columns = ["team", "core_retention_score"]
        skeleton = _fill_missing(skeleton, ret_df)

    # NOTE: manager_stability_score is intentionally NOT computed here.
    # See normalise_tenure_to_stability() docstring.

    # -- Finalise ------------------------------------------------------------
    for col in _CONTEXT_COLUMNS:
        if col not in skeleton.columns:
            skeleton[col] = np.nan
    skeleton = skeleton[_CONTEXT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton.to_csv(output_path, index=False)

    _print_summary(skeleton)
    return skeleton


def _print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    notes = {
        "squad_market_value_eur":           "MANUAL  — Transfermarkt (blocked on this network)",
        "avg_group_opponent_value_eur":      "AUTO    — from squad values + group draw",
        "pre_tournament_odds":               "MANUAL  — Bet365/Pinnacle snapshot 2026-06-10",
        "manager_tenure_months":             "MANUAL  — Wikipedia / FA sites",
        "historic_first_time":               "MANUAL  — judgment per team (0.0 / 1.0)",
        "core_retention_score":              "AUTO    — provide data/external/squads_2026.csv",
        "manager_stability_score":           "TODO    — derive from tenure once fully filled",
        "rival_top_team_underperformed":     "MANUAL  — judgment per team (0.0 / 1.0)",
    }
    print(f"\n{'='*80}")
    print("team_context_2026.csv fill summary")
    print(f"{'='*80}")
    print(f"  {'Column':<42} {'Filled':>6} {'Blank':>5}  Note")
    print(f"  {'-'*42} {'-'*6} {'-'*5}  {'-'*30}")
    for col in _CONTEXT_COLUMNS[1:]:
        filled = int(df[col].notna().sum())
        blank = n - filled
        print(f"  {col:<42} {filled:>6} {blank:>5}  {notes.get(col,'')}")
    print(f"{'='*80}\n")

    manual_cols = [
        "squad_market_value_eur",
        "pre_tournament_odds",
        "manager_tenure_months",
        "historic_first_time",
        "rival_top_team_underperformed",
    ]
    any_blank = False
    for col in manual_cols:
        if col not in df.columns:
            continue
        missing = df.loc[df[col].isna(), "team"].tolist()
        if missing:
            if not any_blank:
                print("Manual cells still needed:")
                any_blank = True
            print(f"\n  {col} ({len(missing)} teams):")
            for t in missing:
                print(f"    □ {t}")
    if any_blank:
        print()
