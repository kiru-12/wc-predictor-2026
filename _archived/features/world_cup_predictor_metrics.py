"""
world_cup_predictor_metrics.py - Fungible Underdog Metrics for a World Cup Predictor
======================================================================================

This module distils the findings of the underdog-football project into a set
of portable, computable metrics that can be plugged directly into any
tournament prediction system.

---------------------------------------------------------------------------
THE FIVE CORE METRICS (all normalised 0-1 unless noted)
---------------------------------------------------------------------------

  1. upset_potential_index (UPI)         -- composite 0-100 underdog score
  2. resource_gap_ratio (RGR)            -- how much weaker the team is financially
  3. leadership_stability_score (LSS)    -- manager tenure + first-time achievement
  4. cohesion_pressure_score (CPS)       -- squad continuity + defensive identity
  5. motivational_momentum_score (MMS)   -- underdog narrative intensity

---------------------------------------------------------------------------
INTEGRATION GUIDE FOR A WORLD CUP PREDICTOR
---------------------------------------------------------------------------

Step 1 -- Prepare a DataFrame with one row per team per tournament:

    Required columns (fill NaN if unavailable):
      - squad_market_value_eur        (float, e.g. 420_000_000)
      - avg_group_opponent_value_eur  (float)
      - pre_tournament_odds           (float, decimal, e.g. 150.0 = 149-1)
      - manager_tenure_months         (float)
      - historic_first_time           (bool)
      - core_retention_score          (float, 0-100)
      - manager_stability_score       (float, 0-1)
      - rival_top_team_underperformed (bool)

Step 2 -- Call compute_metrics(df) to add all five metric columns.

Step 3 -- Integrate into your predictor using the recommended weights:

    Recommended blend with your existing Elo / strength model:
      new_win_prob = (
          base_elo_prob * 0.65
        + upset_potential_index / 100 * 0.35
      )

    Or as an upset adjustment multiplier on any existing probability:
      adjusted_prob = base_prob * (1 + 0.5 * (upset_potential_index - 50) / 100)

Step 4 -- Threshold interpretation:
      UPI > 70  => strong underdog candidate -- consider bumping their win prob
      UPI 50-70 => moderate underdog -- slight positive adjustment
      UPI < 50  => favourite-ish -- no adjustment or small negative

---------------------------------------------------------------------------
EVIDENCE BASE
---------------------------------------------------------------------------

  Metric                  Key underdog supporting it               Weight
  ----------------------  ---------------------------------------  ------
  resource_gap_ratio      Greece (0.39x), Leicester (0.50x)        0.20
  leadership_stability    Denmark (36mo), Greece (36mo),           0.20
                          Clough (48mo)
  cohesion_pressure       Morocco xGA/game, Forest defensive       0.20
  motivational_moment     Iceland, South Korea (co-hosts)          0.10
  contextual_odds         Leicester 5000-1, SK 500-1               0.17
  tactical_clarity        Morocco, Greece set-piece reliance       0.13

---------------------------------------------------------------------------

Usage
-----
    python world_cup_predictor_metrics.py   # demo on case study data
    from world_cup_predictor_metrics import compute_metrics, upset_potential_index
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import PATHS

logging.basicConfig(level=logging.INFO, format="%(levelname)s -- %(message)s")
logger = logging.getLogger("wc_metrics")

# ---------------------------------------------------------------------------
# Evidence-based weights (must sum to 1.0)
# ---------------------------------------------------------------------------
_WEIGHTS = {
    "resource_gap_ratio":          0.20,
    "leadership_stability_score":  0.20,
    "cohesion_pressure_score":     0.20,
    "contextual_odds_score":       0.17,
    "tactical_clarity_score":      0.13,
    "motivational_momentum_score": 0.10,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Individual metric calculators
# ---------------------------------------------------------------------------

def resource_gap_ratio(squad_value, avg_opponent_value):
    """
    Resource gap = team value / average opponent value.
    Clipped to [0, 2] then inverted so largest underdog -> 1.0.
    Returns float or Series in [0, 1].
    """
    ratio = np.clip(
        pd.to_numeric(squad_value, errors="coerce") /
        pd.to_numeric(avg_opponent_value, errors="coerce").replace(0, np.nan),
        0.0, 2.0,
    )
    return 1.0 - (ratio / 2.0)


def leadership_stability_score(manager_tenure_months, historic_first_time):
    """
    Leadership stability = log(tenure+1) normalised to [0,1]
    + 0.15 bonus for historic_first_time, clipped to 1.0.
    """
    tenure = pd.to_numeric(manager_tenure_months, errors="coerce").clip(lower=0)
    log_t = np.log1p(tenure)
    lo, hi = float(np.nanmin(log_t)), float(np.nanmax(log_t))
    if hi > lo:
        norm: pd.Series = (log_t - lo) / (hi - lo)
    else:
        norm = pd.Series(0.5, index=log_t.index)
    debut_bonus = pd.to_numeric(historic_first_time, errors="coerce").fillna(0) * 0.15
    return (norm + debut_bonus).clip(upper=1.0)


def cohesion_pressure_score(core_retention, manager_stability, defensive_solidity=None):
    """
    Cohesion and defensive identity.
    Weighted blend: 0.50 x core_retention + 0.30 x manager_stability + 0.20 x defensive_solidity.
    """
    ret = pd.to_numeric(core_retention, errors="coerce")
    stab = pd.to_numeric(manager_stability, errors="coerce").clip(0, 1)
    # Auto-scale retention if on 0-100 scale
    if isinstance(ret, pd.Series):
        if ret.max(skipna=True) > 1.5:
            ret = ret / 100.0
    elif ret > 1.5:
        ret = ret / 100.0

    if defensive_solidity is not None:
        def_s = pd.to_numeric(defensive_solidity, errors="coerce")
        valid = def_s.dropna() if isinstance(def_s, pd.Series) else def_s
        if isinstance(valid, pd.Series) and len(valid) >= 2:
            lo, hi = float(valid.min()), float(valid.max())
            def_norm: pd.Series = (def_s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=def_s.index)
        else:
            def_norm = pd.Series(0.5, index=def_s.index)
    else:
        idx = ret.index if isinstance(ret, pd.Series) else range(1)
        def_norm = pd.Series(0.5, index=idx)

    score = 0.50 * ret.fillna(0.5) + 0.30 * stab.fillna(0.5) + 0.20 * def_norm
    return score.clip(0, 1) if isinstance(score, pd.Series) else float(np.clip(score, 0, 1))


def contextual_odds_score(pre_tournament_odds):
    """
    Transform bookmaker odds into a 0-1 underdog signal.
    log(odds) normalised: higher odds (bigger underdog) -> closer to 1.0.
    """
    odds = pd.to_numeric(pre_tournament_odds, errors="coerce").clip(lower=1.0)
    log_odds = np.log1p(odds)
    lo, hi = float(np.nanmin(log_odds)), float(np.nanmax(log_odds))
    if hi > lo:
        norm: pd.Series = (log_odds - lo) / (hi - lo)
    else:
        norm = pd.Series(0.5, index=log_odds.index)
    norm = norm.fillna(norm.median())
    return norm.clip(0, 1) if isinstance(norm, pd.Series) else float(np.clip(norm, 0, 1))


def tactical_clarity_score_metric(press_intensity=None, formation_consistency=None):
    """
    Tactical clarity: average of pressing intensity and formation consistency.
    Both inputs expected in [0, 1].
    """
    signals = []
    if press_intensity is not None:
        signals.append(pd.to_numeric(press_intensity, errors="coerce").fillna(0.5).clip(0, 1))
    if formation_consistency is not None:
        signals.append(pd.to_numeric(formation_consistency, errors="coerce").fillna(0.5).clip(0, 1))
    if not signals:
        return 0.5
    stacked = pd.concat(signals, axis=1) if isinstance(signals[0], pd.Series) else signals
    if isinstance(stacked, pd.DataFrame):
        return stacked.mean(axis=1)
    return float(np.nanmean(stacked))


def motivational_momentum_score(rival_underperformed, key_player_dependency=None, home_nation_bonus=None):
    """
    Motivational / narrative momentum.
    +0.40 rival top team underperformed, +0.35 galvanising star player, +0.25 historic debut.
    Returns score in [0, 1].
    """
    rival = pd.to_numeric(rival_underperformed, errors="coerce").fillna(0)
    star = pd.to_numeric(key_player_dependency, errors="coerce").fillna(0) if key_player_dependency is not None else 0
    home = pd.to_numeric(home_nation_bonus, errors="coerce").fillna(0) if home_nation_bonus is not None else 0
    score = 0.40 * rival + 0.35 * star + 0.25 * home
    return score.clip(0, 1) if isinstance(score, pd.Series) else float(np.clip(score, 0, 1))


# ---------------------------------------------------------------------------
# Composite UPI
# ---------------------------------------------------------------------------

def upset_potential_index(df: pd.DataFrame) -> pd.Series:
    """
    Return the Upset Potential Index (0-100) for every row in df.

    This is the primary fungible metric for a World Cup predictor.
    See module docstring for required columns and integration guide.
    """
    components: dict[str, pd.Series] = {}

    # 1. Resource gap
    if "squad_market_value_eur" in df.columns and "avg_group_opponent_value_eur" in df.columns:
        components["resource_gap_ratio"] = resource_gap_ratio(
            df["squad_market_value_eur"], df["avg_group_opponent_value_eur"]
        )
    elif "squad_value_vs_league_avg" in df.columns:
        ratio = pd.to_numeric(df["squad_value_vs_league_avg"], errors="coerce").clip(0, 2)
        components["resource_gap_ratio"] = 1.0 - (ratio / 2.0)

    # 2. Leadership stability
    if "manager_tenure_months" in df.columns:
        debut = df["historic_first_time"] if "historic_first_time" in df.columns else pd.Series(False, index=df.index)
        components["leadership_stability_score"] = leadership_stability_score(df["manager_tenure_months"], debut)

    # 3. Cohesion & pressure
    ret_col = "core_retention_score" if "core_retention_score" in df.columns else None
    stab_col = "manager_stability_score" if "manager_stability_score" in df.columns else None
    if ret_col or stab_col:
        ret = df[ret_col] if ret_col else pd.Series(0.5, index=df.index)
        stab = df[stab_col] if stab_col else pd.Series(0.5, index=df.index)
        def_ = df["defensive_solidity"] if "defensive_solidity" in df.columns else None
        components["cohesion_pressure_score"] = pd.Series(
            cohesion_pressure_score(ret, stab, def_), index=df.index, dtype=float
        )

    # 4. Contextual odds
    odds_col = ("pre_season_odds" if "pre_season_odds" in df.columns
                else "pre_tournament_odds" if "pre_tournament_odds" in df.columns else None)
    if odds_col:
        components["contextual_odds_score"] = pd.Series(
            contextual_odds_score(df[odds_col]), index=df.index, dtype=float
        )

    # 5. Tactical clarity
    p = df["press_intensity_score"] if "press_intensity_score" in df.columns else None
    f = df["tactical_flexibility"] if "tactical_flexibility" in df.columns else None
    if p is not None or f is not None:
        components["tactical_clarity_score"] = pd.Series(
            tactical_clarity_score_metric(p, f), index=df.index, dtype=float
        )

    # 6. Motivational momentum
    if "rival_top_team_underperformed" in df.columns:
        star = df["key_player_dependency"] if "key_player_dependency" in df.columns else None
        components["motivational_momentum_score"] = pd.Series(
            motivational_momentum_score(df["rival_top_team_underperformed"], star), index=df.index, dtype=float
        )

    if not components:
        logger.warning("No computable components -- returning 50.0 for all rows.")
        return pd.Series(50.0, index=df.index)

    scores_df = pd.DataFrame(components, index=df.index)
    available_weights = {k: _WEIGHTS[k] for k in components if k in _WEIGHTS}
    total_w = sum(available_weights.values())

    upi_raw = pd.Series(0.0, index=df.index)
    for col, w in available_weights.items():
        upi_raw += scores_df[col].fillna(0.5) * (w / total_w)

    lo, hi = upi_raw.min(), upi_raw.max()
    upi = (upi_raw - lo) / (hi - lo) * 100 if hi > lo else upi_raw * 100
    return upi.round(2)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all six metric columns plus the UPI to df.

    New columns added:
      resource_gap_ratio, leadership_stability_score, cohesion_pressure_score,
      contextual_odds_score, tactical_clarity_score, motivational_momentum_score,
      upset_potential_index (0-100, the primary metric).
    """
    out = df.copy()

    if "squad_market_value_eur" in out.columns and "avg_group_opponent_value_eur" in out.columns:
        out["resource_gap_ratio"] = resource_gap_ratio(out["squad_market_value_eur"], out["avg_group_opponent_value_eur"])
    elif "squad_value_vs_league_avg" in out.columns:
        ratio = pd.to_numeric(out["squad_value_vs_league_avg"], errors="coerce").clip(0, 2)
        out["resource_gap_ratio"] = 1.0 - (ratio / 2.0)
    else:
        out["resource_gap_ratio"] = np.nan

    if "manager_tenure_months" in out.columns:
        debut = out["historic_first_time"] if "historic_first_time" in out.columns else pd.Series(False, index=out.index)
        out["leadership_stability_score"] = leadership_stability_score(out["manager_tenure_months"], debut)
    else:
        out["leadership_stability_score"] = np.nan

    ret = out["core_retention_score"] if "core_retention_score" in out.columns else pd.Series(0.5, index=out.index)
    stab = out["manager_stability_score"] if "manager_stability_score" in out.columns else pd.Series(0.5, index=out.index)
    def_ = out["defensive_solidity"] if "defensive_solidity" in out.columns else None
    out["cohesion_pressure_score"] = cohesion_pressure_score(ret, stab, def_)

    odds_col = ("pre_season_odds" if "pre_season_odds" in out.columns
                else "pre_tournament_odds" if "pre_tournament_odds" in out.columns else None)
    out["contextual_odds_score"] = contextual_odds_score(out[odds_col]) if odds_col else np.nan

    p = out["press_intensity_score"] if "press_intensity_score" in out.columns else None
    f = out["tactical_flexibility"] if "tactical_flexibility" in out.columns else None
    out["tactical_clarity_score"] = tactical_clarity_score_metric(p, f)

    if "rival_top_team_underperformed" in out.columns:
        star = out["key_player_dependency"] if "key_player_dependency" in out.columns else None
        out["motivational_momentum_score"] = motivational_momentum_score(out["rival_top_team_underperformed"], star)
    else:
        out["motivational_momentum_score"] = np.nan

    out["upset_potential_index"] = upset_potential_index(out)
    return out


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main() -> None:
    dna_path = PATHS["processed"] / "dna_scores.csv"
    if not dna_path.exists():
        print("Run dna_score.py first to generate dna_scores.csv")
        sys.exit(1)

    df = pd.read_csv(dna_path)
    print(f"Loaded {len(df)} teams from {dna_path.name}")

    out = compute_metrics(df)

    metric_cols = [
        "team", "season",
        "resource_gap_ratio",
        "leadership_stability_score",
        "cohesion_pressure_score",
        "contextual_odds_score",
        "tactical_clarity_score",
        "motivational_momentum_score",
        "upset_potential_index",
    ]
    display_cols = [c for c in metric_cols if c in out.columns]
    display = out[display_cols].sort_values("upset_potential_index", ascending=False)

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)

    print()
    print("=" * 90)
    print("WORLD CUP PREDICTOR METRICS -- Underdog Teams")
    print("=" * 90)
    print(display.to_string(index=False))
    print()
    print("-" * 90)
    print("METRIC LEGEND:")
    print("  resource_gap_ratio          -> 1.0 = financially weakest team")
    print("  leadership_stability_score  -> 1.0 = longest-serving manager + debut achievement")
    print("  cohesion_pressure_score     -> 1.0 = most retained squad + strongest defence")
    print("  contextual_odds_score       -> 1.0 = biggest pre-tournament underdog by odds")
    print("  tactical_clarity_score      -> 1.0 = most disciplined / pressing system")
    print("  motivational_momentum_score -> 1.0 = rival faltered + star player + historic debut")
    print("  upset_potential_index (UPI) -> 0-100 composite; >70 = strong upset candidate")
    print()
    print("INTEGRATION FORMULA:")
    print("  adjusted_win_prob = elo_win_prob * 0.65")
    print("                    + (upset_potential_index / 100) * 0.35")
    print("-" * 90)

    out_path = PATHS["processed"] / "wc_predictor_metrics.csv"
    out[display_cols].to_csv(out_path, index=False)
    print(f"\nSaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
