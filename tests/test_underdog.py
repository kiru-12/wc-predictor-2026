"""Tests for features/underdog.py.

Uses a small 4-team synthetic fixture. No network access required.

Fixture design
--------------
  Alpha   – all context columns present, strong underdog profile
  Beta    – all context columns present, clear favourite profile
  Gamma   – missing 'squad_market_value_eur' and 'avg_group_opponent_value_eur'
  Delta   – present in baseline but absent from context CSV entirely
"""

import math
import io
from pathlib import Path

import pandas as pd
import pytest

from features.underdog import (
    _REQUIRED_CONTEXT_COLS,
    _OUTPUT_COLS,
    _load_context,
    _join_context_to_baseline,
    build_underdog_features,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def context_csv(tmp_path: Path) -> Path:
    """Write a minimal 3-row context CSV (Alpha, Beta, Gamma) to tmp_path."""
    rows = [
        # Alpha: small squad value vs big opponents → big underdog
        {
            "team": "Alpha",
            "squad_market_value_eur": 80_000_000,
            "avg_group_opponent_value_eur": 600_000_000,
            "pre_tournament_odds": 250.0,
            "manager_tenure_months": 48,
            "historic_first_time": True,
            "core_retention_score": 85.0,
            "manager_stability_score": 0.90,
            "rival_top_team_underperformed": True,
        },
        # Beta: huge squad value → clear favourite
        {
            "team": "Beta",
            "squad_market_value_eur": 900_000_000,
            "avg_group_opponent_value_eur": 350_000_000,
            "pre_tournament_odds": 5.0,
            "manager_tenure_months": 12,
            "historic_first_time": False,
            "core_retention_score": 70.0,
            "manager_stability_score": 0.50,
            "rival_top_team_underperformed": False,
        },
        # Gamma: missing market-value columns
        {
            "team": "Gamma",
            "squad_market_value_eur": None,
            "avg_group_opponent_value_eur": None,
            "pre_tournament_odds": 80.0,
            "manager_tenure_months": 24,
            "historic_first_time": False,
            "core_retention_score": 75.0,
            "manager_stability_score": 0.70,
            "rival_top_team_underperformed": False,
        },
    ]
    p = tmp_path / "team_context.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


@pytest.fixture()
def baseline_df() -> pd.DataFrame:
    """4-team baseline table; Delta is absent from the context CSV."""
    return pd.DataFrame({
        "team": ["Alpha", "Beta", "Gamma", "Delta"],
        "elo_national": [1450.0, 1800.0, 1500.0, 1600.0],
        "baseline_score": [-0.8, 1.2, -0.1, 0.3],
    })


# ---------------------------------------------------------------------------
# _load_context
# ---------------------------------------------------------------------------

class TestLoadContext:
    def test_loads_valid_csv(self, context_csv):
        ctx = _load_context(context_csv)
        assert len(ctx) == 3
        assert "team" in ctx.columns

    def test_missing_file_returns_empty(self, tmp_path):
        ctx = _load_context(tmp_path / "nonexistent.csv")
        assert ctx.empty

    def test_bool_columns_coerced_to_numeric(self, context_csv):
        ctx = _load_context(context_csv)
        assert ctx["historic_first_time"].dtype in (float, int)
        assert set(ctx["historic_first_time"].dropna().unique()).issubset({0.0, 1.0})

    def test_missing_team_column_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("value\n1\n2\n")
        ctx = _load_context(bad)
        assert ctx.empty

    def test_missing_required_col_does_not_crash(self, tmp_path):
        """A CSV without squad_market_value_eur should still load."""
        p = tmp_path / "partial.csv"
        p.write_text("team,pre_tournament_odds\nAlpha,100.0\n")
        ctx = _load_context(p)
        assert len(ctx) == 1


# ---------------------------------------------------------------------------
# _join_context_to_baseline
# ---------------------------------------------------------------------------

class TestJoinContextToBaseline:
    def test_all_baseline_teams_present_in_output(self, baseline_df, context_csv):
        ctx = _load_context(context_csv)
        merged = _join_context_to_baseline(baseline_df, ctx)
        assert set(merged["team"]) == set(baseline_df["team"])

    def test_delta_has_nan_context(self, baseline_df, context_csv):
        """Delta is in baseline but absent from context → NaN market value."""
        ctx = _load_context(context_csv)
        merged = _join_context_to_baseline(baseline_df, ctx)
        delta_row = merged[merged["team"] == "Delta"].iloc[0]
        assert math.isnan(delta_row["squad_market_value_eur"])

    def test_alpha_context_preserved(self, baseline_df, context_csv):
        ctx = _load_context(context_csv)
        merged = _join_context_to_baseline(baseline_df, ctx)
        alpha_row = merged[merged["team"] == "Alpha"].iloc[0]
        assert alpha_row["squad_market_value_eur"] == pytest.approx(80_000_000)

    def test_missing_team_col_in_baseline(self, context_csv):
        ctx = _load_context(context_csv)
        bad_baseline = pd.DataFrame({"elo": [1500.0]})
        # Should not crash — falls back to context_df
        result = _join_context_to_baseline(bad_baseline, ctx)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# build_underdog_features — normal case
# ---------------------------------------------------------------------------

class TestBuildUnderdogFeaturesNormal:
    def test_output_shape(self, baseline_df, context_csv, tmp_path):
        df = build_underdog_features(
            baseline_df, context_csv=context_csv, output_path=tmp_path / "out.csv"
        )
        assert len(df) == 4  # all baseline teams
        assert set(_OUTPUT_COLS).issubset(set(df.columns))

    def test_csv_written(self, baseline_df, context_csv, tmp_path):
        out = tmp_path / "underdog.csv"
        build_underdog_features(baseline_df, context_csv=context_csv, output_path=out)
        assert out.exists()
        saved = pd.read_csv(out)
        assert len(saved) == 4

    def test_upi_bounds(self, baseline_df, context_csv, tmp_path):
        """UPI must be in [0, 100] for every team."""
        df = build_underdog_features(
            baseline_df, context_csv=context_csv, output_path=tmp_path / "out.csv"
        )
        valid = df["upset_potential_index"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_alpha_higher_upi_than_beta(self, baseline_df, context_csv, tmp_path):
        """Alpha is the bigger underdog so should have a higher UPI than Beta."""
        df = build_underdog_features(
            baseline_df, context_csv=context_csv, output_path=tmp_path / "out.csv"
        )
        alpha_upi = df.loc[df["team"] == "Alpha", "upset_potential_index"].iloc[0]
        beta_upi = df.loc[df["team"] == "Beta", "upset_potential_index"].iloc[0]
        assert alpha_upi > beta_upi

    def test_output_columns_exact(self, baseline_df, context_csv, tmp_path):
        df = build_underdog_features(
            baseline_df, context_csv=context_csv, output_path=tmp_path / "out.csv"
        )
        assert list(df.columns) == _OUTPUT_COLS

    def test_default_output_path(self, baseline_df, context_csv, tmp_path, monkeypatch):
        import features.underdog as ud
        monkeypatch.setattr(ud, "PROCESSED_DIR", tmp_path)
        build_underdog_features(baseline_df, context_csv=context_csv)
        assert (tmp_path / "underdog_features.csv").exists()


# ---------------------------------------------------------------------------
# build_underdog_features — missing input column case
# ---------------------------------------------------------------------------

class TestBuildUnderdogFeaturesMissingCol:
    def test_gamma_missing_market_value_does_not_crash(
        self, baseline_df, context_csv, tmp_path
    ):
        """Gamma has NaN squad_market_value_eur — must not raise."""
        df = build_underdog_features(
            baseline_df, context_csv=context_csv, output_path=tmp_path / "out.csv"
        )
        assert "Gamma" in df["team"].values

    def test_gamma_resource_gap_is_nan(self, baseline_df, context_csv, tmp_path):
        """resource_gap_ratio requires market value — must be NaN for Gamma."""
        df = build_underdog_features(
            baseline_df, context_csv=context_csv, output_path=tmp_path / "out.csv"
        )
        gamma_rgr = df.loc[df["team"] == "Gamma", "resource_gap_ratio"].iloc[0]
        assert math.isnan(gamma_rgr)

    def test_gamma_upi_still_finite(self, baseline_df, context_csv, tmp_path):
        """Even with a missing component, UPI should be a finite number."""
        df = build_underdog_features(
            baseline_df, context_csv=context_csv, output_path=tmp_path / "out.csv"
        )
        gamma_upi = df.loc[df["team"] == "Gamma", "upset_potential_index"].iloc[0]
        assert not math.isnan(gamma_upi)

    def test_delta_absent_from_context_does_not_crash(
        self, baseline_df, context_csv, tmp_path
    ):
        """Delta is in baseline but not in context CSV — must not raise."""
        df = build_underdog_features(
            baseline_df, context_csv=context_csv, output_path=tmp_path / "out.csv"
        )
        assert "Delta" in df["team"].values

    def test_empty_context_returns_empty(self, baseline_df, tmp_path):
        """If context CSV doesn't exist, return empty DataFrame gracefully."""
        df = build_underdog_features(
            baseline_df,
            context_csv=tmp_path / "nonexistent.csv",
            output_path=tmp_path / "out.csv",
        )
        assert df.empty or len(df) == 0


# ---------------------------------------------------------------------------
# UPI bounds — property test with extreme values
# ---------------------------------------------------------------------------

class TestUPIBoundsExtreme:
    def test_all_favourites_upi_range(self, tmp_path):
        """Even a table of all favourites must have UPI in [0, 100]."""
        ctx = pd.DataFrame([
            {"team": "T1", "squad_market_value_eur": 1e9,
             "avg_group_opponent_value_eur": 4e8, "pre_tournament_odds": 3.0,
             "manager_tenure_months": 8, "historic_first_time": False,
             "core_retention_score": 65.0, "manager_stability_score": 0.4,
             "rival_top_team_underperformed": False},
            {"team": "T2", "squad_market_value_eur": 8e8,
             "avg_group_opponent_value_eur": 4e8, "pre_tournament_odds": 4.0,
             "manager_tenure_months": 10, "historic_first_time": False,
             "core_retention_score": 68.0, "manager_stability_score": 0.5,
             "rival_top_team_underperformed": False},
        ])
        ctx_path = tmp_path / "ctx.csv"
        ctx.to_csv(ctx_path, index=False)
        baseline = pd.DataFrame({"team": ["T1", "T2"], "elo_national": [1800.0, 1750.0]})
        df = build_underdog_features(baseline, context_csv=ctx_path, output_path=tmp_path / "out.csv")
        valid = df["upset_potential_index"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_all_underdogs_upi_range(self, tmp_path):
        """All extreme underdogs must also stay within 0-100."""
        ctx = pd.DataFrame([
            {"team": "U1", "squad_market_value_eur": 5e6,
             "avg_group_opponent_value_eur": 9e8, "pre_tournament_odds": 1000.0,
             "manager_tenure_months": 60, "historic_first_time": True,
             "core_retention_score": 95.0, "manager_stability_score": 0.99,
             "rival_top_team_underperformed": True},
            {"team": "U2", "squad_market_value_eur": 8e6,
             "avg_group_opponent_value_eur": 7e8, "pre_tournament_odds": 800.0,
             "manager_tenure_months": 48, "historic_first_time": True,
             "core_retention_score": 90.0, "manager_stability_score": 0.95,
             "rival_top_team_underperformed": True},
        ])
        ctx_path = tmp_path / "ctx.csv"
        ctx.to_csv(ctx_path, index=False)
        baseline = pd.DataFrame({"team": ["U1", "U2"], "elo_national": [1300.0, 1350.0]})
        df = build_underdog_features(baseline, context_csv=ctx_path, output_path=tmp_path / "out.csv")
        valid = df["upset_potential_index"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()
