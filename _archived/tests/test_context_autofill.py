"""Tests for features/context_autofill.py.

All tests use synthetic fixtures — no network access.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.context_autofill import (
    TRANSFERMARKT_ALIAS_MAP,
    apply_canonical_names,
    compute_avg_group_opponent_value,
    compute_core_retention,
    normalise_tenure_to_stability,
    parse_transfermarkt_value,
    _fill_missing,
    _parse_tm_html,
    build_team_context_csv,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def groups_df() -> pd.DataFrame:
    """Two groups of 4 teams each."""
    return pd.DataFrame({
        "team":  ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
        "group": ["A",  "A",  "A",  "A",  "B",  "B",  "B",  "B"],
        "confederation": ["UEFA"] * 8,
        "confirmed": [True] * 8,
    })


@pytest.fixture()
def context_df() -> pd.DataFrame:
    """Context table with squad values for all 8 teams."""
    return pd.DataFrame({
        "team": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
        "squad_market_value_eur": [
            100e6, 200e6, 300e6, 400e6,
            500e6, 600e6, 700e6, 800e6,
        ],
    })


@pytest.fixture()
def squads_2022() -> pd.DataFrame:
    """Minimal WC 2022 roster for 3 teams."""
    rows = []
    pid = 1
    for team, players in {
        "Alpha": ["P1", "P2", "P3", "P4", "P5"],
        "Beta":  ["P6", "P7", "P8", "P9", "P10"],
        "Gamma": ["P11", "P12", "P13"],
    }.items():
        for p in players:
            rows.append({"team": team, "player_name": p, "player_id": pid})
            pid += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def squads_2026() -> pd.DataFrame:
    """2026 squad with partial overlap vs 2022."""
    rows = []
    for team, players in {
        # Alpha: 3 of 5 retained → 60%
        "Alpha": ["P1", "P2", "P3", "NEW1", "NEW2"],
        # Beta: 5 of 5 retained → 100%
        "Beta":  ["P6", "P7", "P8", "P9", "P10"],
        # Gamma: 0 of 3 retained → 0%
        "Gamma": ["NEW3", "NEW4", "NEW5"],
    }.items():
        for p in players:
            rows.append({"team": team, "player_name": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_avg_group_opponent_value
# ---------------------------------------------------------------------------

class TestComputeAvgGroupOpponentValue:
    def test_basic_calculation_group_a(self, context_df, groups_df):
        """T1 (100M) in Group A: avg of T2+T3+T4 = (200+300+400)/3 = 300M."""
        result = compute_avg_group_opponent_value(context_df, groups_df)
        assert result["T1"] == pytest.approx(300e6)

    def test_basic_calculation_group_b(self, context_df, groups_df):
        """T5 (500M): avg of T6+T7+T8 = (600+700+800)/3 ≈ 700M."""
        result = compute_avg_group_opponent_value(context_df, groups_df)
        assert result["T5"] == pytest.approx(700e6)

    def test_symmetric_within_group(self, context_df, groups_df):
        """All Group A teams' avg opponents must use all other 3 members."""
        result = compute_avg_group_opponent_value(context_df, groups_df)
        # Weighted sum check: T1+T2+T3+T4 = 1000M
        # T1 avg = (200+300+400)/3 = 300
        # T2 avg = (100+300+400)/3 = 266.67
        assert result["T2"] == pytest.approx((100+300+400)/3 * 1e6)

    def test_single_group_member_opponent_is_their_value(self, context_df):
        """With only 2 teams in a group, each team's opponent avg = the other's value."""
        grp = pd.DataFrame({
            "team":  ["T1", "T2"],
            "group": ["X", "X"],
        })
        result = compute_avg_group_opponent_value(
            context_df[context_df["team"].isin(["T1", "T2"])], grp
        )
        # T1 opponent avg = T2's value = 200M
        assert result["T1"] == pytest.approx(200e6)

    def test_missing_squad_value_gives_nan(self, groups_df):
        """A team with NaN squad value: opponent avg excludes it; that team gets NaN."""
        ctx = pd.DataFrame({
            "team": ["T1", "T2", "T3", "T4"],
            "squad_market_value_eur": [100e6, np.nan, 300e6, 400e6],
        })
        result = compute_avg_group_opponent_value(ctx, groups_df[groups_df["group"] == "A"])
        # T2 has no squad value; T1's avg = (300+400)/2 = 350M (T2 excluded)
        assert result["T1"] == pytest.approx(350e6)
        # T2's own avg: opponents are T1, T3, T4 = (100+300+400)/3 = 266.67M
        assert result["T2"] == pytest.approx((100+300+400)/3 * 1e6)

    def test_missing_context_column_returns_nan_series(self, groups_df):
        bad = pd.DataFrame({"team": ["T1", "T2"]})
        result = compute_avg_group_opponent_value(bad, groups_df)
        assert result.isna().all()

    def test_missing_group_column_returns_nan_series(self, context_df):
        bad = pd.DataFrame({"team": ["T1", "T2"]})
        result = compute_avg_group_opponent_value(context_df, bad)
        assert result.isna().all()

    def test_returns_series_indexed_by_team(self, context_df, groups_df):
        result = compute_avg_group_opponent_value(context_df, groups_df)
        assert isinstance(result, pd.Series)
        assert "T1" in result.index


# ---------------------------------------------------------------------------
# compute_core_retention
# ---------------------------------------------------------------------------

class TestComputeCoreRetention:
    def test_alpha_60_percent(self, squads_2022, squads_2026):
        result = compute_core_retention(squads_2022, squads_2026)
        assert result["Alpha"] == pytest.approx(60.0)

    def test_beta_100_percent(self, squads_2022, squads_2026):
        result = compute_core_retention(squads_2022, squads_2026)
        assert result["Beta"] == pytest.approx(100.0)

    def test_gamma_0_percent(self, squads_2022, squads_2026):
        result = compute_core_retention(squads_2022, squads_2026)
        assert result["Gamma"] == pytest.approx(0.0)

    def test_empty_2022_returns_nan_series(self, squads_2026):
        empty = pd.DataFrame(columns=["team", "player_name"])
        result = compute_core_retention(empty, squads_2026)
        assert result.isna().all() or result.empty

    def test_empty_2026_returns_nan_series(self, squads_2022):
        empty = pd.DataFrame(columns=["team", "player_name"])
        result = compute_core_retention(squads_2022, empty)
        assert result.isna().all() or result.empty

    def test_missing_column_returns_nan_series(self, squads_2022, squads_2026):
        bad = squads_2026.drop(columns=["player_name"])
        result = compute_core_retention(squads_2022, bad)
        assert result.isna().all() or result.empty

    def test_result_bounds(self, squads_2022, squads_2026):
        result = compute_core_retention(squads_2022, squads_2026)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_match_on_player_id(self, squads_2022, squads_2026):
        """Matching on player_id still works for teams with overlapping IDs."""
        s22 = squads_2022.copy()
        s26 = squads_2026.copy()
        # Assign player_id to s26 for Alpha only (same as s22)
        id_map = dict(zip(s22["player_name"], s22["player_id"]))
        s26["player_id"] = s26["player_name"].map(id_map).fillna(-1).astype(int)
        result = compute_core_retention(s22, s26, match_on="player_id")
        # Alpha: P1(id), P2(id), P3(id) match; NEW1/NEW2 get id=-1 no match
        assert result["Alpha"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# normalise_tenure_to_stability
# ---------------------------------------------------------------------------

class TestNormaliseTenureToStability:
    def test_range_0_to_1(self):
        tenure = pd.Series([0, 12, 24, 60], name="t")
        result = normalise_tenure_to_stability(tenure)
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)

    def test_longer_tenure_higher_score(self):
        tenure = pd.Series([6, 12, 36], name="t")
        result = normalise_tenure_to_stability(tenure)
        assert result.iloc[0] < result.iloc[1] < result.iloc[2]

    def test_nan_tenure_gives_nan_stability(self):
        tenure = pd.Series([12.0, np.nan, 36.0])
        result = normalise_tenure_to_stability(tenure)
        assert math.isnan(result.iloc[1])
        assert not math.isnan(result.iloc[0])

    def test_all_same_returns_half(self):
        tenure = pd.Series([24, 24, 24])
        result = normalise_tenure_to_stability(tenure)
        assert all(abs(v - 0.5) < 1e-9 for v in result)

    def test_single_value_returns_half(self):
        tenure = pd.Series([36])
        result = normalise_tenure_to_stability(tenure)
        assert result.iloc[0] == pytest.approx(0.5)

    def test_zero_tenure_gets_zero(self):
        tenure = pd.Series([0, 12, 48])
        result = normalise_tenure_to_stability(tenure)
        assert result.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _fill_missing
# ---------------------------------------------------------------------------

class TestFillMissing:
    def test_fills_nan_cells(self):
        base = pd.DataFrame({"team": ["A", "B"], "val": [np.nan, 2.0]})
        computed = pd.DataFrame({"team": ["A"], "val": [99.0]})
        result = _fill_missing(base, computed)
        assert result.loc[result["team"] == "A", "val"].iloc[0] == pytest.approx(99.0)

    def test_does_not_overwrite_existing(self):
        base = pd.DataFrame({"team": ["A"], "val": [5.0]})
        computed = pd.DataFrame({"team": ["A"], "val": [99.0]})
        result = _fill_missing(base, computed)
        assert result.loc[result["team"] == "A", "val"].iloc[0] == pytest.approx(5.0)

    def test_adds_new_column(self):
        base = pd.DataFrame({"team": ["A", "B"]})
        computed = pd.DataFrame({"team": ["A"], "new_col": [7.0]})
        result = _fill_missing(base, computed)
        assert "new_col" in result.columns
        assert result.loc[result["team"] == "A", "new_col"].iloc[0] == pytest.approx(7.0)
        assert math.isnan(result.loc[result["team"] == "B", "new_col"].iloc[0])


# ---------------------------------------------------------------------------
# build_team_context_csv (integration)
# ---------------------------------------------------------------------------

class TestBuildTeamContextCsv:
    def _write_groups(self, tmp_path: Path, n: int = 8) -> Path:
        teams = [f"T{i}" for i in range(1, n + 1)]
        groups = ["A"] * 4 + ["B"] * 4
        p = tmp_path / "groups.csv"
        pd.DataFrame({
            "team": teams[:n],
            "group": groups[:n],
            "confederation": ["UEFA"] * n,
            "confirmed": [True] * n,
        }).to_csv(p, index=False)
        return p

    def _write_context(self, tmp_path: Path) -> Path:
        p = tmp_path / "ctx.csv"
        pd.DataFrame([{
            "team": "T1",
            "squad_market_value_eur": 100e6,
            "avg_group_opponent_value_eur": np.nan,
            "pre_tournament_odds": 50.0,
            "manager_tenure_months": 24,
            "historic_first_time": False,
            "core_retention_score": np.nan,
            "manager_stability_score": np.nan,
            "rival_top_team_underperformed": False,
        }]).to_csv(p, index=False)
        return p

    def test_output_has_all_teams(self, tmp_path):
        gp = self._write_groups(tmp_path)
        df = build_team_context_csv(
            groups_path=gp,
            existing_context_path=tmp_path / "nonexistent.csv",
            output_path=tmp_path / "out.csv",
            scrape_transfermarkt=False,
        )
        assert len(df) == 8

    def test_csv_written(self, tmp_path):
        gp = self._write_groups(tmp_path)
        out = tmp_path / "out.csv"
        build_team_context_csv(groups_path=gp, output_path=out, scrape_transfermarkt=False)
        assert out.exists()

    def test_preserves_existing_pre_tournament_odds(self, tmp_path):
        gp = self._write_groups(tmp_path)
        ctx = self._write_context(tmp_path)
        out = tmp_path / "out.csv"
        df = build_team_context_csv(groups_path=gp, existing_context_path=ctx, output_path=out,
                                     scrape_transfermarkt=False)
        assert df.loc[df["team"] == "T1", "pre_tournament_odds"].iloc[0] == pytest.approx(50.0)

    def test_avg_opponent_value_computed(self, tmp_path):
        """T1 with value 100M in Group A should get avg of T2/T3/T4."""
        gp = self._write_groups(tmp_path)
        ctx_rows = []
        for i, v in enumerate([100, 200, 300, 400, 500, 600, 700, 800], 1):
            ctx_rows.append({
                "team": f"T{i}",
                "squad_market_value_eur": v * 1e6,
            })
        ctx = tmp_path / "ctx.csv"
        pd.DataFrame(ctx_rows).to_csv(ctx, index=False)
        out = tmp_path / "out.csv"
        df = build_team_context_csv(groups_path=gp, existing_context_path=ctx,
                                     output_path=out, scrape_transfermarkt=False)
        t1_avg = df.loc[df["team"] == "T1", "avg_group_opponent_value_eur"].iloc[0]
        assert t1_avg == pytest.approx(300e6)

    def test_manager_stability_not_auto_computed(self, tmp_path):
        """manager_stability_score is NOT auto-computed from tenure (see TODO in module)."""
        gp = self._write_groups(tmp_path)
        ctx = tmp_path / "ctx.csv"
        pd.DataFrame([
            {"team": "T1", "manager_tenure_months": 6},
            {"team": "T2", "manager_tenure_months": 60},
        ]).to_csv(ctx, index=False)
        df = build_team_context_csv(
            groups_path=gp,
            existing_context_path=ctx,
            output_path=tmp_path / "out.csv",
            scrape_transfermarkt=False,
        )
        # Both should be NaN because automation was deliberately removed.
        assert df["manager_stability_score"].isna().all()

    def test_core_retention_injected(self, tmp_path, squads_2022, squads_2026):
        """Inject WC 2022 squads and provide squads_2026 to compute retention."""
        gp = tmp_path / "groups.csv"
        pd.DataFrame({
            "team": ["Alpha", "Beta", "Gamma"],
            "group": ["A", "A", "A"],
            "confederation": ["UEFA"] * 3,
            "confirmed": [True] * 3,
        }).to_csv(gp, index=False)

        s26_path = tmp_path / "squads_2026.csv"
        squads_2026.to_csv(s26_path, index=False)

        df = build_team_context_csv(
            groups_path=gp,
            existing_context_path=tmp_path / "none.csv",
            squads_2026_path=s26_path,
            wc2022_squads_df=squads_2022,
            output_path=tmp_path / "out.csv",
            scrape_transfermarkt=False,
        )
        alpha_ret = df.loc[df["team"] == "Alpha", "core_retention_score"].iloc[0]
        assert alpha_ret == pytest.approx(60.0)

    def test_column_order_matches_spec(self, tmp_path):
        from features.context_autofill import _CONTEXT_COLUMNS
        gp = self._write_groups(tmp_path)
        df = build_team_context_csv(groups_path=gp, output_path=tmp_path / "out.csv",
                                     scrape_transfermarkt=False)
        assert list(df.columns) == _CONTEXT_COLUMNS

    def test_missing_groups_file_returns_empty(self, tmp_path):
        df = build_team_context_csv(
            groups_path=tmp_path / "nonexistent.csv",
            output_path=tmp_path / "out.csv",
            scrape_transfermarkt=False,
        )
        assert df.empty


# ---------------------------------------------------------------------------
# parse_transfermarkt_value
# ---------------------------------------------------------------------------

class TestParseTransfermarktValue:
    @pytest.mark.parametrize("raw, expected", [
        ("€1.23bn",        1_230_000_000.0),
        ("€450.00m",         450_000_000.0),
        ("€7.50k",                 7_500.0),
        ("1.5bn",          1_500_000_000.0),
        ("200m",             200_000_000.0),
        ("€1,230,000,000",  1_230_000_000.0),  # comma-separated
        ("€0.50m",               500_000.0),
    ])
    def test_known_formats(self, raw, expected):
        assert parse_transfermarkt_value(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["-", "n/a", "—", "", "  "])
    def test_dash_and_empty_returns_none(self, raw):
        assert parse_transfermarkt_value(raw) is None

    def test_none_input_returns_none(self):
        assert parse_transfermarkt_value(None) is None  # type: ignore

    def test_integer_string(self):
        assert parse_transfermarkt_value("500000000") == pytest.approx(500_000_000.0)

    def test_case_insensitive_suffix(self):
        # 'M' and 'B' upper-case after lower() stripping
        assert parse_transfermarkt_value("€300.00M") == pytest.approx(300_000_000.0)


# ---------------------------------------------------------------------------
# apply_canonical_names
# ---------------------------------------------------------------------------

class TestApplyCanonicalNames:
    """Test alias mapping with a synthetic alias map and canonical team list."""

    _ALIAS = {
        "Türkiye":          "Turkey",
        "Turkey":           "Turkey",
        "Côte d'Ivoire":    "Ivory Coast",
        "Ivory Coast":      "Ivory Coast",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "Korea Republic":   "South Korea",
        "South Korea":      "South Korea",
        "DR Congo":         "DR Congo",
        "Dem. Rep. Congo":  "DR Congo",
        "France":           "France",
        "Germany":          "Germany",
        "Brazil":           "Brazil",
    }
    _CANONICAL = [
        "Turkey", "Ivory Coast", "Bosnia and Herzegovina",
        "South Korea", "DR Congo", "France", "Germany", "Brazil",
    ]

    def _raw(self, names: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "team": names,
            "squad_market_value_eur": [1e8] * len(names),
        })

    def test_direct_canonical_name_passes_through(self):
        raw = self._raw(["France", "Germany"])
        out = apply_canonical_names(raw, self._CANONICAL, self._ALIAS)
        assert set(out["team"]) == {"France", "Germany"}

    def test_alias_resolved_correctly(self):
        raw = self._raw(["Türkiye", "Côte d'Ivoire", "Korea Republic"])
        out = apply_canonical_names(raw, self._CANONICAL, self._ALIAS)
        assert "Turkey" in out["team"].values
        assert "Ivory Coast" in out["team"].values
        assert "South Korea" in out["team"].values

    def test_multiword_alias(self):
        raw = self._raw(["Bosnia-Herzegovina"])
        out = apply_canonical_names(raw, self._CANONICAL, self._ALIAS)
        assert out["team"].iloc[0] == "Bosnia and Herzegovina"

    def test_unmatched_team_dropped_and_logged(self, caplog):
        import logging
        raw = self._raw(["Atlantis FC", "France"])
        with caplog.at_level(logging.WARNING, logger="features.context_autofill"):
            out = apply_canonical_names(raw, self._CANONICAL, self._ALIAS)
        assert "Atlantis FC" not in out["team"].values
        assert "France" in out["team"].values
        assert "Atlantis FC" in caplog.text

    def test_multiple_unmatched_all_logged(self, caplog):
        import logging
        raw = self._raw(["Ghost1", "Ghost2", "Brazil"])
        with caplog.at_level(logging.WARNING, logger="features.context_autofill"):
            out = apply_canonical_names(raw, self._CANONICAL, self._ALIAS)
        assert len(out) == 1
        assert "Ghost1" in caplog.text
        assert "Ghost2" in caplog.text

    def test_all_unmatched_returns_empty(self):
        raw = self._raw(["Wakanda", "Narnia"])
        out = apply_canonical_names(raw, self._CANONICAL, self._ALIAS)
        assert out.empty

    def test_case_insensitive_fallback(self):
        raw = self._raw(["france"])  # lower-case
        out = apply_canonical_names(raw, self._CANONICAL, self._ALIAS)
        assert "France" in out["team"].values

    def test_missing_name_col_returns_copy(self):
        raw = pd.DataFrame({"value": [1e8]})
        out = apply_canonical_names(raw, self._CANONICAL, self._ALIAS)
        assert "value" in out.columns

    def test_default_alias_map_covers_all_48_canonical_teams(self):
        """Every team in groups_2026.csv must appear as a value in the default map."""
        canonical_in_map = set(TRANSFERMARKT_ALIAS_MAP.values())
        from pathlib import Path
        groups_path = (
            Path(__file__).resolve().parent.parent
            / "data" / "external" / "groups_2026.csv"
        )
        if not groups_path.exists():
            pytest.skip("groups_2026.csv not found")
        canonical_teams = pd.read_csv(groups_path)["team"].tolist()
        missing = [t for t in canonical_teams if t not in canonical_in_map]
        assert missing == [], f"Teams not in alias map: {missing}"


# ---------------------------------------------------------------------------
# _parse_tm_html — synthetic HTML fixture
# ---------------------------------------------------------------------------

class TestParseTmHtml:
    """Tests for the HTML table parser using synthetic page content."""

    def _make_html(self, rows: list[tuple[str, str]]) -> str:
        """Build a minimal Transfermarkt-like HTML page.
        
        Pads to at least 20 data rows (required by the >=20 sanity check)
        using dummy rows with parseable values.
        """
        all_rows = list(rows)
        while len(all_rows) < 20:
            all_rows.append((f"PadTeam{len(all_rows)}", "€50.00m"))
        row_html = "\n".join(
            f"<tr><td>{name}</td><td>{value}</td></tr>"
            for name, value in all_rows
        )
        return f"""
        <html><body>
        <table>
          <thead><tr><th>Club</th><th>Market Value</th></tr></thead>
          <tbody>{row_html}</tbody>
        </table>
        </body></html>
        """

    def test_parses_million_values(self):
        html = self._make_html([
            ("France", "€920.00m"),
            ("Brazil", "€800.00m"),
            ("Morocco", "€120.00m"),
        ])
        result = _parse_tm_html(html)
        assert len(result) >= 3
        france_val = result.loc[result["team"] == "France", "squad_market_value_eur"].iloc[0]
        assert france_val == pytest.approx(920_000_000.0)

    def test_parses_billion_values(self):
        html = self._make_html([("England", "€1.20bn")] * 1 + [("Germany", "€0.90bn")] * 1
                               + [(f"T{i}", "€100.00m") for i in range(20)])
        result = _parse_tm_html(html)
        eng = result.loc[result["team"] == "England", "squad_market_value_eur"].iloc[0]
        assert eng == pytest.approx(1_200_000_000.0)

    def test_no_table_returns_empty(self):
        result = _parse_tm_html("<html><body><p>Nothing here</p></body></html>")
        assert result.empty

    def test_table_without_value_column_returns_empty(self):
        html = """
        <html><body>
        <table>
          <thead><tr><th>Club</th><th>Players</th></tr></thead>
          <tbody><tr><td>France</td><td>25</td></tr></tbody>
        </table>
        </body></html>
        """
        result = _parse_tm_html(html)
        assert result.empty

    def test_fewer_than_20_rows_returns_empty(self):
        """Sanity check: a table with <20 valid rows (no padding) is rejected."""
        # Build HTML directly without the padding helper.
        html = """
        <html><body>
        <table>
          <thead><tr><th>Club</th><th>Market Value</th></tr></thead>
          <tbody>
            <tr><td>France</td><td>€920.00m</td></tr>
            <tr><td>Brazil</td><td>€800.00m</td></tr>
          </tbody>
        </table>
        </body></html>
        """
        result = _parse_tm_html(html)
        assert result.empty
