"""Tests for features/team_baseline.py.

Uses a small deterministic synthetic fixture — no network access required.

Fixture design
--------------
Three teams: **Alpha**, **Beta**, **Gamma**.

Match history (all before the 2022-11-01 cutoff):

  - Alpha dominates: wins most matches including WC qualifiers.
  - Beta draws with peers; won a Copa América in 2021.
  - Gamma loses several in a row (non-zero unbeaten streak must be 0).

Tournament variety covers every category so form-weight logic is exercised.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from features.team_baseline import (
    TOURNAMENT_FORM_WEIGHTS,
    classify_tournament,
    compute_national_elo,
    compute_recent_form,
    compute_unbeaten_streak,
    compute_wc_qualifier_gd,
    flag_continental_winner,
    build_team_baseline,
    _to_team_centric,
)

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _row(
    date: str,
    home: str,
    away: str,
    hs: int,
    as_: int,
    tournament: str = "Friendly",
) -> dict:
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "tournament": tournament,
    }


@pytest.fixture()
def results() -> pd.DataFrame:
    """Small deterministic synthetic results fixture."""
    rows = [
        # --- WC qualifier cycle 2018-2021 ---
        _row("2018-09-01", "Alpha", "Beta",  3, 0, "FIFA World Cup qualification"),
        _row("2018-09-05", "Alpha", "Gamma", 2, 1, "FIFA World Cup qualification"),
        _row("2018-10-10", "Beta",  "Gamma", 1, 0, "FIFA World Cup qualification"),
        _row("2019-03-01", "Alpha", "Beta",  1, 0, "FIFA World Cup qualification"),
        _row("2019-06-01", "Gamma", "Alpha", 0, 2, "FIFA World Cup qualification"),

        # --- Copa América 2021 (Beta wins) ---
        _row("2021-06-13", "Beta",  "Gamma", 2, 0, "Copa América"),
        _row("2021-06-17", "Alpha", "Gamma", 1, 0, "Copa América"),
        _row("2021-07-10", "Alpha", "Beta",  0, 1, "Copa América"),  # Beta wins final

        # --- UEFA Nations League ---
        _row("2020-09-03", "Alpha", "Beta",  2, 2, "UEFA Nations League"),
        _row("2020-09-06", "Gamma", "Alpha", 0, 1, "UEFA Nations League"),
        _row("2020-10-07", "Beta",  "Gamma", 3, 1, "UEFA Nations League"),

        # --- Friendlies ---
        _row("2022-03-29", "Alpha", "Gamma", 4, 0, "Friendly"),
        _row("2022-06-01", "Beta",  "Alpha", 1, 2, "Friendly"),
        _row("2022-07-01", "Gamma", "Beta",  0, 2, "Friendly"),
        _row("2022-09-01", "Gamma", "Alpha", 1, 3, "Friendly"),

        # --- Post-cutoff (must be ignored) ---
        _row("2022-11-20", "Alpha", "Gamma", 5, 0, "FIFA World Cup"),
    ]
    return pd.DataFrame(rows)


AS_OF = pd.Timestamp("2022-11-01")


# ---------------------------------------------------------------------------
# classify_tournament
# ---------------------------------------------------------------------------


class TestClassifyTournament:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("FIFA World Cup qualification", "wc_qualifier"),
            ("UEFA Euro qualification", "continental_cup"),  # qualifier for continental cup
            ("Copa América", "continental_cup"),
            ("UEFA Euro", "continental_cup"),
            ("African Cup of Nations", "continental_cup"),
            ("AFC Asian Cup", "continental_cup"),
            ("Gold Cup", "continental_cup"),
            ("UEFA Nations League", "nations_league"),
            ("CONCACAF Nations League", "nations_league"),
            ("Friendly", "friendly"),
            ("King's Cup", "other"),
            ("CECAFA Cup", "other"),
        ],
    )
    def test_classification(self, name: str, expected: str):
        assert classify_tournament(name) == expected


# ---------------------------------------------------------------------------
# _to_team_centric
# ---------------------------------------------------------------------------


class TestToTeamCentric:
    def test_row_count_doubles(self, results):
        """Each match becomes two rows (one per team)."""
        tc = _to_team_centric(results)
        # Exclude the post-cutoff match from the source to make counting clear.
        pre = results[results["date"] < AS_OF]
        tc_pre = _to_team_centric(pre)
        assert len(tc_pre) == len(pre) * 2

    def test_goal_diff_sign(self, results):
        """Home winner's goal_diff must be positive in their row."""
        tc = _to_team_centric(results)
        alpha_wc = tc[
            (tc["team"] == "Alpha")
            & (tc["tournament"] == "FIFA World Cup qualification")
            & (tc["date"] == pd.Timestamp("2018-09-01"))
        ]
        assert float(alpha_wc["goal_diff"].iloc[0]) == pytest.approx(3.0)

    def test_points_win(self, results):
        tc = _to_team_centric(results)
        row = tc[(tc["team"] == "Alpha") & (tc["date"] == pd.Timestamp("2018-09-01"))]
        assert float(row["points"].iloc[0]) == pytest.approx(1.0)

    def test_points_loss(self, results):
        tc = _to_team_centric(results)
        row = tc[(tc["team"] == "Beta") & (tc["date"] == pd.Timestamp("2018-09-01"))]
        assert float(row["points"].iloc[0]) == pytest.approx(0.0)

    def test_points_draw(self, results):
        tc = _to_team_centric(results)
        row = tc[(tc["team"] == "Alpha") & (tc["date"] == pd.Timestamp("2020-09-03"))]
        assert float(row["points"].iloc[0]) == pytest.approx(0.5)

    def test_missing_column_returns_empty(self):
        bad = pd.DataFrame({"date": ["2022-01-01"], "home_team": ["A"]})
        tc = _to_team_centric(bad)
        assert tc.empty


# ---------------------------------------------------------------------------
# compute_national_elo
# ---------------------------------------------------------------------------


class TestComputeNationalElo:
    def test_returns_series(self, results):
        elo = compute_national_elo(results, as_of=AS_OF)
        assert isinstance(elo, pd.Series)

    def test_all_teams_present(self, results):
        elo = compute_national_elo(results, as_of=AS_OF)
        for t in ("Alpha", "Beta", "Gamma"):
            assert t in elo.index

    def test_alpha_highest_elo(self, results):
        """Alpha wins most matches so should have the highest ELO."""
        elo = compute_national_elo(results, as_of=AS_OF)
        assert elo["Alpha"] > elo["Beta"]
        assert elo["Alpha"] > elo["Gamma"]

    def test_post_cutoff_excluded(self, results):
        """Adding a post-cutoff win for Gamma must not raise Gamma's rating."""
        elo_before = compute_national_elo(results, as_of=AS_OF)
        extra = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2023-01-01"),
                    "home_team": "Gamma",
                    "away_team": "Alpha",
                    "home_score": 5,
                    "away_score": 0,
                    "tournament": "Friendly",
                }
            ]
        )
        elo_after = compute_national_elo(
            pd.concat([results, extra], ignore_index=True), as_of=AS_OF
        )
        assert math.isclose(elo_before["Gamma"], elo_after["Gamma"], rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_recent_form
# ---------------------------------------------------------------------------


class TestComputeRecentForm:
    def test_returns_float(self, results):
        tc = _to_team_centric(results)
        score = compute_recent_form(tc, "Alpha", as_of=AS_OF)
        assert isinstance(score, float)
        assert not math.isnan(score)

    def test_alpha_better_form_than_gamma(self, results):
        tc = _to_team_centric(results)
        assert compute_recent_form(tc, "Alpha", as_of=AS_OF) > \
               compute_recent_form(tc, "Gamma", as_of=AS_OF)

    def test_unknown_team_returns_nan(self, results):
        tc = _to_team_centric(results)
        assert math.isnan(compute_recent_form(tc, "Atlantis", as_of=AS_OF))

    def test_form_bounded_0_1(self, results):
        tc = _to_team_centric(results)
        for team in ("Alpha", "Beta", "Gamma"):
            f = compute_recent_form(tc, team, as_of=AS_OF)
            assert 0.0 <= f <= 1.0

    def test_post_cutoff_not_counted(self, results):
        """Matches on or after as_of must not affect the form score."""
        tc = _to_team_centric(results)
        form_pre = compute_recent_form(tc, "Alpha", as_of=AS_OF, n=5)
        # The post-cutoff row (2022-11-20) is already in `results` fixture.
        tc_full = _to_team_centric(results)
        form_full = compute_recent_form(tc_full, "Alpha", as_of=AS_OF, n=5)
        assert math.isclose(form_pre, form_full, rel_tol=1e-9)

    def test_friendly_weight_less_than_wc_qualifier(self, results):
        """Replacing a win in a friendly with a win in a WC qualifier must increase form."""
        tc = _to_team_centric(results)
        base = compute_recent_form(tc, "Alpha", as_of=AS_OF)
        # Replace all friendlies with WC qualifiers for Alpha.
        tc_upg = tc.copy()
        mask = (tc_upg["team"] == "Alpha") & (tc_upg["tournament"] == "Friendly")
        tc_upg.loc[mask, "tournament"] = "FIFA World Cup qualification"
        tc_upg.loc[mask, "category"] = "wc_qualifier"
        upgraded = compute_recent_form(tc_upg, "Alpha", as_of=AS_OF)
        # Upgrading all-win friendlies to WC qualifiers must not decrease form.
        assert upgraded >= base - 1e-9


# ---------------------------------------------------------------------------
# compute_unbeaten_streak
# ---------------------------------------------------------------------------


class TestComputeUnbeatenStreak:
    def test_gamma_streak_zero(self, results):
        """Gamma's last match before cutoff is a loss — streak should be 0."""
        tc = _to_team_centric(results)
        assert compute_unbeaten_streak(tc, "Gamma", as_of=AS_OF) == 0

    def test_alpha_positive_streak(self, results):
        """Alpha's recent run is all wins — streak must be >= 1."""
        tc = _to_team_centric(results)
        assert compute_unbeaten_streak(tc, "Alpha", as_of=AS_OF) >= 1

    def test_unknown_team_returns_zero(self, results):
        tc = _to_team_centric(results)
        assert compute_unbeaten_streak(tc, "Atlantis", as_of=AS_OF) == 0

    def test_returns_int(self, results):
        tc = _to_team_centric(results)
        val = compute_unbeaten_streak(tc, "Alpha", as_of=AS_OF)
        assert isinstance(val, int)


# ---------------------------------------------------------------------------
# compute_wc_qualifier_gd
# ---------------------------------------------------------------------------


class TestComputeWcQualifierGd:
    def test_alpha_positive_gd(self, results):
        tc = _to_team_centric(results)
        gd = compute_wc_qualifier_gd(tc, "Alpha", 2018, 2021)
        # Alpha: +3, +1, +1, +2 in qualifiers = +7
        assert gd == pytest.approx(7.0)

    def test_gamma_negative_gd(self, results):
        tc = _to_team_centric(results)
        gd = compute_wc_qualifier_gd(tc, "Gamma", 2018, 2021)
        assert gd < 0.0

    def test_no_qualifiers_returns_zero(self, results):
        tc = _to_team_centric(results)
        gd = compute_wc_qualifier_gd(tc, "Alpha", 2030, 2033)
        assert gd == pytest.approx(0.0)

    def test_outside_cycle_excluded(self, results):
        """Qualifiers outside the cycle window must not be counted."""
        tc = _to_team_centric(results)
        gd_tight = compute_wc_qualifier_gd(tc, "Alpha", 2019, 2019)
        gd_full = compute_wc_qualifier_gd(tc, "Alpha", 2018, 2021)
        assert gd_tight < gd_full


# ---------------------------------------------------------------------------
# flag_continental_winner
# ---------------------------------------------------------------------------


class TestFlagContinentalWinner:
    def test_beta_wins_copa(self, results):
        """Beta won the Copa América 2021 final."""
        assert flag_continental_winner(results, "Beta", 2018, 2022) == 1

    def test_alpha_did_not_win(self, results):
        """Alpha lost the Copa América 2021 final."""
        assert flag_continental_winner(results, "Alpha", 2018, 2022) == 0

    def test_gamma_did_not_win(self, results):
        assert flag_continental_winner(results, "Gamma", 2018, 2022) == 0

    def test_no_continental_in_range(self, results):
        assert flag_continental_winner(results, "Beta", 2030, 2033) == 0

    def test_missing_columns_returns_zero(self):
        bad = pd.DataFrame({"date": ["2022-01-01"]})
        assert flag_continental_winner(bad, "Beta", 2018, 2022) == 0


# ---------------------------------------------------------------------------
# build_team_baseline (integration)
# ---------------------------------------------------------------------------


class TestBuildTeamBaseline:
    def test_output_shape(self, results, tmp_path):
        teams = ["Alpha", "Beta", "Gamma"]
        df = build_team_baseline(
            results, teams, as_of="2022-11-01",
            wc_cycle_start=2018, wc_cycle_end=2021,
            output_path=tmp_path / "baseline.csv",
        )
        assert len(df) == 3
        expected_cols = {
            "team", "elo_national", "form_score", "unbeaten_streak",
            "wc_qualifier_gd", "continental_winner", "baseline_score",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_csv_written(self, results, tmp_path):
        out = tmp_path / "baseline.csv"
        build_team_baseline(
            results, ["Alpha", "Beta"], as_of="2022-11-01",
            wc_cycle_start=2018, wc_cycle_end=2021,
            output_path=out,
        )
        assert out.exists()
        saved = pd.read_csv(out)
        assert len(saved) == 2

    def test_alpha_highest_baseline(self, results, tmp_path):
        """Alpha is the strongest team in the fixture so must rank first."""
        teams = ["Alpha", "Beta", "Gamma"]
        df = build_team_baseline(
            results, teams, as_of="2022-11-01",
            wc_cycle_start=2018, wc_cycle_end=2021,
            output_path=tmp_path / "baseline.csv",
        )
        best = df.sort_values("baseline_score", ascending=False).iloc[0]["team"]
        assert best == "Alpha"

    def test_baseline_score_no_nan(self, results, tmp_path):
        """baseline_score must be finite for every team in the fixture."""
        teams = ["Alpha", "Beta", "Gamma"]
        df = build_team_baseline(
            results, teams, as_of="2022-11-01",
            wc_cycle_start=2018, wc_cycle_end=2021,
            output_path=tmp_path / "baseline.csv",
        )
        assert df["baseline_score"].notna().all()

    def test_default_output_path_created(self, results, tmp_path, monkeypatch):
        """Default output path (PROCESSED_DIR) is created and written."""
        import features.team_baseline as tb
        monkeypatch.setattr(tb, "PROCESSED_DIR", tmp_path)
        build_team_baseline(
            results, ["Alpha"], as_of="2022-11-01",
            wc_cycle_start=2018, wc_cycle_end=2021,
        )
        assert (tmp_path / "team_baseline.csv").exists()
