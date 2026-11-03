"""Tests for features/player_impact.py.

Uses a small synthetic fixture — no network access required.

Fixture design
--------------
Two national teams: **Alpha** and **Beta** (6 players each).

Alpha has:
  - One dominant star (Striker A: top scorer, UCL club, high rating)
  - Moderate squad depth

Beta has:
  - No single dominant player but good squad depth
  - A Copa win (not relevant here, but included for realism)

The fixture exercises every function including edge-case guards (missing
columns, zero minutes, empty series).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.player_impact import (
    AVG_RATING_STAR_THRESHOLD,
    DEPTH_WEIGHT,
    STAR_DEPENDENCY_THRESHOLD,
    UCL_UEL_MULTIPLIER,
    UCL_UEL_MIN_START_FRACTION,
    build_player_impact,
    compute_squad_score,
    compute_star_dependency,
    compute_star_scores,
    flag_star_players,
    normalise_per90,
)


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def players_df() -> pd.DataFrame:
    """12-player synthetic roster across two national teams."""
    rows = [
        # Alpha squad
        {
            "player": "Striker A",
            "national_team": "Alpha",
            "club": "UCL Club 1",
            "league": "Top League",
            "season": 2022,
            "minutes": 2700,
            "starts": 29,
            "matches": 30,
            "goals": 27,
            "assists": 8,
            "xg": 22.0,
            "xa": 6.0,
            "avg_rating": 8.2,
            "is_ucl_uel_club": True,
        },
        {
            "player": "Midfielder A1",
            "national_team": "Alpha",
            "club": "UCL Club 2",
            "league": "Top League",
            "season": 2022,
            "minutes": 2520,
            "starts": 28,
            "matches": 30,
            "goals": 10,
            "assists": 15,
            "xg": 9.0,
            "xa": 13.0,
            "avg_rating": 7.8,
            "is_ucl_uel_club": True,
        },
        {
            "player": "Defender A1",
            "national_team": "Alpha",
            "club": "Mid Club 1",
            "league": "Top League",
            "season": 2022,
            "minutes": 2880,
            "starts": 32,
            "matches": 33,
            "goals": 2,
            "assists": 3,
            "xg": 1.5,
            "xa": 2.5,
            "avg_rating": 7.0,
            "is_ucl_uel_club": False,
        },
        {
            "player": "Defender A2",
            "national_team": "Alpha",
            "club": "Mid Club 2",
            "league": "Top League",
            "season": 2022,
            "minutes": 2700,
            "starts": 30,
            "matches": 32,
            "goals": 1,
            "assists": 2,
            "xg": 1.0,
            "xa": 1.5,
            "avg_rating": 6.8,
            "is_ucl_uel_club": False,
        },
        {
            "player": "Goalkeeper A",
            "national_team": "Alpha",
            "club": "UCL Club 3",
            "league": "Top League",
            "season": 2022,
            "minutes": 3060,
            "starts": 34,
            "matches": 34,
            "goals": 0,
            "assists": 0,
            "xg": 0.0,
            "xa": 0.0,
            "avg_rating": 7.6,
            "is_ucl_uel_club": True,
        },
        {
            "player": "Sub A1",
            "national_team": "Alpha",
            "club": "Small Club 1",
            "league": "Top League",
            "season": 2022,
            "minutes": 900,
            "starts": 5,
            "matches": 20,
            "goals": 3,
            "assists": 2,
            "xg": 2.5,
            "xa": 1.5,
            "avg_rating": 6.5,
            "is_ucl_uel_club": False,
        },

        # Beta squad — more evenly distributed
        {
            "player": "Striker B1",
            "national_team": "Beta",
            "club": "UCL Club 4",
            "league": "Top League",
            "season": 2022,
            "minutes": 2610,
            "starts": 29,
            "matches": 31,
            "goals": 18,
            "assists": 9,
            "xg": 16.0,
            "xa": 8.0,
            "avg_rating": 7.7,
            "is_ucl_uel_club": True,
        },
        {
            "player": "Striker B2",
            "national_team": "Beta",
            "club": "UCL Club 5",
            "league": "Top League",
            "season": 2022,
            "minutes": 2430,
            "starts": 27,
            "matches": 30,
            "goals": 16,
            "assists": 6,
            "xg": 14.5,
            "xa": 5.5,
            "avg_rating": 7.5,
            "is_ucl_uel_club": True,
        },
        {
            "player": "Midfielder B1",
            "national_team": "Beta",
            "club": "UCL Club 6",
            "league": "Top League",
            "season": 2022,
            "minutes": 2700,
            "starts": 30,
            "matches": 32,
            "goals": 8,
            "assists": 14,
            "xg": 7.5,
            "xa": 12.0,
            "avg_rating": 7.6,
            "is_ucl_uel_club": True,
        },
        {
            "player": "Defender B1",
            "national_team": "Beta",
            "club": "Mid Club 3",
            "league": "Top League",
            "season": 2022,
            "minutes": 2700,
            "starts": 30,
            "matches": 31,
            "goals": 3,
            "assists": 4,
            "xg": 2.5,
            "xa": 3.5,
            "avg_rating": 7.2,
            "is_ucl_uel_club": False,
        },
        {
            "player": "Defender B2",
            "national_team": "Beta",
            "club": "Mid Club 4",
            "league": "Top League",
            "season": 2022,
            "minutes": 2520,
            "starts": 28,
            "matches": 30,
            "goals": 2,
            "assists": 3,
            "xg": 1.8,
            "xa": 2.8,
            "avg_rating": 7.0,
            "is_ucl_uel_club": False,
        },
        {
            "player": "Sub B1",
            "national_team": "Beta",
            "club": "Small Club 2",
            "league": "Top League",
            "season": 2022,
            "minutes": 810,
            "starts": 4,
            "matches": 18,
            "goals": 4,
            "assists": 2,
            "xg": 3.5,
            "xa": 1.8,
            "avg_rating": 6.9,
            "is_ucl_uel_club": False,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# normalise_per90
# ---------------------------------------------------------------------------

class TestNormalisePer90:
    def test_adds_four_columns(self, players_df):
        df = normalise_per90(players_df)
        for col in ("goals90", "assists90", "xg90", "xa90"):
            assert col in df.columns

    def test_correct_value(self, players_df):
        """Striker A: 27 goals in 2700 min → 27/(2700/90) = 0.9 goals90."""
        df = normalise_per90(players_df)
        row = df[df["player"] == "Striker A"].iloc[0]
        assert row["goals90"] == pytest.approx(0.9)

    def test_zero_minutes_produces_nan(self):
        df = pd.DataFrame({"player": ["Ghost"], "minutes": [0],
                           "goals": [5], "assists": [0], "xg": [0], "xa": [0]})
        out = normalise_per90(df)
        assert math.isnan(out["goals90"].iloc[0])

    def test_missing_minutes_all_nan(self):
        df = pd.DataFrame({"player": ["Ghost"], "goals": [5]})
        out = normalise_per90(df)
        for col in ("goals90", "assists90", "xg90", "xa90"):
            assert col in out.columns
            assert math.isnan(out[col].iloc[0])

    def test_missing_goals_col_is_nan(self, players_df):
        df = players_df.drop(columns=["goals"])
        out = normalise_per90(df)
        assert out["goals90"].isna().all()

    def test_does_not_mutate_input(self, players_df):
        orig_cols = list(players_df.columns)
        normalise_per90(players_df)
        assert list(players_df.columns) == orig_cols


# ---------------------------------------------------------------------------
# flag_star_players
# ---------------------------------------------------------------------------

class TestFlagStarPlayers:
    def _prepared(self, players_df):
        return normalise_per90(players_df)

    def test_adds_is_star_column(self, players_df):
        df = flag_star_players(self._prepared(players_df))
        assert "is_star" in df.columns

    def test_striker_a_is_star_top_scorer(self, players_df):
        """Striker A has 27 goals — must be top scorer in league."""
        df = flag_star_players(self._prepared(players_df))
        row = df[df["player"] == "Striker A"].iloc[0]
        assert row["is_star"] == True

    def test_high_rating_criterion(self, players_df):
        """Goalkeeper A has avg_rating 7.6 — must be flagged via rating."""
        df = flag_star_players(self._prepared(players_df))
        row = df[df["player"] == "Goalkeeper A"].iloc[0]
        assert row["is_star"] == True

    def test_ucl_starter_criterion(self, players_df):
        """Midfielder A1: UCL club, starts/matches = 28/30 = 0.93 → star."""
        df = flag_star_players(self._prepared(players_df))
        row = df[df["player"] == "Midfielder A1"].iloc[0]
        assert row["is_star"] == True

    def test_low_start_fraction_not_ucl_star(self, players_df):
        """Sub A1: UCL not a club, starts/matches = 5/20 = 0.25 → not star via UCL."""
        df = flag_star_players(self._prepared(players_df))
        row = df[df["player"] == "Sub A1"].iloc[0]
        # Sub A1 has avg_rating 6.5 and is not top-3 goals/assists and low start frac.
        assert row["is_star"] == False

    def test_ucl_clubs_override(self, players_df):
        """Passing ucl_uel_clubs overrides is_ucl_uel_club column."""
        df = self._prepared(players_df)
        # Force Sub A1's club into UCL set and give them enough starts.
        df.loc[df["player"] == "Sub A1", "starts"] = 14  # 14/20 = 0.70
        out = flag_star_players(df, ucl_uel_clubs={"Small Club 1"})
        row = out[out["player"] == "Sub A1"].iloc[0]
        assert row["is_star"] == True

    def test_missing_avg_rating_does_not_crash(self, players_df):
        df = self._prepared(players_df).drop(columns=["avg_rating"])
        out = flag_star_players(df)
        assert "is_star" in out.columns

    def test_missing_goals_does_not_crash(self, players_df):
        df = self._prepared(players_df).drop(columns=["goals", "goals90"])
        out = flag_star_players(df)
        assert "is_star" in out.columns

    def test_missing_is_ucl_col_does_not_crash(self, players_df):
        df = self._prepared(players_df).drop(columns=["is_ucl_uel_club"])
        out = flag_star_players(df)
        assert "is_star" in out.columns


# ---------------------------------------------------------------------------
# compute_star_scores
# ---------------------------------------------------------------------------

class TestComputeStarScores:
    def _prepared(self, players_df):
        df = normalise_per90(players_df)
        return flag_star_players(df)

    def test_adds_star_score_column(self, players_df):
        df = compute_star_scores(self._prepared(players_df))
        assert "star_score" in df.columns

    def test_ucl_multiplier_applied(self, players_df):
        df = self._prepared(players_df)
        out = compute_star_scores(df)
        # UCL player's score should be 1.3x a non-UCL player with same raw stats.
        # Build two identical players differing only in is_ucl_uel_club.
        twin = pd.DataFrame([
            {"player": "UCL Twin", "minutes": 900, "starts": 10, "matches": 10,
             "goals": 9, "assists": 9, "xg": 9.0, "xa": 9.0,
             "avg_rating": 7.0, "is_ucl_uel_club": True,
             "national_team": "X", "league": "L", "season": 2022},
            {"player": "Non UCL Twin", "minutes": 900, "starts": 10, "matches": 10,
             "goals": 9, "assists": 9, "xg": 9.0, "xa": 9.0,
             "avg_rating": 7.0, "is_ucl_uel_club": False,
             "national_team": "X", "league": "L", "season": 2022},
        ])
        twin = normalise_per90(twin)
        twin = flag_star_players(twin)
        twin = compute_star_scores(twin)
        ucl_score = twin.loc[twin["player"] == "UCL Twin", "star_score"].iloc[0]
        non_ucl_score = twin.loc[twin["player"] == "Non UCL Twin", "star_score"].iloc[0]
        assert ucl_score == pytest.approx(non_ucl_score * UCL_UEL_MULTIPLIER)

    def test_no_negative_scores(self, players_df):
        df = compute_star_scores(self._prepared(players_df))
        assert (df["star_score"] >= 0).all()

    def test_missing_xg_col_does_not_crash(self, players_df):
        df = self._prepared(players_df).drop(columns=["xg", "xg90"], errors="ignore")
        out = compute_star_scores(df)
        assert "star_score" in out.columns

    def test_missing_ucl_col_does_not_crash(self, players_df):
        df = self._prepared(players_df).drop(columns=["is_ucl_uel_club"])
        out = compute_star_scores(df)
        assert "star_score" in out.columns

    def test_higher_scorer_has_higher_score(self, players_df):
        """Striker A (more goals) should outscore Striker B1."""
        df = normalise_per90(players_df)
        df = flag_star_players(df)
        df = compute_star_scores(df)
        score_a = df.loc[df["player"] == "Striker A", "star_score"].iloc[0]
        score_b = df.loc[df["player"] == "Striker B1", "star_score"].iloc[0]
        assert score_a > score_b


# ---------------------------------------------------------------------------
# compute_squad_score
# ---------------------------------------------------------------------------

class TestComputeSquadScore:
    def test_empty_series_returns_nan(self):
        assert math.isnan(compute_squad_score(pd.Series(dtype=float)))

    def test_single_player(self):
        s = pd.Series([5.0])
        result = compute_squad_score(s, n_top=11)
        assert result == pytest.approx(5.0)

    def test_depth_bonus_added(self):
        # 11 top players with score 10, then 1 bench player with score 4.
        scores = pd.Series([10.0] * 11 + [4.0])
        result = compute_squad_score(scores, n_top=11)
        expected = 10.0 + DEPTH_WEIGHT * 4.0
        assert result == pytest.approx(expected)

    def test_no_depth_if_only_11(self):
        scores = pd.Series([10.0] * 11)
        result = compute_squad_score(scores, n_top=11)
        assert result == pytest.approx(10.0)

    def test_correct_top_selection(self):
        """Top-11 mean should use the highest 11 values."""
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_squad_score(scores, n_top=3)
        top3_mean = (5.0 + 4.0 + 3.0) / 3
        depth_mean = (2.0 + 1.0) / 2
        expected = top3_mean + DEPTH_WEIGHT * depth_mean
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_star_dependency
# ---------------------------------------------------------------------------

class TestComputeStarDependency:
    def test_empty_returns_nan(self):
        assert math.isnan(compute_star_dependency(pd.Series(dtype=float)))

    def test_all_zero_returns_nan(self):
        assert math.isnan(compute_star_dependency(pd.Series([0.0, 0.0])))

    def test_single_player_is_one(self):
        assert compute_star_dependency(pd.Series([5.0])) == pytest.approx(1.0)

    def test_equal_players_fraction(self):
        """4 equal players → dependency = 1/4 = 0.25."""
        assert compute_star_dependency(pd.Series([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.25)

    def test_dominant_player_flags_threshold(self):
        """One player with 40% of total → above threshold."""
        scores = pd.Series([4.0, 2.0, 2.0, 2.0])
        dep = compute_star_dependency(scores)
        assert dep > STAR_DEPENDENCY_THRESHOLD

    def test_balanced_squad_below_threshold(self):
        scores = pd.Series([1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        dep = compute_star_dependency(scores)
        assert dep <= STAR_DEPENDENCY_THRESHOLD


# ---------------------------------------------------------------------------
# build_player_impact (integration)
# ---------------------------------------------------------------------------

class TestBuildPlayerImpact:
    def test_output_shape(self, players_df, tmp_path):
        df = build_player_impact(players_df, output_path=tmp_path / "pi.csv")
        assert len(df) == 2
        expected_cols = {
            "national_team", "squad_score", "top_star_score",
            "star_dependency", "high_dependency", "player_impact",
            "n_stars", "n_players",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_csv_written(self, players_df, tmp_path):
        out = tmp_path / "pi.csv"
        build_player_impact(players_df, output_path=out)
        assert out.exists()
        saved = pd.read_csv(out)
        assert len(saved) == 2

    def test_player_impact_finite(self, players_df, tmp_path):
        df = build_player_impact(players_df, output_path=tmp_path / "pi.csv")
        assert df["player_impact"].notna().all()

    def test_player_impact_formula(self, players_df, tmp_path):
        """player_impact = 0.6*squad_score + 0.4*top_star_score."""
        df = build_player_impact(players_df, output_path=tmp_path / "pi.csv")
        for _, row in df.iterrows():
            expected = 0.6 * row["squad_score"] + 0.4 * row["top_star_score"]
            assert row["player_impact"] == pytest.approx(expected)

    def test_high_dependency_alpha_exceeds_beta(self, players_df, tmp_path):
        """Alpha's star_dependency should be higher than Beta's (more skewed to one player)."""
        df = build_player_impact(players_df, output_path=tmp_path / "pi.csv")
        alpha_dep = df.loc[df["national_team"] == "Alpha", "star_dependency"].iloc[0]
        beta_dep = df.loc[df["national_team"] == "Beta", "star_dependency"].iloc[0]
        assert alpha_dep > beta_dep

    def test_n_players_correct(self, players_df, tmp_path):
        df = build_player_impact(players_df, output_path=tmp_path / "pi.csv")
        for _, row in df.iterrows():
            assert row["n_players"] == 6

    def test_missing_national_team_returns_empty(self, players_df, tmp_path):
        bad = players_df.drop(columns=["national_team"])
        df = build_player_impact(bad, output_path=tmp_path / "pi.csv")
        assert df.empty

    def test_default_output_path(self, players_df, tmp_path, monkeypatch):
        import features.player_impact as pi
        monkeypatch.setattr(pi, "PROCESSED_DIR", tmp_path)
        build_player_impact(players_df)
        assert (tmp_path / "player_impact.csv").exists()

    def test_ucl_clubs_override_propagated(self, players_df, tmp_path):
        """Passing ucl_uel_clubs should change scores for newly added clubs."""
        df_no_ucl = build_player_impact(
            players_df, ucl_uel_clubs=set(), output_path=tmp_path / "pi_no_ucl.csv"
        )
        df_all_ucl = build_player_impact(
            players_df,
            ucl_uel_clubs={r["club"] for r in players_df.to_dict("records")},
            output_path=tmp_path / "pi_all_ucl.csv",
        )
        # All UCL multipliers should give higher or equal scores.
        for team in ("Alpha", "Beta"):
            score_no = df_no_ucl.loc[df_no_ucl["national_team"] == team, "player_impact"].iloc[0]
            score_all = df_all_ucl.loc[df_all_ucl["national_team"] == team, "player_impact"].iloc[0]
            assert score_all >= score_no - 1e-9
