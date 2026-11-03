"""Tests for simulation.tournament and simulation.monte_carlo."""

import numpy as np
import pandas as pd
import pytest

from models.dixon_coles import DixonColesModel, fit
from simulation.tournament import (
    build_knockout_bracket,
    simulate_group_stage,
    simulate_knockout,
    simulate_tournament,
)
from simulation.monte_carlo import run_simulations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_groups(n_groups: int = 4) -> pd.DataFrame:
    """Create synthetic groups for testing (4 teams per group)."""
    teams = []
    for g_idx in range(n_groups):
        group_letter = chr(ord("A") + g_idx)
        for t_idx in range(4):
            teams.append({
                "team": f"Team_{group_letter}{t_idx+1}",
                "group": group_letter,
            })
    return pd.DataFrame(teams)


def _make_model(groups_df: pd.DataFrame, seed: int = 42) -> DixonColesModel:
    """Create a fitted model from synthetic match data for all teams in groups."""
    rng = np.random.default_rng(seed)
    teams = groups_df["team"].tolist()
    rows = []
    base_date = pd.Timestamp("2020-01-01")
    idx = 0
    # Generate enough matches between all pairs
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i == j:
                continue
            for _ in range(3):
                rows.append({
                    "date": base_date + pd.Timedelta(days=idx),
                    "home_team": home,
                    "away_team": away,
                    "home_score": int(rng.poisson(1.3)),
                    "away_score": int(rng.poisson(1.0)),
                    "tournament": "Friendly",
                    "city": "Test",
                    "country": "Test",
                    "neutral": True,
                })
                idx += 1

    matches = pd.DataFrame(rows)
    return fit(matches)


@pytest.fixture
def groups_4() -> pd.DataFrame:
    """4 groups of 4 teams = 16 teams."""
    return _make_groups(4)


@pytest.fixture
def model_16(groups_4: pd.DataFrame) -> DixonColesModel:
    """Fitted model for the 16-team fixture."""
    return _make_model(groups_4)


# ---------------------------------------------------------------------------
# Group stage tests
# ---------------------------------------------------------------------------


class TestSimulateGroupStage:
    def test_all_teams_present(self, model_16, groups_4):
        standings = simulate_group_stage(model_16, groups_4, rng=np.random.default_rng(1))
        assert set(standings["team"]) == set(groups_4["team"])

    def test_correct_row_count(self, model_16, groups_4):
        standings = simulate_group_stage(model_16, groups_4, rng=np.random.default_rng(1))
        assert len(standings) == 16  # 4 groups * 4 teams

    def test_group_ranks_1_to_4(self, model_16, groups_4):
        standings = simulate_group_stage(model_16, groups_4, rng=np.random.default_rng(1))
        for _, grp in standings.groupby("group"):
            ranks = sorted(grp["group_rank"].tolist())
            assert ranks == [1, 2, 3, 4]

    def test_points_non_negative(self, model_16, groups_4):
        standings = simulate_group_stage(model_16, groups_4, rng=np.random.default_rng(1))
        assert (standings["pts"] >= 0).all()

    def test_max_points_is_9(self, model_16, groups_4):
        """Max possible points in a group of 4 is 9 (3 wins)."""
        standings = simulate_group_stage(model_16, groups_4, rng=np.random.default_rng(1))
        assert (standings["pts"] <= 9).all()

    def test_deterministic_with_seed(self, model_16, groups_4):
        s1 = simulate_group_stage(model_16, groups_4, rng=np.random.default_rng(123))
        s2 = simulate_group_stage(model_16, groups_4, rng=np.random.default_rng(123))
        pd.testing.assert_frame_equal(s1, s2)


# ---------------------------------------------------------------------------
# Knockout bracket tests
# ---------------------------------------------------------------------------


class TestBuildKnockoutBracket:
    def test_16_matchups_from_4_groups(self, model_16, groups_4):
        """4 groups: top-2 (8) + best 3rd (4 of 4, so all 3rds) = 12?
        Actually with 4 groups we don't have 12 groups.  This tests the
        general logic with what we have."""
        standings = simulate_group_stage(model_16, groups_4, rng=np.random.default_rng(1))
        # For 4 groups: 4 firsts + 4 seconds + up to 4 thirds
        # build_knockout_bracket expects 12 groups; test with real 12-group data
        # For now just verify it doesn't crash and produces matchups
        # Skip this test for 4-group setup since bracket logic is for 12 groups
        pytest.skip("Bracket logic designed for 12 groups; tested in integration")


# ---------------------------------------------------------------------------
# Full tournament simulation tests
# ---------------------------------------------------------------------------


class TestSimulateTournament:
    @pytest.fixture
    def groups_12(self) -> pd.DataFrame:
        """12 groups of 4 = 48 teams (realistic 2026 format)."""
        return _make_groups(12)

    @pytest.fixture
    def model_48(self, groups_12: pd.DataFrame) -> DixonColesModel:
        """Fitted model for 48 teams — use fewer matches for speed."""
        rng = np.random.default_rng(7)
        teams = groups_12["team"].tolist()
        rows = []
        base_date = pd.Timestamp("2020-01-01")
        idx = 0
        # 2 matches per directional pair would be 48*47*2=4512 — too slow
        # Instead: each team plays 10 random opponents
        for team in teams:
            opponents = rng.choice([t for t in teams if t != team], 10, replace=False)
            for opp in opponents:
                rows.append({
                    "date": base_date + pd.Timedelta(days=idx),
                    "home_team": team,
                    "away_team": opp,
                    "home_score": int(rng.poisson(1.3)),
                    "away_score": int(rng.poisson(1.0)),
                    "tournament": "Friendly",
                    "city": "Test",
                    "country": "Test",
                    "neutral": True,
                })
                idx += 1
        return fit(pd.DataFrame(rows))

    def test_produces_winner(self, model_48, groups_12):
        result = simulate_tournament(model_48, groups_12, rng=np.random.default_rng(42))
        assert result["winner"] in groups_12["team"].tolist()

    def test_32_knockout_teams(self, model_48, groups_12):
        result = simulate_tournament(model_48, groups_12, rng=np.random.default_rng(42))
        assert len(result["knockout_teams"]) == 32

    def test_knockout_teams_are_subset(self, model_48, groups_12):
        result = simulate_tournament(model_48, groups_12, rng=np.random.default_rng(42))
        assert set(result["knockout_teams"]).issubset(set(groups_12["team"]))

    def test_deterministic(self, model_48, groups_12):
        r1 = simulate_tournament(model_48, groups_12, rng=np.random.default_rng(99))
        r2 = simulate_tournament(model_48, groups_12, rng=np.random.default_rng(99))
        assert r1["winner"] == r2["winner"]


# ---------------------------------------------------------------------------
# Monte Carlo tests
# ---------------------------------------------------------------------------


class TestRunSimulations:
    @pytest.fixture
    def groups_12(self) -> pd.DataFrame:
        return _make_groups(12)

    @pytest.fixture
    def model_48(self, groups_12: pd.DataFrame) -> DixonColesModel:
        rng = np.random.default_rng(7)
        teams = groups_12["team"].tolist()
        rows = []
        base_date = pd.Timestamp("2020-01-01")
        idx = 0
        for team in teams:
            opponents = rng.choice([t for t in teams if t != team], 10, replace=False)
            for opp in opponents:
                rows.append({
                    "date": base_date + pd.Timedelta(days=idx),
                    "home_team": team,
                    "away_team": opp,
                    "home_score": int(rng.poisson(1.3)),
                    "away_score": int(rng.poisson(1.0)),
                    "tournament": "Friendly",
                    "city": "Test",
                    "country": "Test",
                    "neutral": True,
                })
                idx += 1
        return fit(pd.DataFrame(rows))

    def test_win_pct_sums_near_100(self, model_48, groups_12):
        """Sum of all win_pct should be ~100%."""
        results = run_simulations(model_48, groups_12, n=100, seed=42, show_progress=False)
        total = results["win_pct"].sum()
        assert abs(total - 100.0) < 1e-6

    def test_all_teams_present(self, model_48, groups_12):
        results = run_simulations(model_48, groups_12, n=50, seed=42, show_progress=False)
        assert set(results["team"]) == set(groups_12["team"])

    def test_deterministic(self, model_48, groups_12):
        r1 = run_simulations(model_48, groups_12, n=50, seed=123, show_progress=False)
        r2 = run_simulations(model_48, groups_12, n=50, seed=123, show_progress=False)
        pd.testing.assert_frame_equal(r1, r2)

    def test_sorted_by_win_pct(self, model_48, groups_12):
        results = run_simulations(model_48, groups_12, n=50, seed=42, show_progress=False)
        assert results["win_pct"].is_monotonic_decreasing
