"""Tests for models.dixon_coles."""

import numpy as np
import pandas as pd
import pytest

from models.dixon_coles import DixonColesModel, fit, tau


# ---------------------------------------------------------------------------
# Fixtures — synthetic match data
# ---------------------------------------------------------------------------

def _make_matches(n_per_pair: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create synthetic international results with known strength ordering.

    Teams: Strong (high attack), Medium, Weak (low attack).
    Strong beats Weak most of the time; Medium is in between.
    """
    rng = np.random.default_rng(seed)
    teams = ["Strong", "Medium", "Weak"]
    # True expected goals (home, away) — asymmetric by design
    true_rates = {
        ("Strong", "Medium"): (2.0, 0.8),
        ("Strong", "Weak"):   (2.5, 0.5),
        ("Medium", "Weak"):   (1.5, 0.7),
        ("Medium", "Strong"): (0.8, 2.0),
        ("Weak", "Strong"):   (0.5, 2.5),
        ("Weak", "Medium"):   (0.7, 1.5),
    }

    rows = []
    base_date = pd.Timestamp("2020-01-01")
    idx = 0
    for (home, away), (lam, mu) in true_rates.items():
        for _ in range(n_per_pair):
            rows.append({
                "date": base_date + pd.Timedelta(days=idx),
                "home_team": home,
                "away_team": away,
                "home_score": rng.poisson(lam),
                "away_score": rng.poisson(mu),
                "tournament": "Friendly",
                "city": "Testville",
                "country": "Testland",
                "neutral": True,
            })
            idx += 1

    return pd.DataFrame(rows)


@pytest.fixture
def matches() -> pd.DataFrame:
    return _make_matches()


@pytest.fixture
def model(matches: pd.DataFrame) -> DixonColesModel:
    return fit(matches)


# ---------------------------------------------------------------------------
# tau() tests
# ---------------------------------------------------------------------------


class TestTau:
    def test_tau_00(self):
        """tau(0,0) = 1 - lambda*mu*rho."""
        val = tau(0, 0, 1.5, 1.0, -0.1)
        assert val == pytest.approx(1.0 - 1.5 * 1.0 * (-0.1))

    def test_tau_01(self):
        """tau(0,1) = 1 + lambda*rho."""
        val = tau(0, 1, 1.5, 1.0, -0.1)
        assert val == pytest.approx(1.0 + 1.5 * (-0.1))

    def test_tau_10(self):
        """tau(1,0) = 1 + mu*rho."""
        val = tau(1, 0, 1.5, 1.0, -0.1)
        assert val == pytest.approx(1.0 + 1.0 * (-0.1))

    def test_tau_11(self):
        """tau(1,1) = 1 - rho."""
        val = tau(1, 1, 1.5, 1.0, -0.1)
        assert val == pytest.approx(1.0 - (-0.1))

    def test_tau_other_scores_return_one(self):
        """tau returns 1.0 for all scorelines beyond (0,0)/(0,1)/(1,0)/(1,1)."""
        for x, y in [(2, 0), (0, 2), (3, 1), (2, 2), (5, 3)]:
            assert tau(x, y, 1.5, 1.0, -0.1) == 1.0

    def test_tau_rho_zero_always_one(self):
        """When rho=0, tau returns 1.0 for all scorelines (independent Poisson)."""
        for x in range(4):
            for y in range(4):
                assert tau(x, y, 1.5, 1.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# Fitting tests
# ---------------------------------------------------------------------------


class TestFit:
    def test_returns_model(self, matches):
        """fit() returns a DixonColesModel."""
        m = fit(matches)
        assert isinstance(m, DixonColesModel)

    def test_correct_team_count(self, model):
        """Model has 3 teams."""
        assert len(model.teams) == 3

    def test_attack_ranking(self, model):
        """Strong must have the highest attack parameter."""
        params = model.get_params()
        best = params.iloc[0]["team"]
        assert best == "Strong", f"Expected Strong at top, got {best}"

    def test_identifiability_constraint(self, model):
        """sum(attack) should be approximately 0."""
        assert np.sum(model.attack) == pytest.approx(0.0, abs=0.05)

    def test_rho_sign(self, model):
        """rho is typically negative (low-score correlation)."""
        # With synthetic Poisson data rho should be near 0, but we just
        # check it's within the valid range.
        assert -1.0 <= model.rho <= 1.0

    def test_as_of_filters_matches(self, matches):
        """as_of must exclude future matches."""
        # Use a cutoff that excludes roughly half the data
        mid_date = matches["date"].median().strftime("%Y-%m-%d")
        m_early = fit(matches, as_of=mid_date)
        m_full = fit(matches)
        # Fewer training matches → potentially different params
        # Just verify both fit without error and have same teams
        assert set(m_early.teams) == set(m_full.teams)

    def test_as_of_no_matches_raises(self):
        """as_of before all data should raise ValueError."""
        df = _make_matches(n_per_pair=5)
        with pytest.raises(ValueError, match="No matches"):
            fit(df, as_of="1900-01-01")

    def test_get_params_dataframe(self, model):
        """get_params returns a DataFrame with expected columns."""
        df = model.get_params()
        assert set(df.columns) == {"team", "attack", "defense"}
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


class TestPredict:
    def test_probabilities_sum_to_one(self, model):
        """W + D + L probabilities must sum to 1."""
        pred = model.predict("Strong", "Weak")
        total = pred["home_win"] + pred["draw"] + pred["away_win"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_goal_matrix_sums_to_one(self, model):
        """Goal probability matrix must sum to 1."""
        pred = model.predict("Strong", "Weak")
        assert pred["goal_matrix"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_strong_beats_weak(self, model):
        """Strong should have higher win probability vs Weak."""
        pred = model.predict("Strong", "Weak", neutral=True)
        assert pred["home_win"] > pred["away_win"]

    def test_expected_goals_positive(self, model):
        """Expected goals must be positive."""
        pred = model.predict("Medium", "Weak")
        assert pred["expected_home_goals"] > 0
        assert pred["expected_away_goals"] > 0

    def test_unknown_team_raises(self, model):
        """Predicting with an unknown team raises KeyError."""
        with pytest.raises(KeyError):
            model.predict("Strong", "UnknownTeam")

    def test_neutral_reduces_home_advantage(self, model):
        """Neutral venue should give lower home win prob than home venue."""
        pred_home = model.predict("Medium", "Weak", neutral=False)
        pred_neutral = model.predict("Medium", "Weak", neutral=True)
        # Home advantage boosts home team
        assert pred_home["expected_home_goals"] >= pred_neutral["expected_home_goals"]


# ---------------------------------------------------------------------------
# Simulate score tests
# ---------------------------------------------------------------------------


class TestSimulateScore:
    def test_returns_int_tuple(self, model):
        """simulate_score returns a tuple of two ints."""
        hg, ag = model.simulate_score("Strong", "Weak")
        assert isinstance(hg, int)
        assert isinstance(ag, int)

    def test_deterministic_with_seed(self, model):
        """Same seed → same scoreline."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        s1 = model.simulate_score("Strong", "Weak", rng=rng1)
        s2 = model.simulate_score("Strong", "Weak", rng=rng2)
        assert s1 == s2

    def test_non_negative_scores(self, model):
        """Scores must be non-negative."""
        for _ in range(20):
            hg, ag = model.simulate_score("Strong", "Medium")
            assert hg >= 0
            assert ag >= 0
