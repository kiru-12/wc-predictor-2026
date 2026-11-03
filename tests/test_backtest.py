"""Tests for evaluation.metrics and evaluation.backtest."""

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import brier_score, calibration_table, log_loss
from evaluation.backtest import (
    _outcome_to_onehot,
    backtest_all,
    backtest_tournament,
    get_tournament_matches,
)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_prediction(self):
        """Perfect prediction → Brier score = 0."""
        pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert brier_score(pred, out) == pytest.approx(0.0)

    def test_worst_prediction(self):
        """Completely wrong prediction → Brier score = 2."""
        pred = np.array([[0.0, 0.0, 1.0]])
        out = np.array([[1.0, 0.0, 0.0]])
        # (0-1)^2 + (0-0)^2 + (1-0)^2 = 2
        assert brier_score(pred, out) == pytest.approx(2.0)

    def test_uniform_prediction(self):
        """Uniform prediction [1/3, 1/3, 1/3] → known Brier score."""
        pred = np.array([[1 / 3, 1 / 3, 1 / 3]])
        out = np.array([[1.0, 0.0, 0.0]])
        # (1/3-1)^2 + (1/3-0)^2 + (1/3-0)^2 = (2/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9
        expected = 6.0 / 9.0
        assert brier_score(pred, out) == pytest.approx(expected, abs=1e-10)

    def test_shape_mismatch_raises(self):
        pred = np.array([[0.5, 0.3, 0.2]])
        out = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="Shape mismatch"):
            brier_score(pred, out)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            brier_score(np.array([]).reshape(0, 3), np.array([]).reshape(0, 3))


class TestLogLoss:
    def test_perfect_prediction_near_zero(self):
        """Near-perfect prediction → log loss near 0."""
        pred = np.array([[0.99, 0.005, 0.005]])
        out = np.array([[1.0, 0.0, 0.0]])
        assert log_loss(pred, out) < 0.02

    def test_uniform_prediction(self):
        """Uniform [1/3, 1/3, 1/3] → log(3)."""
        pred = np.array([[1 / 3, 1 / 3, 1 / 3]])
        out = np.array([[1.0, 0.0, 0.0]])
        assert log_loss(pred, out) == pytest.approx(np.log(3), abs=1e-6)

    def test_worse_prediction_higher_loss(self):
        """More confident wrong prediction → higher loss."""
        pred_good = np.array([[0.7, 0.2, 0.1]])
        pred_bad = np.array([[0.1, 0.2, 0.7]])
        out = np.array([[1.0, 0.0, 0.0]])
        assert log_loss(pred_bad, out) > log_loss(pred_good, out)

    def test_handles_zero_prob(self):
        """Should not crash on 0 probabilities (clipped)."""
        pred = np.array([[0.0, 0.0, 1.0]])
        out = np.array([[1.0, 0.0, 0.0]])
        result = log_loss(pred, out)
        assert np.isfinite(result)


class TestCalibrationTable:
    def test_returns_dataframe(self):
        pred = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        out = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        table = calibration_table(pred, out, n_bins=5)
        assert isinstance(table, pd.DataFrame)
        assert set(table.columns) == {
            "bin_lower", "bin_upper", "mean_predicted", "observed_freq", "count"
        }

    def test_count_matches_input(self):
        """Total count across bins should equal n_matches * 3 (flattened)."""
        pred = np.array([[0.5, 0.3, 0.2]] * 10)
        out = np.array([[1.0, 0.0, 0.0]] * 10)
        table = calibration_table(pred, out, n_bins=10)
        assert table["count"].sum() == 30  # 10 matches * 3 classes


# ---------------------------------------------------------------------------
# Backtest tests
# ---------------------------------------------------------------------------


class TestOutcomeToOnehot:
    def test_home_win(self):
        assert list(_outcome_to_onehot(2, 1)) == [1.0, 0.0, 0.0]

    def test_draw(self):
        assert list(_outcome_to_onehot(1, 1)) == [0.0, 1.0, 0.0]

    def test_away_win(self):
        assert list(_outcome_to_onehot(0, 3)) == [0.0, 0.0, 1.0]


class TestGetTournamentMatches:
    @pytest.fixture
    def results(self) -> pd.DataFrame:
        """Synthetic results with WC and non-WC matches."""
        return pd.DataFrame([
            {"date": "2022-11-21", "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup",
             "city": "X", "country": "Qatar", "neutral": True},
            {"date": "2022-11-22", "home_team": "C", "away_team": "D",
             "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup",
             "city": "X", "country": "Qatar", "neutral": True},
            {"date": "2022-03-15", "home_team": "A", "away_team": "C",
             "home_score": 1, "away_score": 0,
             "tournament": "FIFA World Cup qualification",
             "city": "Y", "country": "Z", "neutral": False},
            {"date": "2022-06-01", "home_team": "B", "away_team": "D",
             "home_score": 0, "away_score": 0, "tournament": "Friendly",
             "city": "Y", "country": "Z", "neutral": False},
        ])

    def test_extracts_wc_only(self, results):
        wc = get_tournament_matches(results, 2022)
        assert len(wc) == 2

    def test_excludes_qualifiers(self, results):
        wc = get_tournament_matches(results, 2022)
        assert not wc["tournament"].str.contains("qualifier", case=False).any()

    def test_wrong_year_empty(self, results):
        wc = get_tournament_matches(results, 2018)
        assert len(wc) == 0


class TestBacktestTournament:
    @pytest.fixture
    def synth_results(self) -> pd.DataFrame:
        """Synthetic data with enough history + WC matches to backtest."""
        rng = np.random.default_rng(42)
        rows = []
        # Training data: 200 matches between 4 teams, 2018-2022
        teams = ["Alpha", "Beta", "Gamma", "Delta"]
        for i in range(200):
            home, away = rng.choice(teams, 2, replace=False)
            rows.append({
                "date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(i * 5)),
                "home_team": home,
                "away_team": away,
                "home_score": int(rng.poisson(1.5)),
                "away_score": int(rng.poisson(1.0)),
                "tournament": "Friendly",
                "city": "TestCity",
                "country": "TestLand",
                "neutral": False,
            })
        # WC 2022 matches
        wc_date = pd.Timestamp("2022-11-21")
        for i, (h, a) in enumerate([("Alpha", "Beta"), ("Gamma", "Delta"),
                                     ("Alpha", "Gamma"), ("Beta", "Delta")]):
            rows.append({
                "date": wc_date + pd.Timedelta(days=i),
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(1.5)),
                "away_score": int(rng.poisson(1.0)),
                "tournament": "FIFA World Cup",
                "city": "Doha",
                "country": "Qatar",
                "neutral": True,
            })
        return pd.DataFrame(rows)

    def test_returns_valid_result(self, synth_results):
        result = backtest_tournament(synth_results, 2022, start_year=2018)
        assert result["year"] == 2022
        assert result["n_matches"] == 4
        assert 0 <= result["brier_score"] <= 2
        assert result["log_loss"] > 0

    def test_predictions_dataframe_shape(self, synth_results):
        result = backtest_tournament(synth_results, 2022, start_year=2018)
        df = result["predictions"]
        assert len(df) == 4
        assert "p_home_win" in df.columns
        assert "actual_outcome" in df.columns

    def test_probabilities_sum_to_one(self, synth_results):
        result = backtest_tournament(synth_results, 2022, start_year=2018)
        df = result["predictions"]
        sums = df["p_home_win"] + df["p_draw"] + df["p_away_win"]
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_invalid_year_raises(self, synth_results):
        with pytest.raises(ValueError, match="Unknown WC year"):
            backtest_tournament(synth_results, 2010)


class TestBacktestAll:
    @pytest.fixture
    def synth_results(self) -> pd.DataFrame:
        """Minimal synthetic results with WC 2022 only."""
        rng = np.random.default_rng(99)
        rows = []
        teams = ["A", "B", "C", "D"]
        for i in range(150):
            home, away = rng.choice(teams, 2, replace=False)
            rows.append({
                "date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(i * 7)),
                "home_team": home,
                "away_team": away,
                "home_score": int(rng.poisson(1.3)),
                "away_score": int(rng.poisson(1.0)),
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": False,
            })
        # WC 2022
        for i, (h, a) in enumerate([("A", "B"), ("C", "D")]):
            rows.append({
                "date": pd.Timestamp("2022-11-21") + pd.Timedelta(days=i),
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(1.5)),
                "away_score": int(rng.poisson(1.0)),
                "tournament": "FIFA World Cup",
                "city": "Doha",
                "country": "Qatar",
                "neutral": True,
            })
        return pd.DataFrame(rows)

    def test_returns_summary(self, synth_results):
        summary = backtest_all(synth_results, years=[2022], start_year=2015)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1
        assert "brier_score" in summary.columns

    def test_skips_missing_years(self, synth_results):
        """Years with no WC matches are skipped gracefully."""
        summary = backtest_all(synth_results, years=[2014, 2022], start_year=2015)
        # 2014 should be skipped (no data), 2022 should succeed
        assert len(summary) == 1
        assert summary.iloc[0]["year"] == 2022
