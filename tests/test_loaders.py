"""Smoke tests for data/loaders.py.

All network calls are mocked — these tests never hit the real internet.
They verify cache-path logic and that the right functions are called
(or skipped) depending on cache presence.
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS = pd.DataFrame(
    {
        "date": pd.to_datetime(["2018-06-14", "2022-11-20", "1990-06-08"]),
        "home_team": ["Russia", "Qatar", "Italy"],
        "away_team": ["Saudi Arabia", "Ecuador", "USA"],
        "home_score": [5, 0, 1],
        "away_score": [0, 2, 1],
        "tournament": ["FIFA World Cup"] * 3,
        "city": ["Moscow", "Lusail", "Rome"],
        "country": ["Russia", "Qatar", "Italy"],
        "neutral": [False, False, True],
    }
)

_SAMPLE_ELO = pd.DataFrame({"Club": ["Bayern Munich", "Man City"], "Elo": [2000, 1990]})

_SAMPLE_SHOOTING = pd.DataFrame(
    {"xg": [1.5, 2.0], "xga": [0.8, 1.1]},
    index=pd.MultiIndex.from_tuples(
        [
            ("ENG-Premier League", 2022, "Arsenal", 1),
            ("ENG-Premier League", 2022, "Liverpool", 1),
        ],
        names=["league", "season", "team", "game"],
    ),
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _patch_raw_dir(monkeypatch, tmp_path: Path):
    """Point loaders.RAW_DIR at a temporary directory for isolation."""
    import data.loaders as loaders  # import here so monkeypatch takes effect

    monkeypatch.setattr(loaders, "RAW_DIR", tmp_path)
    return loaders


# ---------------------------------------------------------------------------
# load_international_results
# ---------------------------------------------------------------------------


class TestLoadInternationalResults:
    def test_cache_hit_skips_network(self, tmp_path, monkeypatch):
        """When results.csv exists, pd.read_csv is called with the local path only."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)
        cache = tmp_path / "results.csv"
        _SAMPLE_RESULTS.to_csv(cache, index=False)

        with patch("data.loaders.pd.read_csv", wraps=pd.read_csv) as mock_csv:
            df = loaders.load_international_results(start_year=1990, end_year=2022)

        # Only one call, and it must be the local file path (not a URL).
        assert mock_csv.call_count == 1
        first_arg = str(mock_csv.call_args[0][0])
        assert first_arg == str(cache)

        assert len(df) == 3
        assert set(df.columns) >= {"home_team", "away_team", "home_score"}

    def test_year_filter_lower_bound(self, tmp_path, monkeypatch):
        """Only rows >= start_year are returned."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)
        (tmp_path / "results.csv").write_text(
            _SAMPLE_RESULTS.to_csv(index=False)
        )
        df = loaders.load_international_results(start_year=2018, end_year=2022)
        assert all(pd.to_datetime(df["date"]).dt.year >= 2018)

    def test_year_filter_exact_year(self, tmp_path, monkeypatch):
        """Filtering to a single year returns only that year's rows."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)
        (tmp_path / "results.csv").write_text(
            _SAMPLE_RESULTS.to_csv(index=False)
        )
        df = loaders.load_international_results(start_year=2018, end_year=2018)
        assert len(df) == 1
        assert df.iloc[0]["home_team"] == "Russia"

    def test_cache_miss_calls_url(self, tmp_path, monkeypatch):
        """No cache → pd.read_csv is called with the GitHub URL."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)

        with patch("data.loaders.pd.read_csv", return_value=_SAMPLE_RESULTS) as mock_csv:
            loaders.load_international_results()

        first_arg = str(mock_csv.call_args[0][0])
        assert first_arg.startswith("https://")

    def test_cache_miss_writes_csv(self, tmp_path, monkeypatch):
        """After a download the data is written to results.csv."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)

        with patch("data.loaders.pd.read_csv", return_value=_SAMPLE_RESULTS):
            loaders.load_international_results()

        assert (tmp_path / "results.csv").exists()


# ---------------------------------------------------------------------------
# load_elo_ratings
# ---------------------------------------------------------------------------


class TestLoadEloRatings:
    def test_cache_hit_skips_soccerdata(self, tmp_path, monkeypatch):
        """When the CSV cache exists, soccerdata.ClubElo is never instantiated."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)
        elo_dir = tmp_path / "clubelo"
        elo_dir.mkdir()
        (elo_dir / "clubelo_latest.csv").write_text(_SAMPLE_ELO.to_csv(index=False))

        with patch("data.loaders.sd") as mock_sd:
            df = loaders.load_elo_ratings()

        mock_sd.ClubElo.assert_not_called()
        assert len(df) == 2

    def test_cache_miss_calls_clubelo(self, tmp_path, monkeypatch):
        """No cache → ClubElo is instantiated and read_by_date is called."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)

        mock_reader = MagicMock()
        mock_reader.read_by_date.return_value = _SAMPLE_ELO

        with patch("data.loaders.sd.ClubElo", return_value=mock_reader):
            df = loaders.load_elo_ratings(date="2022-11-01")

        mock_reader.read_by_date.assert_called_once_with("2022-11-01")
        assert len(df) == 2

    def test_cache_miss_writes_csv(self, tmp_path, monkeypatch):
        """After fetching, a CSV is written at the expected path."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)

        mock_reader = MagicMock()
        mock_reader.read_by_date.return_value = _SAMPLE_ELO

        with patch("data.loaders.sd.ClubElo", return_value=mock_reader):
            loaders.load_elo_ratings(date="2022-11-01")

        assert (tmp_path / "clubelo" / "clubelo_2022-11-01.csv").exists()

    def test_date_none_uses_latest_slug(self, tmp_path, monkeypatch):
        """date=None produces the cache file named 'clubelo_latest.csv'."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)

        mock_reader = MagicMock()
        mock_reader.read_by_date.return_value = _SAMPLE_ELO

        with patch("data.loaders.sd.ClubElo", return_value=mock_reader):
            loaders.load_elo_ratings(date=None)

        assert (tmp_path / "clubelo" / "clubelo_latest.csv").exists()


# ---------------------------------------------------------------------------
# load_fbref_shooting
# ---------------------------------------------------------------------------


class TestLoadFbrefShooting:
    _leagues = ["ENG-Premier League"]
    _seasons = [2022]

    def _write_cache(self, tmp_path: Path) -> Path:
        """Pickle sample data at the expected cache location and return the path."""
        import data.loaders as loaders

        fbref_dir = tmp_path / "fbref"
        fbref_dir.mkdir(parents=True, exist_ok=True)
        path = loaders._fbref_cache_path(self._leagues, self._seasons)
        # _fbref_cache_path uses the module's RAW_DIR — we must build it manually.
        import hashlib

        key = ",".join(sorted(self._leagues)) + "|" + ",".join(
            str(s) for s in sorted(self._seasons)
        )
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        cache_path = fbref_dir / f"shooting_{digest}.pkl"
        with open(cache_path, "wb") as fh:
            pickle.dump(_SAMPLE_SHOOTING, fh)
        return cache_path

    def test_cache_hit_skips_fbref(self, tmp_path, monkeypatch):
        """When the pickle cache exists, FBref is never instantiated."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)
        self._write_cache(tmp_path)

        with patch("data.loaders.sd.FBref") as mock_cls:
            df = loaders.load_fbref_shooting(self._leagues, self._seasons)

        mock_cls.assert_not_called()
        assert len(df) == 2

    def test_cache_miss_calls_fbref(self, tmp_path, monkeypatch):
        """No cache → FBref is instantiated and read_team_match_stats is called."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)

        mock_reader = MagicMock()
        mock_reader.read_team_match_stats.return_value = _SAMPLE_SHOOTING

        with patch("data.loaders.sd.FBref", return_value=mock_reader):
            df = loaders.load_fbref_shooting(self._leagues, self._seasons)

        mock_reader.read_team_match_stats.assert_called_once_with(stat_type="shooting")
        assert len(df) == 2

    def test_cache_miss_writes_pickle(self, tmp_path, monkeypatch):
        """After fetching, a .pkl file is written inside data/raw/fbref/."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)

        mock_reader = MagicMock()
        mock_reader.read_team_match_stats.return_value = _SAMPLE_SHOOTING

        with patch("data.loaders.sd.FBref", return_value=mock_reader):
            loaders.load_fbref_shooting(self._leagues, self._seasons)

        fbref_dir = tmp_path / "fbref"
        pkl_files = list(fbref_dir.glob("shooting_*.pkl"))
        assert len(pkl_files) == 1

    def test_different_inputs_produce_different_cache_keys(self, tmp_path, monkeypatch):
        """Two distinct league/season combinations must not share a cache file."""
        loaders = _patch_raw_dir(monkeypatch, tmp_path)

        path_a = loaders._fbref_cache_path(["ENG-Premier League"], [2021])
        path_b = loaders._fbref_cache_path(["ESP-La Liga"], [2021])
        assert path_a != path_b
