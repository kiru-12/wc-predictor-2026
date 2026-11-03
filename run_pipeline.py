"""End-to-end pipeline for the WC 2026 Predictor.

Usage
-----
    python run_pipeline.py           # Full run (100k sims)
    python run_pipeline.py --quick   # Quick run (1k sims, ~2 min)
    python run_pipeline.py -n 10000  # Custom sim count

What it does
------------
1. Loads international match results (cached to data/raw/results.csv).
2. Fits a Dixon-Coles model (filtered to relevant teams, L-BFGS-B).
3. Pre-computes all 2,256 pairwise matchup CDFs.
4. Runs N Monte Carlo tournament simulations.
5. Prints top-10 predicted winners with win probabilities.
6. Runs backtests on 2014/2018/2022 World Cups (Brier score + Log Loss).
7. Saves results to data/processed/.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.loaders import load_international_results
from evaluation.backtest import backtest_all
from models.dixon_coles import fit
from simulation.monte_carlo import run_simulations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

PROCESSED_DIR = ROOT / "data" / "processed"
GROUPS_CSV = ROOT / "data" / "external" / "groups_2026.csv"
AS_OF = "2026-06-11"


def main() -> None:
    """Run the full prediction pipeline."""
    parser = argparse.ArgumentParser(description="WC 2026 Predictor Pipeline")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 1k simulations instead of 100k",
    )
    parser.add_argument(
        "-n", "--num-sims", type=int, default=None,
        help="Number of Monte Carlo simulations (default: 100k, --quick: 1k)",
    )
    parser.add_argument(
        "--skip-backtest", action="store_true",
        help="Skip backtesting step (faster for iteration)",
    )
    args = parser.parse_args()

    n_sims = args.num_sims or (1_000 if args.quick else 100_000)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # --- 1. Load data ---
    logger.info("=" * 60)
    logger.info("STEP 1: Loading international results")
    logger.info("=" * 60)
    results = load_international_results(start_year=1990, end_year=2026)
    logger.info("Loaded %d matches (1990–2026)", len(results))

    # --- 2. Load groups to know required teams ---
    groups_df = pd.read_csv(GROUPS_CSV)
    wc_teams = groups_df["team"].tolist()
    logger.info("WC 2026: %d teams in %d groups", len(wc_teams), groups_df["group"].nunique())

    # --- 3. Fit Dixon-Coles model ---
    logger.info("=" * 60)
    logger.info("STEP 2: Fitting Dixon-Coles model (as_of=%s)", AS_OF)
    logger.info("=" * 60)
    t_fit = time.time()
    model = fit(
        results,
        as_of=AS_OF,
        half_life_days=730.0,
        min_team_appearances=20,
        required_teams=wc_teams,
    )
    logger.info(
        "Model fitted in %.1fs: %d teams, gamma=%.4f, rho=%.4f",
        time.time() - t_fit, len(model.teams), model.home_advantage, model.rho,
    )

    # Save parameters
    params_df = model.get_params()
    params_path = PROCESSED_DIR / "dixon_coles_params.csv"
    params_df.to_csv(params_path, index=False)
    print("\nTop-10 attack ratings:")
    print(params_df.head(10).to_string(index=False))

    # --- 4. Pre-compute matchup CDFs ---
    logger.info("=" * 60)
    logger.info("STEP 3: Pre-computing matchup CDFs")
    logger.info("=" * 60)
    t_cache = time.time()
    model.precompute_matchups(wc_teams, neutral=True)
    logger.info("Cached in %.1fs", time.time() - t_cache)

    # --- 5. Run Monte Carlo simulation ---
    logger.info("=" * 60)
    logger.info("STEP 4: Monte Carlo simulation (%d runs)", n_sims)
    logger.info("=" * 60)
    t_sim = time.time()
    sim_results = run_simulations(
        model, groups_df, n=n_sims, seed=42, show_progress=True
    )
    logger.info("Simulation done in %.1fs", time.time() - t_sim)

    # Save
    sim_path = PROCESSED_DIR / "simulation_results.csv"
    sim_results.to_csv(sim_path, index=False)

    print("\n" + "=" * 60)
    print(f"PREDICTED WIN PROBABILITIES (Top 10, n={n_sims:,})")
    print("=" * 60)
    top10 = sim_results.head(10)[["team", "win_pct", "knockout_pct"]].copy()
    top10.columns = ["Team", "Win %", "Knockout %"]
    print(top10.to_string(index=False))

    # --- 6. Backtest ---
    if not args.skip_backtest:
        logger.info("=" * 60)
        logger.info("STEP 5: Backtesting on 2014, 2018, 2022 World Cups")
        logger.info("=" * 60)
        t_bt = time.time()
        summary = backtest_all(results, years=[2014, 2018, 2022], start_year=1990)
        logger.info("Backtest done in %.1fs", time.time() - t_bt)

        if len(summary):
            print("\n" + "=" * 60)
            print("BACKTEST RESULTS")
            print("=" * 60)
            print(summary.to_string(index=False))
            print(f"\nAverage Brier Score: {summary['brier_score'].mean():.4f}")
            print(f"Average Log Loss:   {summary['log_loss'].mean():.4f}")

            backtest_path = PROCESSED_DIR / "backtest_results.csv"
            summary.to_csv(backtest_path, index=False)
    else:
        logger.info("Skipping backtest (--skip-backtest flag)")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
