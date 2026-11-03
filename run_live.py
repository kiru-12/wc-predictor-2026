"""Live prediction pipeline — conditioned on actual WC 2026 results.

Usage
-----
    python run_live.py              # 10k sims (default, ~5 min)
    python run_live.py --quick      # 1k sims (~30s)
    python run_live.py -n 100000   # full 100k sims

Conditions predictions on all WC 2026 matches played through June 25, 2026.
Groups A, B, C are locked (all matches played). Groups D-L have 1 matchday
remaining — those matches are simulated.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.loaders import load_international_results
from models.dixon_coles import fit
from models.momentum import compute_momentum_adjustment
from simulation.live import load_wc2026_results, run_live_simulations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_pipeline")

PROCESSED_DIR = ROOT / "data" / "processed"
GROUPS_CSV = ROOT / "data" / "external" / "groups_2026.csv"
WC_RESULTS_CSV = ROOT / "data" / "external" / "wc2026_results.csv"
AS_OF = "2026-06-11"  # Fit model on pre-tournament data only


def main() -> None:
    parser = argparse.ArgumentParser(description="Live WC 2026 Predictor")
    parser.add_argument("--quick", action="store_true", help="1k sims")
    parser.add_argument("-n", "--num-sims", type=int, default=None)
    args = parser.parse_args()

    n_sims = args.num_sims or (1_000 if args.quick else 10_000)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # --- Load data ---
    logger.info("Loading international results...")
    results = load_international_results(start_year=1990, end_year=2026)
    groups_df = pd.read_csv(GROUPS_CSV)
    wc_teams = groups_df["team"].tolist()

    # --- Load actual WC 2026 results ---
    wc_results = load_wc2026_results(str(WC_RESULTS_CSV))
    n_played = len(wc_results)
    complete_groups = []
    for g, matches in wc_results.groupby("group"):
        if len(matches) == 6:
            complete_groups.append(g)
    logger.info(
        "WC 2026 results: %d matches played, %d groups complete (%s)",
        n_played, len(complete_groups), ", ".join(sorted(complete_groups))
    )

    # --- Fit Dixon-Coles (pre-tournament data only) ---
    logger.info("Fitting Dixon-Coles (as_of=%s)...", AS_OF)
    t_fit = time.time()
    model = fit(
        results, as_of=AS_OF, half_life_days=730.0,
        min_team_appearances=20, required_teams=wc_teams,
    )
    logger.info("Model fitted in %.1fs (%d teams)", time.time() - t_fit, len(model.teams))

    # Print top attack ratings
    params = model.get_params()
    print("\nPre-tournament attack ratings (top 10):")
    print(params.head(10).to_string(index=False))

    # --- Apply momentum adjustment from WC results ---
    logger.info("Applying momentum adjustment (eta=0.4)...")
    model = compute_momentum_adjustment(model, wc_results, eta=0.4)

    params_adj = model.get_params()
    print("\nAdjusted attack ratings (post-momentum, top 10):")
    print(params_adj.head(10).to_string(index=False))

    # --- Pre-compute CDFs ---
    model.precompute_matchups(wc_teams, neutral=True)

    # --- Run live simulation ---
    print(f"\n{'='*60}")
    print(f"LIVE SIMULATION ({n_sims:,} runs, conditioned on {n_played} actual matches)")
    print(f"{'='*60}")
    t_sim = time.time()
    sim_results = run_live_simulations(
        model, groups_df, wc_results,
        n=n_sims, seed=42, show_progress=True,
    )
    logger.info("Simulation done in %.1fs", time.time() - t_sim)

    # --- Results ---
    sim_path = PROCESSED_DIR / "live_simulation_results.csv"
    sim_results.to_csv(sim_path, index=False)

    # Teams already eliminated (0% knockout)
    eliminated = sim_results[sim_results["knockout_pct"] == 0.0]["team"].tolist()

    print(f"\n{'='*60}")
    print(f"PREDICTED WINNER (Top 15)")
    print(f"{'='*60}")
    top15 = sim_results.head(15)[["team", "win_pct", "knockout_pct"]].copy()
    top15.columns = ["Team", "Win %", "Knockout %"]
    print(top15.to_string(index=False))

    if eliminated:
        print(f"\nEliminated ({len(eliminated)}): {', '.join(eliminated)}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
