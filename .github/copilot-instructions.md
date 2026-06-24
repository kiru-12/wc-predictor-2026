# Project: World Cup 2026 Predictor

## Goal
Predict the 2026 FIFA World Cup via Monte Carlo simulation (100k runs). Combine
team strength, player impact, and an underdog factor. Backtest on 2014/2018/2022.

## Environment
- Python 3.11+, virtual env in ./venv, dependencies in requirements.txt.
- Structure: data/{raw,processed,external}, features/, models/, simulation/,
  evaluation/, dashboard/, scrapers/ (last resort only), notebooks/, tests/.

## Hard rules — data
- PREFER libraries/APIs over scraping. Use `soccerdata` (FBref, ELO), StatsBomb
  open data via `statsbombpy`, and football-data.co.uk CSVs.
- Do NOT scrape SofaScore (anti-bot + ToS). Find the stat elsewhere.
- If scraping is unavoidable: check robots.txt and ToS first, throttle to ~1
  request / 3-5s, add retries + caching, and flag it to me in a comment.
- Cache all downloads to data/raw/. Never re-download on every run.

## Hard rules — modelling
- Normalise all player counting stats to per-90 before scoring. Never reward
  raw volume.
- Use Dixon-Coles (not plain independent Poisson) for match goals — it handles
  draws and low-score correlation.
- ONE underdog signal only. Use the Underdog UPI metric as an input FEATURE to
  the logistic upset model; let the model learn its weight. Do NOT also apply a
  separate hardcoded underdog multiplier. No magic blend constants like
  0.65/0.35 — every weight is learned or calibrated by backtesting.
- Backtests must use POINT-IN-TIME data only (ELO/odds/form as of before the
  tournament). No post-tournament leakage. State this guarantee in each backtest
  function's docstring.
- Evaluate with Brier score and Log Loss.

## Code style
- Type hints + concise docstrings on every public function.
- Use the `logging` module, not print, for pipeline status.
- Small, testable functions. Add a pytest test for each non-trivial function.
- A missing input column should log a warning and produce NaN, not crash.

## Process
- Build ONE file/module at a time. Don't scaffold ahead of what I asked for.
- After creating a file, tell me how to run it and what output to expect.
