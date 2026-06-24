# World Cup 2026 Predictor

This project forecasts the 2026 FIFA World Cup using a Monte Carlo simulation (100 000 runs) that combines Dixon-Coles goal modelling, team ELO ratings, player impact scores (normalised to per-90 minutes), and a logistic upset model driven by the Underdog UPI metric. All weights are learned through backtesting on the 2014, 2018, and 2022 tournaments using only point-in-time data, and performance is evaluated with Brier score and Log Loss.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Project structure

```
data/
  raw/          # cached downloads — never re-downloaded
  processed/    # cleaned, feature-ready files
  external/     # third-party reference data
features/       # feature engineering modules
models/         # Dixon-Coles, upset model, team strength
simulation/     # Monte Carlo engine
evaluation/     # backtesting, Brier score, Log Loss
dashboard/      # Streamlit app
scrapers/       # last-resort scrapers (robots.txt + throttling required)
notebooks/      # exploratory analysis
tests/          # pytest test suite
```
