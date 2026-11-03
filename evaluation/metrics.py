"""Evaluation metrics for the WC 2026 Predictor.

Provides proper scoring rules for probabilistic match outcome predictions:
    - **Brier score** (multi-class, lower is better)
    - **Log loss** (cross-entropy, lower is better)
    - **Calibration table** for reliability diagrams

All functions operate on arrays of predicted probabilities vs. observed
outcomes encoded as one-hot vectors (W/D/L = 3 classes).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def brier_score(
    predicted: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """Compute the multi-class Brier score.

    The Brier score is the mean squared error between the predicted
    probability vector and the one-hot outcome vector, averaged over all
    matches.  Lower is better; 0 = perfect, 2 = worst possible for 3 classes.

    Parameters
    ----------
    predicted : np.ndarray of shape (n_matches, 3)
        Predicted probabilities [P(home_win), P(draw), P(away_win)] per match.
        Each row must sum to ~1.
    outcomes : np.ndarray of shape (n_matches, 3)
        One-hot encoded actual outcomes.  E.g. [1, 0, 0] = home win.

    Returns
    -------
    float
        Mean Brier score across all matches.

    Raises
    ------
    ValueError
        If shapes don't match or arrays are empty.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if predicted.shape != outcomes.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} vs outcomes {outcomes.shape}"
        )
    if predicted.size == 0:
        raise ValueError("Empty arrays.")

    # Brier score = mean of sum((p - o)^2) over classes, averaged over matches
    return float(np.mean(np.sum((predicted - outcomes) ** 2, axis=1)))


def log_loss(
    predicted: np.ndarray,
    outcomes: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Compute multi-class log loss (cross-entropy).

    Parameters
    ----------
    predicted : np.ndarray of shape (n_matches, 3)
        Predicted probabilities per match.
    outcomes : np.ndarray of shape (n_matches, 3)
        One-hot encoded actual outcomes.
    eps : float
        Clipping floor for predicted probabilities to avoid log(0).

    Returns
    -------
    float
        Mean log loss across all matches.  Lower is better.

    Raises
    ------
    ValueError
        If shapes don't match or arrays are empty.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if predicted.shape != outcomes.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape} vs outcomes {outcomes.shape}"
        )
    if predicted.size == 0:
        raise ValueError("Empty arrays.")

    # Clip to avoid log(0)
    clipped = np.clip(predicted, eps, 1.0 - eps)
    # Re-normalise rows after clipping
    clipped = clipped / clipped.sum(axis=1, keepdims=True)

    # Cross-entropy: -mean(sum(o * log(p)))
    return float(-np.mean(np.sum(outcomes * np.log(clipped), axis=1)))


def calibration_table(
    predicted: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a calibration (reliability) table.

    Groups predictions into bins by predicted probability and compares
    the mean predicted probability to the observed frequency within
    each bin.

    Parameters
    ----------
    predicted : np.ndarray of shape (n_matches, 3)
        Predicted probabilities.
    outcomes : np.ndarray of shape (n_matches, 3)
        One-hot outcomes.
    n_bins : int
        Number of equal-width bins from 0 to 1.

    Returns
    -------
    pd.DataFrame
        Columns: ``bin_lower, bin_upper, mean_predicted, observed_freq, count``.
        One row per non-empty bin (across all classes pooled together).
    """
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    outcomes = np.asarray(outcomes, dtype=np.float64).ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (predicted >= lo) & (predicted <= hi)
        else:
            mask = (predicted >= lo) & (predicted < hi)

        count = mask.sum()
        if count == 0:
            continue

        rows.append({
            "bin_lower": lo,
            "bin_upper": hi,
            "mean_predicted": float(predicted[mask].mean()),
            "observed_freq": float(outcomes[mask].mean()),
            "count": int(count),
        })

    return pd.DataFrame(rows)
