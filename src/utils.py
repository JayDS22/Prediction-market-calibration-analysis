"""
utils.py
Helper functions for prediction market calibration analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List


def binomial_ci(successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score confidence interval for binomial proportion.
    More accurate than normal approximation for extreme probabilities.
    """
    if trials == 0:
        return (0.0, 0.0)
    
    p_hat = successes / trials
    z = stats.norm.ppf(1 - alpha / 2)
    
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
    
    return (max(0, center - spread), min(1, center + spread))


def brier_score(forecasts: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score: mean squared error between forecasts and binary outcomes."""
    return np.mean((forecasts - outcomes) ** 2)


def brier_decomposition(forecasts: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> dict:
    """
    Murphy decomposition of Brier score into reliability, resolution, and uncertainty.
    
    BS = Reliability - Resolution + Uncertainty
    - Reliability: how well calibrated (lower = better)
    - Resolution: how much forecasts vary with outcomes (higher = better)
    - Uncertainty: base rate variance (constant for given dataset)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    base_rate = outcomes.mean()
    uncertainty = base_rate * (1 - base_rate)
    
    reliability = 0.0
    resolution = 0.0
    
    for i in range(n_bins):
        mask = (forecasts >= bin_edges[i]) & (forecasts < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (forecasts == bin_edges[i + 1])
        
        n_k = mask.sum()
        if n_k == 0:
            continue
        
        avg_forecast = forecasts[mask].mean()
        avg_outcome = outcomes[mask].mean()
        
        reliability += n_k * (avg_forecast - avg_outcome) ** 2
        resolution += n_k * (avg_outcome - base_rate) ** 2
    
    n = len(forecasts)
    reliability /= n
    resolution /= n
    
    return {
        'brier_score': brier_score(forecasts, outcomes),
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
        'skill_score': 1 - brier_score(forecasts, outcomes) / uncertainty
    }


def log_loss(forecasts: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Log loss (cross-entropy) between forecasts and outcomes."""
    forecasts = np.clip(forecasts, eps, 1 - eps)
    return -np.mean(outcomes * np.log(forecasts) + (1 - outcomes) * np.log(1 - forecasts))


def isotonic_recalibration(forecasts: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """
    Apply isotonic regression to recalibrate forecast probabilities.
    Returns recalibrated probabilities that are monotonically increasing.
    """
    from sklearn.isotonic import IsotonicRegression
    
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(forecasts, outcomes)
    return ir.predict(forecasts)


def platt_scaling(forecasts: np.ndarray, outcomes: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Apply Platt scaling (logistic recalibration).
    Fits: P(y=1|f) = 1 / (1 + exp(a*f + b))
    
    Returns:
        Tuple of (recalibrated_probs, params_dict)
    """
    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression(C=1e10, solver='lbfgs')
    lr.fit(forecasts.reshape(-1, 1), outcomes)
    
    recalibrated = lr.predict_proba(forecasts.reshape(-1, 1))[:, 1]
    
    params = {
        'coefficient': lr.coef_[0][0],
        'intercept': lr.intercept_[0]
    }
    
    return recalibrated, params


def expected_calibration_error(forecasts: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE): weighted average of per-bin calibration error.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(forecasts)
    
    for i in range(n_bins):
        mask = (forecasts >= bin_edges[i]) & (forecasts < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (forecasts == bin_edges[i + 1])
        
        n_k = mask.sum()
        if n_k == 0:
            continue
        
        avg_forecast = forecasts[mask].mean()
        avg_outcome = outcomes[mask].mean()
        ece += (n_k / n) * abs(avg_forecast - avg_outcome)
    
    return ece
