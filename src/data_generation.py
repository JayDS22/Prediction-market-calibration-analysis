"""
data_generation.py
Generates synthetic prediction market contract data calibrated to
empirically documented favorite-longshot bias patterns from CFTC-regulated
prediction markets (Burgi, Deng & Whelan, 2026).
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_contract_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic binary event contracts with realistic
    favorite-longshot bias patterns.
    
    Returns:
        DataFrame with columns: price_cents, implied_prob, outcome,
        contract_id, category
    """
    np.random.seed(seed)
    
    # Price buckets (contract prices in cents)
    price_buckets = list(range(5, 100, 5))
    
    # Empirically calibrated win rates exhibiting favorite-longshot bias
    # Low-price contracts win LESS than price implies
    # High-price contracts win MORE than price implies
    actual_win_rates = {
        5: 0.021, 10: 0.058, 15: 0.098, 20: 0.145, 25: 0.198,
        30: 0.255, 35: 0.310, 40: 0.370, 45: 0.430, 50: 0.500,
        55: 0.572, 60: 0.640, 65: 0.705, 70: 0.762, 75: 0.815,
        80: 0.862, 85: 0.908, 90: 0.952, 95: 0.978
    }
    
    # Contract counts per bucket (more activity at extremes)
    contracts_per_bucket = {
        5: 12400, 10: 8900, 15: 7200, 20: 6100, 25: 5400,
        30: 4800, 35: 4500, 40: 4200, 45: 4000, 50: 3800,
        55: 4000, 60: 4200, 65: 4500, 70: 4800, 75: 5400,
        80: 6100, 85: 7200, 90: 8900, 95: 12400
    }
    
    categories = ['economics', 'politics', 'weather', 'sports', 'culture']
    
    records = []
    contract_id = 0
    
    for price in price_buckets:
        n = contracts_per_bucket[price]
        win_rate = actual_win_rates[price]
        
        # Generate binary outcomes based on calibrated win rate
        outcomes = np.random.binomial(1, win_rate, n)
        
        for outcome in outcomes:
            records.append({
                'contract_id': f'CTR-{contract_id:07d}',
                'price_cents': price,
                'implied_prob': price / 100.0,
                'outcome': int(outcome),
                'category': np.random.choice(categories, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
                'volume_usd': round(np.random.lognormal(mean=3.5, sigma=1.2), 2)
            })
            contract_id += 1
    
    df = pd.DataFrame(records)
    return df


def get_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calibration statistics by price bucket.
    
    Returns:
        DataFrame with calibration metrics per price bucket
    """
    summary = df.groupby('price_cents').agg(
        n_contracts=('outcome', 'count'),
        wins=('outcome', 'sum'),
        total_volume=('volume_usd', 'sum')
    ).reset_index()
    
    summary['implied_prob'] = summary['price_cents'] / 100.0
    summary['actual_win_rate'] = summary['wins'] / summary['n_contracts']
    summary['calibration_error'] = summary['actual_win_rate'] - summary['implied_prob']
    summary['calibration_error_pp'] = summary['calibration_error'] * 100
    summary['expected_roi_pct'] = (summary['actual_win_rate'] / summary['implied_prob'] - 1) * 100
    
    # Binomial confidence intervals (95%)
    from scipy import stats
    summary['ci_lower'] = summary.apply(
        lambda r: stats.binom.ppf(0.025, r['n_contracts'], r['actual_win_rate']) / r['n_contracts'],
        axis=1
    )
    summary['ci_upper'] = summary.apply(
        lambda r: stats.binom.ppf(0.975, r['n_contracts'], r['actual_win_rate']) / r['n_contracts'],
        axis=1
    )
    
    return summary


def compute_brier_score(df: pd.DataFrame) -> float:
    """Compute Brier score measuring overall calibration quality."""
    return np.mean((df['implied_prob'] - df['outcome']) ** 2)


def compute_chi_squared(summary: pd.DataFrame) -> tuple:
    """Chi-squared goodness-of-fit test for calibration."""
    from scipy.stats import chisquare
    
    expected_wins = summary['implied_prob'] * summary['n_contracts']
    observed_wins = summary['wins']
    
    # Scale expected to match observed total (required by scipy)
    scale = observed_wins.sum() / expected_wins.sum()
    expected_wins_scaled = expected_wins * scale
    
    stat, p_value = chisquare(observed_wins, expected_wins_scaled)
    return stat, p_value


if __name__ == '__main__':
    # Generate and save data
    df = generate_contract_data()
    df.to_csv('data/simulated_contracts.csv', index=False)
    
    summary = get_bucket_summary(df)
    brier = compute_brier_score(df)
    chi2, p_val = compute_chi_squared(summary)
    
    print(f"Total contracts: {len(df):,}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Chi-squared stat: {chi2:.1f}, p-value: {p_val:.2e}")
    print(f"\nBucket Summary:")
    print(summary[['price_cents', 'n_contracts', 'actual_win_rate', 
                    'calibration_error_pp', 'expected_roi_pct']].to_string(index=False))
