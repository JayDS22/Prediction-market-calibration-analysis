"""
calibration_analysis.py
Core analysis pipeline for prediction market calibration research.
Runs end-to-end: data generation -> statistical analysis -> visualization -> report.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
from data_generation import generate_contract_data, get_bucket_summary, compute_brier_score, compute_chi_squared
from visualizations import create_all_figures


def run_bootstrap_roi(df: pd.DataFrame, n_bootstrap: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for ROI by price bucket.
    
    Args:
        df: Contract-level DataFrame
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed
        
    Returns:
        DataFrame with ROI point estimates and 95% CI
    """
    np.random.seed(seed)
    results = []
    
    for price, group in df.groupby('price_cents'):
        implied_prob = price / 100.0
        outcomes = group['outcome'].values
        
        boot_rois = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(outcomes, size=len(outcomes), replace=True)
            win_rate = sample.mean()
            roi = (win_rate / implied_prob - 1) * 100
            boot_rois.append(roi)
        
        boot_rois = np.array(boot_rois)
        results.append({
            'price_cents': price,
            'roi_mean': boot_rois.mean(),
            'roi_ci_lower': np.percentile(boot_rois, 2.5),
            'roi_ci_upper': np.percentile(boot_rois, 97.5),
            'roi_std': boot_rois.std()
        })
    
    return pd.DataFrame(results)


def analyze_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare calibration across event categories.
    
    Returns:
        DataFrame with per-category calibration metrics
    """
    results = []
    
    for category, group in df.groupby('category'):
        summary = get_bucket_summary(group)
        brier = compute_brier_score(group)
        
        # Mean absolute calibration error
        mace = summary['calibration_error'].abs().mean()
        
        # Weighted by volume
        weighted_error = np.average(
            summary['calibration_error'].abs(),
            weights=summary['n_contracts']
        )
        
        results.append({
            'category': category,
            'n_contracts': len(group),
            'brier_score': brier,
            'mean_abs_cal_error': mace,
            'weighted_abs_cal_error': weighted_error,
            'total_volume_usd': group['volume_usd'].sum()
        })
    
    return pd.DataFrame(results).sort_values('brier_score')


def compute_calibration_regression(summary: pd.DataFrame) -> dict:
    """
    Fit linear regression: actual_win_rate = a + b * implied_prob
    Perfect calibration: a=0, b=1
    
    Returns:
        Dict with regression coefficients, R-squared, and test statistics
    """
    x = summary['implied_prob'].values
    y = summary['actual_win_rate'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Test if slope differs from 1 (perfect calibration)
    t_stat_slope = (slope - 1) / std_err
    p_slope_diff_1 = 2 * (1 - stats.t.cdf(abs(t_stat_slope), df=len(x) - 2))
    
    # Test if intercept differs from 0
    n = len(x)
    x_mean = x.mean()
    se_intercept = std_err * np.sqrt(np.sum(x**2) / n)
    t_stat_intercept = intercept / se_intercept
    p_intercept_diff_0 = 2 * (1 - stats.t.cdf(abs(t_stat_intercept), df=n - 2))
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'slope_std_err': std_err,
        'slope_p_value': p_value,
        'slope_diff_from_1_p': p_slope_diff_1,
        'intercept_diff_from_0_p': p_intercept_diff_0
    }


def print_report(summary, brier, chi2, p_val, regression, bootstrap_roi, category_analysis):
    """Print formatted analysis report to console."""
    
    print("=" * 70)
    print("CALIBRATION ANALYSIS OF BINARY EVENT CONTRACTS")
    print("IN CFTC-REGULATED PREDICTION MARKETS")
    print("=" * 70)
    
    print(f"\n--- Overall Statistics ---")
    print(f"Total contracts analyzed: {summary['n_contracts'].sum():,}")
    print(f"Brier Score: {brier:.4f} (lower = better calibrated)")
    print(f"Chi-squared: {chi2:.1f} (p = {p_val:.2e})")
    
    print(f"\n--- Calibration Regression ---")
    print(f"Win Rate = {regression['intercept']:.4f} + {regression['slope']:.4f} * Implied Prob")
    print(f"R-squared: {regression['r_squared']:.4f}")
    print(f"Slope = {regression['slope']:.4f} (perfect = 1.000)")
    print(f"  H0: slope = 1, p = {regression['slope_diff_from_1_p']:.4f}")
    print(f"Intercept = {regression['intercept']:.4f} (perfect = 0.000)")
    print(f"  H0: intercept = 0, p = {regression['intercept_diff_from_0_p']:.4f}")
    
    print(f"\n--- Calibration by Price Bucket ---")
    cols = ['price_cents', 'n_contracts', 'actual_win_rate', 'calibration_error_pp', 'expected_roi_pct']
    print(summary[cols].to_string(index=False, float_format='%.2f'))
    
    print(f"\n--- Bootstrap ROI (95% CI) ---")
    for _, row in bootstrap_roi.iterrows():
        print(f"  {int(row['price_cents']):3d}c: ROI = {row['roi_mean']:+6.1f}% "
              f"[{row['roi_ci_lower']:+6.1f}%, {row['roi_ci_upper']:+6.1f}%]")
    
    print(f"\n--- Category Analysis ---")
    print(category_analysis.to_string(index=False, float_format='%.4f'))
    
    print(f"\n--- Key Findings ---")
    longshot_roi = bootstrap_roi[bootstrap_roi['price_cents'] <= 25]['roi_mean'].mean()
    favorite_roi = bootstrap_roi[bootstrap_roi['price_cents'] >= 75]['roi_mean'].mean()
    print(f"Longshot (5-25c) avg ROI: {longshot_roi:+.1f}%")
    print(f"Favorite (75-95c) avg ROI: {favorite_roi:+.1f}%")
    print(f"Bias is {'statistically significant' if p_val < 0.05 else 'not significant'} (p = {p_val:.2e})")
    print("=" * 70)


def main():
    """Run complete analysis pipeline."""
    
    print("Generating synthetic contract data...")
    df = generate_contract_data()
    
    # Save raw data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/simulated_contracts.csv', index=False)
    print(f"Saved {len(df):,} contracts to data/simulated_contracts.csv")
    
    print("Computing calibration statistics...")
    summary = get_bucket_summary(df)
    brier = compute_brier_score(df)
    chi2, p_val = compute_chi_squared(summary)
    
    print("Running calibration regression...")
    regression = compute_calibration_regression(summary)
    
    print("Running bootstrap ROI analysis (10,000 iterations)...")
    bootstrap_roi = run_bootstrap_roi(df, n_bootstrap=10000)
    
    print("Analyzing calibration by category...")
    category_analysis = analyze_by_category(df)
    
    print("Generating publication-quality figures...")
    os.makedirs('figures', exist_ok=True)
    create_all_figures(summary, output_dir='figures')
    
    # Print report
    print_report(summary, brier, chi2, p_val, regression, bootstrap_roi, category_analysis)
    
    print("\nAnalysis complete. Figures saved to figures/")


if __name__ == '__main__':
    main()
