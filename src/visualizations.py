"""
visualizations.py
Publication-quality figures for prediction market calibration analysis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Color palette
ACCENT = '#2B5C8A'
RED = '#C0392B'
GREEN = '#27AE60'
GRAY = '#7F8C8D'
DARK = '#2C3E50'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def plot_calibration(summary: pd.DataFrame, output_path: str):
    """
    Figure 1: Calibration plot - observed win rate vs implied probability.
    Shows the favorite-longshot bias with shaded regions.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    
    x = summary['implied_prob'].values
    y = summary['actual_win_rate'].values
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], '--', color=GRAY, linewidth=1, label='Perfect Calibration', zorder=1)
    
    # Data points and connecting line
    ax.scatter(x, y, c=ACCENT, s=50, zorder=3, edgecolors='white', linewidth=0.5)
    ax.plot(x, y, '-', color=ACCENT, linewidth=1.5, alpha=0.7, label='Observed Win Rate', zorder=2)
    
    # Shade bias regions
    midpoint = len(x) // 2
    ax.fill_between(x[:midpoint+1], x[:midpoint+1], y[:midpoint+1],
                    alpha=0.15, color=RED, label='Longshot Overpricing')
    ax.fill_between(x[midpoint:], x[midpoint:], y[midpoint:],
                    alpha=0.15, color=GREEN, label='Favorite Underpricing')
    
    ax.set_xlabel('Contract Price (Implied Probability)')
    ax.set_ylabel('Actual Win Rate')
    ax.set_title('Calibration Analysis: Favorite-Longshot Bias in Prediction Markets')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    fig.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_roi(summary: pd.DataFrame, output_path: str):
    """
    Figure 2: Expected ROI by contract price bucket.
    Red = negative returns, Green = positive returns.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    
    prices = summary['price_cents'].values
    roi = summary['expected_roi_pct'].values
    bar_colors = [RED if r < 0 else GREEN for r in roi]
    
    ax.bar(prices, roi, width=4, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color=DARK, linewidth=0.8, linestyle='-')
    
    ax.set_xlabel('Contract Price (cents)')
    ax.set_ylabel('Expected ROI (%)')
    ax.set_title('Expected Return on Investment by Contract Price')
    ax.set_xticks(prices[::2])
    
    fig.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_calibration_error(summary: pd.DataFrame, output_path: str):
    """
    Figure 3: Calibration error decomposition by price bucket.
    Shows deviation from perfect calibration in percentage points.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    
    prices = summary['price_cents'].values
    errors = summary['calibration_error_pp'].values
    error_colors = [RED if e < 0 else GREEN for e in errors]
    
    ax.bar(prices, errors, width=4, color=error_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color=DARK, linewidth=0.8)
    
    ax.set_xlabel('Contract Price (cents)')
    ax.set_ylabel('Calibration Error (pp)')
    ax.set_title('Calibration Error by Contract Price Bucket')
    ax.set_xticks(prices[::2])
    
    fig.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def create_all_figures(summary: pd.DataFrame, output_dir: str = 'figures'):
    """Generate all publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    plot_calibration(summary, os.path.join(output_dir, 'fig1_calibration.png'))
    plot_roi(summary, os.path.join(output_dir, 'fig2_roi.png'))
    plot_calibration_error(summary, os.path.join(output_dir, 'fig3_calibration_error.png'))
