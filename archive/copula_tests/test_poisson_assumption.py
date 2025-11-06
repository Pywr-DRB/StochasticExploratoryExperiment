"""
Test whether drought occurrence follows a Poisson process.

This script analyzes the drought events from SSI calculations to determine:
1. Whether drought counts per realization follow Poisson distribution
2. Whether inter-event times follow exponential distribution
3. How to best calculate interarrival time for return period analysis

Usage:
    python test_poisson_assumption.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_drought_data(fname):
    """Load and preprocess drought event data."""
    df = pd.read_csv(fname)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    return df


def test_poisson_count_distribution(df, n_years=70):
    """
    Test if drought counts per realization follow Poisson distribution.

    For Poisson process: P(N events in time T) ~ Poisson(λT)
    """
    print("="*60)
    print("TEST 1: POISSON COUNT DISTRIBUTION")
    print("="*60)

    # Count droughts per realization
    counts = df.groupby('realization_id').size()
    n_realizations = counts.shape[0]

    print(f"\nDrought counts across {n_realizations} realizations ({n_years} years each):")
    print(f"  Mean: {counts.mean():.2f} droughts/realization")
    print(f"  Std:  {counts.std():.2f}")
    print(f"  Min:  {counts.min()}")
    print(f"  Max:  {counts.max()}")

    # Expected Poisson parameters
    lambda_hat = counts.mean()

    print(f"\nPoisson fit: lambda = {lambda_hat:.3f} events/realization")
    print(f"  -> Rate = {lambda_hat/n_years:.4f} events/year")

    # Test if variance ≈ mean (Poisson property)
    variance = counts.var()
    print(f"\nPoisson property check (variance should equal mean):")
    print(f"  Mean:     {lambda_hat:.3f}")
    print(f"  Variance: {variance:.3f}")
    print(f"  Ratio:    {variance/lambda_hat:.3f} (should be ~1.0)")

    if abs(variance/lambda_hat - 1.0) < 0.15:
        print(f"  [PASS] Ratio close to 1.0 (within 15%)")
    else:
        print(f"  [FAIL] Ratio deviates significantly from 1.0")

    # Chi-squared goodness of fit test
    observed_counts = counts.value_counts().sort_index()
    k_values = observed_counts.index
    k_min, k_max = k_values.min(), k_values.max()

    # Expected frequencies under Poisson
    expected_freq = {}
    for k in range(k_min, k_max + 1):
        expected_freq[k] = n_realizations * stats.poisson.pmf(k, lambda_hat)

    # Combine bins with low expected frequency
    obs = []
    exp = []
    for k in range(k_min, k_max + 1):
        obs.append(observed_counts.get(k, 0))
        exp.append(expected_freq[k])

    obs = np.array(obs)
    exp = np.array(exp)

    # Chi-squared test (combine bins with expected < 5)
    mask = exp >= 5
    if mask.sum() >= 2:
        chi2_stat = np.sum((obs[mask] - exp[mask])**2 / exp[mask])
        df_chi2 = mask.sum() - 1 - 1  # bins - 1 - estimated parameters
        p_value = 1 - stats.chi2.cdf(chi2_stat, df_chi2)

        print(f"\nChi-squared goodness-of-fit test:")
        print(f"  Chi-squared statistic: {chi2_stat:.3f}")
        print(f"  df: {df_chi2}")
        print(f"  p-value: {p_value:.4f}")

        if p_value > 0.05:
            print(f"  [PASS] Cannot reject Poisson hypothesis (p > 0.05)")
        else:
            print(f"  [FAIL] Reject Poisson hypothesis (p < 0.05)")

    return counts, lambda_hat


def test_exponential_interarrival(df, n_years=70):
    """
    Test if inter-event times follow exponential distribution.

    For Poisson process: Inter-event times ~ Exponential(λ)
    """
    print("\n" + "="*60)
    print("TEST 2: EXPONENTIAL INTER-EVENT TIMES")
    print("="*60)

    # Calculate inter-event times (start-to-start) within each realization
    df_sorted = df.sort_values(['realization_id', 'start'])

    interarrival_times = []
    for rid, group in df_sorted.groupby('realization_id'):
        starts = group['start'].values
        if len(starts) > 1:
            intervals = np.diff(starts).astype('timedelta64[D]').astype(float)
            interarrival_times.extend(intervals)

    interarrival_times = np.array(interarrival_times) / 365.25  # Convert to years

    print(f"\nInter-event times (n={len(interarrival_times)} intervals):")
    print(f"  Mean: {interarrival_times.mean():.2f} years")
    print(f"  Std:  {interarrival_times.std():.2f} years")
    print(f"  Min:  {interarrival_times.min():.2f} years")
    print(f"  Max:  {interarrival_times.max():.2f} years")

    # Fit exponential distribution
    lambda_fit = 1.0 / interarrival_times.mean()

    print(f"\nExponential fit: lambda = {lambda_fit:.4f} events/year")
    print(f"  -> Mean inter-event time = {1/lambda_fit:.2f} years")

    # Coefficient of variation (should be 1.0 for exponential)
    cv = interarrival_times.std() / interarrival_times.mean()
    print(f"\nCoefficient of variation:")
    print(f"  Observed: {cv:.3f}")
    print(f"  Expected (exponential): 1.000")

    if abs(cv - 1.0) < 0.15:
        print(f"  [PASS] CV close to 1.0 (within 15%)")
    else:
        print(f"  [FAIL] CV deviates from 1.0")

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.kstest(interarrival_times,
                                     lambda x: stats.expon.cdf(x, scale=1/lambda_fit))

    print(f"\nKolmogorov-Smirnov test:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_pval:.4f}")

    if ks_pval > 0.05:
        print(f"  [PASS] Cannot reject exponential hypothesis (p > 0.05)")
    else:
        print(f"  [FAIL] Reject exponential hypothesis (p < 0.05)")

    return interarrival_times, lambda_fit


def compare_rate_estimates(df, n_years=70, n_realizations=1000):
    """
    Compare different methods of estimating drought occurrence rate.
    """
    print("\n" + "="*60)
    print("TEST 3: RATE ESTIMATION METHODS")
    print("="*60)

    # Method 1: Direct count rate
    total_droughts = len(df)
    total_years = n_realizations * n_years
    lambda_direct = total_droughts / total_years

    print(f"\nMethod 1: Direct count")
    print(f"  Total droughts: {total_droughts}")
    print(f"  Total years: {total_years}")
    print(f"  Rate: lambda = {lambda_direct:.4f} events/year")
    print(f"  Mean inter-event: {1/lambda_direct:.2f} years")

    # Method 2: Mean inter-event interval (current approach)
    df_sorted = df.sort_values(['realization_id', 'start'])
    starts = (
        df_sorted.groupby('realization_id')['start']
        .apply(lambda s: s.sort_values().diff().dt.days.dropna())
        .to_numpy()
    )
    mean_interval_days = np.mean(starts)
    mean_interval_years = mean_interval_days / 365.25
    lambda_interval = 1.0 / mean_interval_years

    print(f"\nMethod 2: Mean inter-event interval (current code)")
    print(f"  Mean interval: {mean_interval_years:.2f} years")
    print(f"  Rate: lambda = {lambda_interval:.4f} events/year")

    # Method 3: Average per-realization rate
    counts = df.groupby('realization_id').size()
    rates_per_realization = counts / n_years
    lambda_avg_rate = rates_per_realization.mean()

    print(f"\nMethod 3: Average per-realization rate")
    print(f"  Rate: lambda = {lambda_avg_rate:.4f} events/year")
    print(f"  Mean inter-event: {1/lambda_avg_rate:.2f} years")

    # Compare methods
    print(f"\n" + "-"*60)
    print(f"COMPARISON:")
    print(f"  Method 1 (direct):     lambda = {lambda_direct:.4f} events/year")
    print(f"  Method 2 (intervals):  lambda = {lambda_interval:.4f} events/year")
    print(f"  Method 3 (avg rate):   lambda = {lambda_avg_rate:.4f} events/year")

    print(f"\n  Difference (Method 1 vs 2): {abs(lambda_direct - lambda_interval)/lambda_direct * 100:.2f}%")
    print(f"  Difference (Method 1 vs 3): {abs(lambda_direct - lambda_avg_rate)/lambda_direct * 100:.2f}%")

    if abs(lambda_direct - lambda_interval)/lambda_direct < 0.05:
        print(f"\n  [PASS] Methods agree within 5%")
    else:
        print(f"\n  [WARN] Methods differ by more than 5%")

    return lambda_direct, lambda_interval, lambda_avg_rate


def create_diagnostic_plots(counts, interarrival_times, lambda_hat, lambda_fit):
    """
    Create diagnostic plots for Poisson/exponential testing.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Drought counts histogram vs Poisson
    ax = axes[0, 0]
    count_values = counts.value_counts().sort_index()
    x = count_values.index
    ax.bar(x, count_values.values, alpha=0.7, label='Observed', color='skyblue')

    # Overlay Poisson PMF
    x_theory = np.arange(counts.min(), counts.max() + 1)
    poisson_probs = stats.poisson.pmf(x_theory, lambda_hat) * len(counts)
    ax.plot(x_theory, poisson_probs, 'ro-', lw=2, label=f'Poisson(lambda={lambda_hat:.2f})')

    ax.set_xlabel('Number of droughts per realization')
    ax.set_ylabel('Frequency')
    ax.set_title('Drought Count Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Q-Q plot for Poisson
    ax = axes[0, 1]
    # Use empirical quantiles vs theoretical Poisson quantiles
    sorted_counts = np.sort(counts)
    p = (np.arange(len(sorted_counts)) + 0.5) / len(sorted_counts)
    theoretical_quantiles = stats.poisson.ppf(p, lambda_hat)

    ax.scatter(theoretical_quantiles, sorted_counts, alpha=0.5, s=20)
    ax.plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
            [theoretical_quantiles.min(), theoretical_quantiles.max()],
            'r--', lw=2, label='Perfect fit')
    ax.set_xlabel('Theoretical Poisson quantiles')
    ax.set_ylabel('Observed quantiles')
    ax.set_title('Q-Q Plot: Count Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Inter-event times histogram vs Exponential
    ax = axes[1, 0]
    ax.hist(interarrival_times, bins=30, density=True, alpha=0.7,
            label='Observed', color='lightcoral')

    # Overlay exponential PDF
    x_exp = np.linspace(0, interarrival_times.max(), 100)
    exp_pdf = stats.expon.pdf(x_exp, scale=1/lambda_fit)
    ax.plot(x_exp, exp_pdf, 'r-', lw=2, label=f'Exponential(lambda={lambda_fit:.4f})')

    ax.set_xlabel('Inter-event time (years)')
    ax.set_ylabel('Density')
    ax.set_title('Inter-Event Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Q-Q plot for Exponential
    ax = axes[1, 1]
    sorted_intervals = np.sort(interarrival_times)
    p = (np.arange(len(sorted_intervals)) + 0.5) / len(sorted_intervals)
    theoretical_quantiles = stats.expon.ppf(p, scale=1/lambda_fit)

    ax.scatter(theoretical_quantiles, sorted_intervals, alpha=0.5, s=20)
    ax.plot([0, theoretical_quantiles.max()],
            [0, theoretical_quantiles.max()],
            'r--', lw=2, label='Perfect fit')
    ax.set_xlabel('Theoretical exponential quantiles (years)')
    ax.set_ylabel('Observed quantiles (years)')
    ax.set_title('Q-Q Plot: Inter-Event Times')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main analysis function."""

    print("\n" + "="*60)
    print("POISSON PROCESS ASSUMPTION TESTING")
    print("Drought Events Analysis")
    print("="*60)

    # Load data
    fname = "./pywrdrb/drought_metrics/stationary_ensemble_ssi12_drought_events.csv"
    print(f"\nLoading: {fname}")

    df = load_drought_data(fname)

    # Get metadata
    n_realizations = df['realization_id'].nunique()
    n_years = 70  # From your config

    print(f"  Realizations: {n_realizations}")
    print(f"  Years per realization: {n_years}")
    print(f"  Total drought events: {len(df)}")

    # Test 1: Poisson count distribution
    counts, lambda_hat = test_poisson_count_distribution(df, n_years)

    # Test 2: Exponential inter-event times
    interarrival_times, lambda_fit = test_exponential_interarrival(df, n_years)

    # Test 3: Compare rate estimation methods
    lambda_direct, lambda_interval, lambda_avg = compare_rate_estimates(
        df, n_years, n_realizations
    )

    # Summary and recommendation
    print("\n" + "="*60)
    print("SUMMARY AND RECOMMENDATION")
    print("="*60)

    # Check consistency
    variance_ratio = counts.var() / counts.mean()
    cv = interarrival_times.std() / interarrival_times.mean()
    rate_diff = abs(lambda_direct - lambda_interval) / lambda_direct

    poisson_score = 0
    if abs(variance_ratio - 1.0) < 0.15:
        poisson_score += 1
        print("\n[PASS] Variance-to-mean ratio supports Poisson")
    else:
        print("\n[FAIL] Variance-to-mean ratio does not support Poisson")

    if abs(cv - 1.0) < 0.15:
        poisson_score += 1
        print("[PASS] Coefficient of variation supports exponential inter-event times")
    else:
        print("[FAIL] Coefficient of variation does not support exponential")

    if rate_diff < 0.05:
        poisson_score += 1
        print("[PASS] Rate estimation methods agree")
    else:
        print("[FAIL] Rate estimation methods differ")

    print(f"\nPoisson score: {poisson_score}/3")

    if poisson_score >= 2:
        print("\n" + "="*60)
        print("CONCLUSION: Poisson assumption is REASONABLE")
        print("="*60)
        print("\nRECOMMENDATION:")
        print("  Either approach is valid:")
        print(f"  1. Use direct rate: lambda = {lambda_direct:.4f} events/year")
        print(f"  2. Use inter-event: E[L] = {1/lambda_interval:.2f} years")
        print("\n  Current code approach (Method 2) is theoretically sound.")
        print("  The methods are equivalent for large samples.")
    else:
        print("\n" + "="*60)
        print("CONCLUSION: Poisson assumption is QUESTIONABLE")
        print("="*60)
        print("\nRECOMMENDATION:")
        print("  Use direct rate calculation (more robust):")
        print(f"    lambda = {lambda_direct:.4f} events/year")
        print(f"    E[L] = {1/lambda_direct:.2f} years")
        print("\n  Consider non-Poisson models:")
        print("    - Overdispersed Poisson (Negative Binomial)")
        print("    - Renewal process with non-exponential inter-event times")
        print("    - Empirical return periods without parametric assumptions")

    # Create plots
    print("\nCreating diagnostic plots...")
    fig = create_diagnostic_plots(counts, interarrival_times, lambda_hat, lambda_fit)

    output_file = "./pywrdrb/drought_metrics/poisson_diagnostics.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
