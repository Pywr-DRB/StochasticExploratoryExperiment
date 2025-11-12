"""
Test to verify the log transformation chain in Kirsch generator.
"""
import numpy as np
import pandas as pd
from methods.load import load_baseline_historical_flow
from sglib.methods.generation.nonparametric.kirsch import KirschGenerator

print("=" * 80)
print("TESTING LOG TRANSFORMATION CHAIN")
print("=" * 80)

# Load historical data
print("\n1. Loading historical data...")
Q_hist = load_baseline_historical_flow(gage_flow=True, period='baseline')
test_sites = ['cannonsville']
Q_hist = Q_hist.loc[:, test_sites]

n_zeros = (Q_hist == 0.0).sum().sum()
if n_zeros > 0:
    Q_hist.replace(0, np.nan, inplace=True)

print(f"   Daily data: {len(Q_hist)} days")
print(f"   Daily mean: {Q_hist['cannonsville'].mean():.2f}")
print(f"   Daily std: {Q_hist['cannonsville'].std():.2f}")

# Manually compute monthly aggregation
monthly_manual = Q_hist.groupby([Q_hist.index.year, Q_hist.index.month]).sum()
print(f"\n2. Monthly aggregation (before log):")
print(f"   Monthly mean: {monthly_manual['cannonsville'].mean():.2f}")
print(f"   Monthly std: {monthly_manual['cannonsville'].std():.2f}")
print(f"   Monthly min: {monthly_manual['cannonsville'].min():.2f}")
print(f"   Monthly max: {monthly_manual['cannonsville'].max():.2f}")

# Apply log
monthly_log = np.log(monthly_manual.clip(lower=1e-6))
print(f"\n3. After log transformation:")
print(f"   Log-monthly mean: {monthly_log['cannonsville'].mean():.4f}")
print(f"   Log-monthly std: {monthly_log['cannonsville'].std():.4f}")

# Fit Kirsch generator
print(f"\n4. Fitting Kirsch generator...")
kirsch_gen = KirschGenerator(Q_hist, generate_using_log_flow=True, debug=False)
kirsch_gen.preprocessing()
kirsch_gen.fit()

print(f"   Kirsch mean_month (Jan): {kirsch_gen.mean_month.iloc[0]['cannonsville']:.4f}")
print(f"   Kirsch std_month (Jan): {kirsch_gen.std_month.iloc[0]['cannonsville']:.4f}")

# Generate ONE synthetic series
print(f"\n5. Generating synthetic series...")
ensemble = kirsch_gen.generate(n_realizations=1, n_years=40, seed=42)
synth_daily = ensemble.data_by_realization[0]['cannonsville']

print(f"   Synthetic daily mean: {synth_daily.mean():.2f}")
print(f"   Synthetic daily std: {synth_daily.std():.2f}")

# Aggregate synthetic to monthly for comparison
synth_monthly = synth_daily.resample('MS').sum()
print(f"\n6. Synthetic monthly:")
print(f"   Synthetic monthly mean: {synth_monthly.mean():.2f}")
print(f"   Synthetic monthly std: {synth_monthly.std():.2f}")
print(f"   Synthetic monthly min: {synth_monthly.min():.2f}")
print(f"   Synthetic monthly max: {synth_monthly.max():.2f}")

# Compare
print(f"\n7. Comparison:")
print(f"   Historical monthly mean: {monthly_manual['cannonsville'].mean():.2f}")
print(f"   Synthetic monthly mean: {synth_monthly.mean():.2f}")
print(f"   Ratio (synth/hist): {synth_monthly.mean() / monthly_manual['cannonsville'].mean():.2f}x")

# Check if destandardization parameters are correct
print(f"\n8. Checking destandardization...")
print(f"   mean_month values (first 3 months): {kirsch_gen.mean_month['cannonsville'].iloc[:3].values}")
print(f"   std_month values (first 3 months): {kirsch_gen.std_month['cannonsville'].iloc[:3].values}")

# Manually check: these should be mean/std of LOG-TRANSFORMED monthly data
jan_log_mean = monthly_log['cannonsville'].loc[(slice(None), 1)].mean()
jan_log_std = monthly_log['cannonsville'].loc[(slice(None), 1)].std()
print(f"\n   Manual January log stats:")
print(f"     Mean(log(Q_jan)): {jan_log_mean:.4f}")
print(f"     Std(log(Q_jan)): {jan_log_std:.4f}")
print(f"   Kirsch January log stats:")
print(f"     mean_month[Jan]: {kirsch_gen.mean_month.iloc[0]['cannonsville']:.4f}")
print(f"     std_month[Jan]: {kirsch_gen.std_month.iloc[0]['cannonsville']:.4f}")

match_mean = np.isclose(jan_log_mean, kirsch_gen.mean_month.iloc[0]['cannonsville'], rtol=0.01)
match_std = np.isclose(jan_log_std, kirsch_gen.std_month.iloc[0]['cannonsville'], rtol=0.01)
print(f"\n   Mean matches: {match_mean}")
print(f"   Std matches: {match_std}")

print("\n" + "=" * 80)
