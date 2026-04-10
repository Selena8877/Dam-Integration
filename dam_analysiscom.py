"""
MATH 4020U - Course Project: Section 3.8.3 Dam Integration
Site: Columbia River at The Dalles, OR (USGS Site 14105700)
Data: Daily Mean Discharge (ft^3/s), March 1, 2025 - February 28, 2026
Students: Bushrat Zahan (100919304), and Selena Bradshaw (100919722)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import simpson

#loading and cleaning the data
df = pd.read_csv('items.csv')

#keeps only the columns we need
df = df[['time', 'value', 'unit_of_measure']].copy()
df.columns = ['date', 'discharge_cfs', 'unit']

#parses through the dates and sorts in chronological order
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

#filter to exactly one year (March 1, 2025 - February 28, 2026)
df = df[(df['date'] >= '2025-03-01') & (df['date'] <= '2026-02-28')].reset_index(drop=True)

#drops any rows with missing discharge values
df = df.dropna(subset=['discharge_cfs']).reset_index(drop=True)

print("=" * 60)
print("DATA SUMMARY")
print("=" * 60)
print(f"Site:          Columbia River at The Dalles, OR")
print(f"USGS Site No.: 14105700")
print(f"Date range:    {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total days:    {len(df)}")
print(f"Units:         ft^3/s (cubic feet per second)")
print(f"Min discharge: {df['discharge_cfs'].min():,.0f} ft^3/s")
print(f"Max discharge: {df['discharge_cfs'].max():,.0f} ft^3/s")
print(f"Mean discharge:{df['discharge_cfs'].mean():,.0f} ft^3/s")
print()


#creating the plot
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df['date'], df['discharge_cfs'], color='steelblue', linewidth=1.5, label='Daily Mean Discharge')
ax.fill_between(df['date'], df['discharge_cfs'], alpha=0.2, color='steelblue')

#formatting the plot title, lables, ticks, and grid
ax.set_title('Daily Mean Discharge – Columbia River at The Dalles, OR\n'
             'USGS Site 14105700 | March 2025 – February 2026', fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Discharge (ft³/s)', fontsize=11)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('discharge_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as: discharge_plot.png")
print()

# ─────────────────────────────────────────────
#Numerical Integration Estimations
#each time step h = 1 day = 86,400 seconds
#Volume = integral of Q(t) dt  [ft^3/s * s = ft^3]
# ─────────────────────────────────────────────
h = 86400  #seconds per day
Q = df['discharge_cfs'].values  #ft^3/s
n = len(Q)

#Method 1: Trapezoidal Rule, uses all 365 points (364 intervals)
volume_trap = np.trapezoid(Q, dx=h)  #ft^3



#Method 2: Simpson's Rule, uses all 365 points (364 intervals).
if n % 2 == 0:
    Q_simp = Q[:-1]
    note = f"(used {len(Q_simp)} points; last point dropped for even-interval requirement)"
else:
    Q_simp = Q
    note = f"(used all {len(Q_simp)} points — no adjustment needed)"
volume_simp = simpson(Q_simp, dx=h)  # ft^3


#Unit Conversions: 1 acre-foot = 43,560 ft^3
ACRE_FOOT = 43560  #ft^3 per acre-foot

volume_trap_af = volume_trap / ACRE_FOOT
volume_simp_af = volume_simp / ACRE_FOOT

print("=" * 60)
print("NUMERICAL INTEGRATION RESULTS")
print("=" * 60)
print(f"\nTime step h = 86,400 seconds (1 day)")
print(f"Number of data points: {n}")
print()
print(f"Method 1: Trapezoidal Rule")
print(f"  Total volume = {volume_trap:.4e} ft³")
print(f"               = {volume_trap_af:,.0f} acre-feet")
print()
print(f"Method 2: Simpson's Rule {note}")
print(f"  Total volume = {volume_simp:.4e} ft³")
print(f"               = {volume_simp_af:,.0f} acre-feet")
print()
print(f"Difference (Simp - Trap) = {abs(volume_simp - volume_trap):.4e} ft³")
print(f"Relative difference      = {abs(volume_simp - volume_trap)/volume_simp * 100:.6f}%")
print()

# ─────────────────────────────────────────────
#Error Analysis
#Trapezoidal error bound:
#|E_T| <= (b-a) * h^2 / 12 * max|f''(x)|
#Simpson's error bound:
#|E_S| <= (b-a) * h^4 / 180 * max|f''''(x)|
# ─────────────────────────────────────────────

#estimate second derivative using finite differences
#f''(x_i) ≈ (f(x_{i+1}) - 2f(x_i) + f(x_{i-1})) / h^2
f2 = np.abs(np.diff(Q, n=2)) / (h**2)
max_f2 = np.max(f2)
b_minus_a = (n - 1) * h  #total time in seconds

error_trap = (b_minus_a * h**2 / 12) * max_f2
error_trap_af = error_trap / ACRE_FOOT

#estimate fourth derivative using finite differences
#f''''(x_i) ≈ (f_{i+2} - 4f_{i+1} + 6f_i - 4f_{i-1} + f_{i-2}) / h^4
f4 = np.abs(np.diff(Q, n=4)) / (h**4)
max_f4 = np.max(f4)
error_simp = (b_minus_a * h**4 / 180) * max_f4
error_simp_af = error_simp / ACRE_FOOT

print("=" * 60)
print("MARGIN OF ERROR")
print("=" * 60)
print()
print(f"Trapezoidal Rule Error Bound:")
print(f"  |E_T| <= (b-a)*h²/12 * max|f''(x)|")
print(f"  max|f''(x)| ≈ {max_f2:.6e} (ft^3/s)/s²")
print(f"  |E_T| <= {error_trap:.4e} ft³")
print(f"          <= {error_trap_af:,.2f} acre-feet")
print()
print(f"Simpson's Rule Error Bound:")
print(f"  |E_S| <= (b-a)*h⁴/180 * max|f''''(x)|")
print(f"  max|f''''(x)| ≈ {max_f4:.6e} (ft^3/s)/s⁴")
print(f"  |E_S| <= {error_simp:.4e} ft³")
print(f"          <= {error_simp_af:,.2f} acre-feet")
print()

#practical error, which is the difference between the two methods
practical_error = abs(volume_simp - volume_trap)
print(f"Practical Error Estimate (|Simpson - Trapezoid|):")
print(f"  = {practical_error:.4e} ft³ = {practical_error/ACRE_FOOT:,.2f} acre-feet")
print()

#summary table of the volume estimates and error bounds
print("=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Method':<20} {'Volume (ft³)':>18} {'Volume (acre-ft)':>18} {'Error Bound (ft³)':>20}")
print("-" * 78)
print(f"{'Trapezoidal':<20} {volume_trap:>18.4e} {volume_trap_af:>18,.0f} {error_trap:>20.4e}")
print(f"{'Simpsons':<20} {volume_simp:>18.4e} {volume_simp_af:>18,.0f} {error_simp:>20.4e}")
print()
print("All results computed using Python (NumPy, SciPy).")
print("Data source: USGS NWIS, Site 14105700, Columbia River at The Dalles, OR")