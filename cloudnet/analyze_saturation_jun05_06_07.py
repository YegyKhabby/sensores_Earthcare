#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigate Ze saturation on June 6, 2025.
Examine:
  1. Time-height Ze structure (identify when/where saturation occurs)
  2. Quality bits (check for bad data flags)
  3. Z_error, Z_sensitivity (understand uncertainty)
  4. Compare with nearby days (June 5, 7)
"""

from pathlib import Path
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

CLOUDNET_BASE = Path('/data/obs/site/jue/cloudnet/clu_processing/categorize/2025')
OUT_DIR = Path('/work/yegy_project/Feb2026_new/output/cloudnet')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATES = ['20250605', '20250606', '20250607']  # Compare these 3 days
N_GATES = 8


def analyze_file(date_str: str) -> dict:
    """Load and analyze one day's cloudnet file."""
    nc_path = CLOUDNET_BASE / f'{date_str}_juelich_categorize.nc'
    
    try:
        ds = nc.Dataset(str(nc_path))
        
        Z = np.array(ds.variables['Z'][:], dtype=float)  # (time, height)
        time = np.array(ds.variables['time'][:], dtype=float)  # seconds since midnight
        height = np.array(ds.variables['height'][:], dtype=float)  # meters AGL
        
        # Quality bits and uncertainties
        quality_bits = np.array(ds.variables['quality_bits'][:], dtype=int)
        Z_error = np.array(ds.variables['Z_error'][:], dtype=float) if 'Z_error' in ds.variables else None
        Z_sensitivity = np.array(ds.variables['Z_sensitivity'][:], dtype=float) if 'Z_sensitivity' in ds.variables else None
        
        ds.close()
        
        # Filter fill values
        Z_valid = Z.copy()
        Z_valid[Z_valid > 1e10] = np.nan
        
        return {
            'date': date_str,
            'Z': Z_valid,
            'time': time,
            'height': height,
            'quality_bits': quality_bits,
            'Z_error': Z_error,
            'Z_sensitivity': Z_sensitivity,
            'n_timesteps': Z.shape[0],
            'n_heights': Z.shape[1],
        }
    except Exception as e:
        print(f'Error loading {nc_path}: {e}')
        return None


def compute_stats(data: dict, gate: int) -> dict:
    """Compute stats for one gate across all timesteps."""
    Z_gate = data['Z'][:, gate]
    Z_gate = Z_gate[~np.isnan(Z_gate)]
    
    if len(Z_gate) == 0:
        return {}
    
    # Count values at likely saturation threshold (~26 dBz)
    sat_count = np.sum((Z_gate > 25) & (Z_gate < 27))
    
    return {
        'min': np.nanmin(Z_gate),
        'max': np.nanmax(Z_gate),
        'mean': np.nanmean(Z_gate),
        'median': np.nanmedian(Z_gate),
        'std': np.nanstd(Z_gate),
        'count_25to27': sat_count,
        'pct_25to27': 100.0 * sat_count / len(Z_gate) if len(Z_gate) > 0 else 0,
    }


def main():
    print("="*70)
    print("CLOUDNET Ze SATURATION ANALYSIS - June 5, 6, 7 (2025)")
    print("="*70)
    
    datasets = []
    for date_str in DATES:
        data = analyze_file(date_str)
        if data:
            datasets.append(data)
            print(f"\n✓ Loaded {date_str}: {data['n_timesteps']} time steps, {data['n_heights']} height gates")
    
    # Analyze saturation for first 8 gates
    print("\n" + "="*70)
    print("STATISTICS BY GATE (first 8 gates)")
    print("="*70)
    
    for gate in range(N_GATES):
        print(f"\nGate {gate} (height = {datasets[0]['height'][gate]:.1f} m):")
        print(f"{'Date':<12} {'Min':<8} {'Max':<8} {'Mean':<8} {'N_25-27':<10} {'%_25-27':<10}")
        print("-" * 60)
        
        for data in datasets:
            stats = compute_stats(data, gate)
            if stats:
                print(f"{data['date']:<12} "
                      f"{stats['min']:>7.2f} "
                      f"{stats['max']:>7.2f} "
                      f"{stats['mean']:>7.2f} "
                      f"{stats['count_25to27']:>9d} "
                      f"{stats['pct_25to27']:>9.1f}%")
    
    # Identify time periods with high Ze (potential saturation)
    print("\n" + "="*70)
    print("HIGH Ze PERIODS (Ze > 25 dBz)")
    print("="*70)
    
    for data in datasets:
        Z = data['Z']
        # Find timesteps where any of first 8 gates has Ze > 25
        high_ze_times = np.where(np.nanmax(Z[:, :N_GATES], axis=1) > 25)[0]
        
        if len(high_ze_times) > 0:
            # Time is already in hours (from NetCDF)
            time_hours = data['time'][high_ze_times]
            print(f"\n{data['date']}: {len(high_ze_times)} timesteps with Ze > 25 dBz")
            print(f"  Time range: {time_hours[0]:.2f} to {time_hours[-1]:.2f} UTC hours")
            print(f"  Duration: ~{(time_hours[-1] - time_hours[0]):.1f} hours")
        else:
            print(f"\n{data['date']}: No high Ze periods")
    
    # Create heatmaps
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for idx, data in enumerate(datasets):
        ax = axes[idx]
        Z = data['Z'][:, :N_GATES]  # first 8 gates
        time_hours = data['time']  # Already in hours
        
        im = ax.pcolormesh(time_hours, range(N_GATES), Z.T, cmap='viridis', vmin=-10, vmax=40, shading='auto')
        ax.set_ylabel('Gate')
        ax.set_title(f"{data['date']}: Time-Height Ze (first 8 gates)")
        ax.set_yticks(range(N_GATES))
        cbar = plt.colorbar(im, ax=ax, label='Ze [dBz]')
    
    axes[-1].set_xlabel('Time (UTC hours since 00:00)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'ze_saturation_analysis_jun05-07.png', dpi=150)
    print(f"  → {OUT_DIR / 'ze_saturation_analysis_jun05-07.png'}")
    
    # Statistics summary to CSV
    summary = []
    for data in datasets:
        for gate in range(N_GATES):
            stats = compute_stats(data, gate)
            if stats:
                summary.append({
                    'date': data['date'],
                    'gate': gate,
                    'height_m': data['height'][gate],
                    'min_ze': stats['min'],
                    'max_ze': stats['max'],
                    'mean_ze': stats['mean'],
                    'count_25to27': stats['count_25to27'],
                    'pct_25to27': stats['pct_25to27'],
                })
    
    df = pd.DataFrame(summary)
    out_csv = OUT_DIR / 'ze_saturation_analysis_jun05-07.csv'
    df.to_csv(out_csv, index=False, float_format='%.2f')
    print(f"  → {out_csv}")
    
    print("\nDone.")


if __name__ == '__main__':
    main()
