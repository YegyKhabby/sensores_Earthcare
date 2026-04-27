#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check maximum Ze for the first 8 gates (gates 0-7) separately
for cloudnet categorize data in May, June, July 2025.

Source: /data/obs/site/jue/cloudnet/clu_processing/categorize/2025/
Variable: Z (reflectivity, dBz)
Heights: gates 0-7 at ~255 m to ~507 m AGL

Output: cloudnet_max_ze_gates_0to7_may_jun_jul_2025.csv
  Columns: date, gate_0_max_ze, gate_1_max_ze, ..., gate_7_max_ze
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import csv

import numpy as np
import netCDF4 as nc
import pandas as pd

OUT_DIR = Path('/work/yegy_project/Feb2026_new/output/cloudnet')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLOUDNET_BASE = Path('/data/obs/site/jue/cloudnet/clu_processing/categorize/2025')
MONTHS = [5, 6, 7]  # May, June, July


def get_cloudnet_files(month: int) -> list[Path]:
    """Return sorted list of categorize NC files for a given month."""
    month_str = f'{month:02d}'
    files = sorted(CLOUDNET_BASE.glob(f'202505??_*' if month == 5 
                                       else f'202506??_*' if month == 6 
                                       else f'202507??_*'))
    return [f for f in files if f.name.endswith('_categorize.nc')]


def process_file(nc_path: Path, n_gates: int = 8) -> dict | None:
    """
    Extract max Ze for each of the first n_gates.
    Returns dict: {date_str, gate_0_max, gate_1_max, ..., gate_N_max}
    or None if file cannot be processed.
    """
    try:
        ds = nc.Dataset(str(nc_path))
        
        # Extract date from filename: 20250501_juelich_categorize.nc
        date_str = nc_path.name[:8]  # YYYYMMDD
        
        Z = np.array(ds.variables['Z'][:], dtype=float)
        ds.close()
        
        # Z shape: (time, height)
        # We want max over time dimension for each gate (height index 0-7)
        result = {'date': date_str}
        for gate in range(min(n_gates, Z.shape[1])):
            # Get column for this gate, ignore fill values
            z_gate = Z[:, gate]
            z_gate = z_gate[z_gate < 1e10]  # filter fill values
            
            if len(z_gate) > 0:
                max_z = float(np.nanmax(z_gate))
            else:
                max_z = np.nan
            
            result[f'gate_{gate}_max_ze'] = max_z
        
        return result
    
    except Exception as e:
        print(f'Error processing {nc_path.name}: {e}')
        return None


def main():
    rows = []
    
    for month in MONTHS:
        files = get_cloudnet_files(month)
        print(f'Month {month:02d}: {len(files)} files')
        
        for fpath in files:
            rec = process_file(fpath)
            if rec is not None:
                rows.append(rec)
                if len(rows) % 10 == 0:
                    print(f'  processed {len(rows)} files')
    
    if not rows:
        print('No files processed.')
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Output CSV
    out_csv = OUT_DIR / 'cloudnet_max_ze_gates_0to7_may_jun_jul_2025.csv'
    df.to_csv(out_csv, index=False, float_format='%.2f')
    
    print(f'\nProcessed {len(rows)} days')
    print(f'Output: {out_csv}')
    print(f'\nFirst 5 rows:')
    print(df.head())
    print(f'\nStatistics by gate:')
    for gate in range(8):
        col = f'gate_{gate}_max_ze'
        if col in df.columns:
            print(f'  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}')


if __name__ == '__main__':
    main()
