#!/usr/bin/env python3
"""
Script to read HDF5 metadata and clearly identify Zg dataset dimensions.
Determines which dimension is time and which is height/range.
"""

import h5py
import numpy as np
from pathlib import Path

# Path to HDF5 file
data_dir = "/data/obs/site/jue/joyrad35/2025/06/06"
file_list = list(Path(data_dir).glob("*tower.znc"))

if not file_list:
    print("ERROR: No tower.znc files found!")
    exit(1)

file_path = str(file_list[0])
print(f"Analyzing: {file_path}\n")

with h5py.File(file_path, "r") as f:
    print("=" * 80)
    print("ZG DATASET ANALYSIS")
    print("=" * 80)
    
    # Find Zg dataset
    zg_key = "Zg" if "Zg" in f else ("zg" if "zg" in f else None)
    
    if not zg_key:
        print("ERROR: Zg dataset not found!")
        exit(1)
    
    zg = f[zg_key]
    print(f"\nDataset name: {zg_key}")
    print(f"Shape: {zg.shape}")
    print(f"  Dimension 0 (axis 0): {zg.shape[0]} elements")
    print(f"  Dimension 1 (axis 1): {zg.shape[1]} elements")
    
    # Check related datasets to identify dimensions
    print("\n" + "-" * 80)
    print("RELATED DATASETS FOR IDENTIFICATION")
    print("-" * 80)
    
    datasets_to_check = ["time", "range", "height", "azi", "elv"]
    
    for ds_name in datasets_to_check:
        if ds_name in f:
            ds = f[ds_name]
            print(f"\n{ds_name}: shape = {ds.shape}")
            
            # Try to read first few values
            try:
                if ds.shape[0] <= 10:
                    vals = ds[:]
                else:
                    vals = ds[:5]
                print(f"  First values: {vals}")
            except:
                print(f"  [Could not read values directly]")
            
            # Check attributes
            if "long_name" in ds.attrs:
                try:
                    ln = ds.attrs["long_name"]
                    if hasattr(ln, '__iter__') and not isinstance(ln, str):
                        print(f"  long_name: {[x for x in ln]}")
                    else:
                        print(f"  long_name: {ln}")
                except:
                    pass
            
            if "units" in ds.attrs:
                try:
                    units = ds.attrs["units"]
                    if hasattr(units, '__iter__') and not isinstance(units, str):
                        print(f"  units: {[x for x in units]}")
                    else:
                        print(f"  units: {units}")
                except:
                    pass
    
    # Dimension identification logic
    print("\n" + "-" * 80)
    print("DIMENSION IDENTIFICATION")
    print("-" * 80)
    
    time_size = f["time"].shape[0] if "time" in f else None
    range_size = f["range"].shape[0] if "range" in f else None
    
    print(f"\nZg shape: {zg.shape}")
    print(f"Time dataset size: {time_size}")
    print(f"Range dataset size: {range_size}")
    
    # Determine which dimension is which
    if time_size and range_size:
        if zg.shape[0] == time_size and zg.shape[1] == range_size:
            print(f"\n✓ CONFIRMED:")
            print(f"  Dimension 0 (axis 0, size {zg.shape[0]}): TIME")
            print(f"  Dimension 1 (axis 1, size {zg.shape[1]}): HEIGHT/RANGE")
            print(f"\nZg[t, h] format: time-indexed rows, height-indexed columns")
            
        elif zg.shape[0] == range_size and zg.shape[1] == time_size:
            print(f"\n✓ CONFIRMED:")
            print(f"  Dimension 0 (axis 0, size {zg.shape[0]}): HEIGHT/RANGE")
            print(f"  Dimension 1 (axis 1, size {zg.shape[1]}): TIME")
            print(f"\nZg[h, t] format: height-indexed rows, time-indexed columns")
        else:
            print(f"\n⚠ WARNING: Zg dimensions don't match time/range datasets!")
    
    # Print full metadata for Zg
    print("\n" + "-" * 80)
    print("ZG ATTRIBUTES")
    print("-" * 80)
    
    for attr_name in zg.attrs:
        try:
            val = zg.attrs[attr_name]
            if isinstance(val, np.ndarray):
                if val.size <= 5:
                    print(f"  {attr_name}: {val}")
                else:
                    print(f"  {attr_name}: [array of {val.size} elements]")
            else:
                print(f"  {attr_name}: {val}")
        except Exception as e:
            print(f"  {attr_name}: [unreadable: {type(e).__name__}]")

print("\n" + "=" * 80)
