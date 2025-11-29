#!/usr/bin/env python3
"""
Simple utility to view NPZ file contents
Alternative to npzviewer when it has issues
"""

import sys
import numpy as np
import argparse
from pathlib import Path


def view_npz(file_path: str, show_stats: bool = True, show_sample: bool = True, max_items: int = 5):
    """View contents of an NPZ file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    print("=" * 80)
    print(f"NPZ File: {file_path.name}")
    print("=" * 80)
    
    try:
        data = np.load(file_path, allow_pickle=True)
        keys = list(data.keys())
        
        print(f"\nNumber of arrays/items: {len(keys)}")
        print(f"\nKeys:")
        for i, key in enumerate(keys, 1):
            print(f"  {i}. {key}")
        
        if show_stats:
            print(f"\n{'=' * 80}")
            print("Statistics:")
            print(f"{'=' * 80}")
            
            for key in keys[:max_items] if max_items else keys:
                arr = data[key]
                print(f"\nKey: {key}")
                print(f"  Type: {type(arr)}")
                
                if isinstance(arr, np.ndarray):
                    print(f"  Shape: {arr.shape}")
                    print(f"  Dtype: {arr.dtype}")
                    print(f"  Size: {arr.size} elements")
                    print(f"  Memory: {arr.nbytes / 1024:.2f} KB")
                    
                    if arr.size > 0:
                        if arr.dtype == object:
                            print(f"  Note: Object array (may contain Python objects)")
                        else:
                            print(f"  Min: {np.nanmin(arr):.4f}")
                            print(f"  Max: {np.nanmax(arr):.4f}")
                            print(f"  Mean: {np.nanmean(arr):.4f}")
                            if arr.size > 1:
                                print(f"  Std: {np.nanstd(arr):.4f}")
                else:
                    print(f"  Value: {arr}")
        
        if show_sample and len(keys) > 0:
            print(f"\n{'=' * 80}")
            print("Sample Data (first key):")
            print(f"{'=' * 80}")
            first_key = keys[0]
            arr = data[first_key]
            
            if isinstance(arr, np.ndarray):
                print(f"\nKey: {first_key}")
                print(f"Shape: {arr.shape}")
                
                if arr.size <= 100:
                    print(f"\nFull array:")
                    print(arr)
                else:
                    print(f"\nFirst 10 elements:")
                    flat = arr.flatten()
                    print(flat[:10])
                    print(f"\nLast 10 elements:")
                    print(flat[-10:])
        
        # Summary
        print(f"\n{'=' * 80}")
        print("Summary:")
        print(f"{'=' * 80}")
        total_size = sum(data[k].nbytes if isinstance(data[k], np.ndarray) else 0 for k in keys)
        print(f"Total file size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"Total data size: {total_size / 1024 / 1024:.2f} MB")
        print(f"Number of keys: {len(keys)}")
        
    except Exception as e:
        print(f"Error reading NPZ file: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="View NPZ file contents")
    parser.add_argument("file", help="NPZ file to view")
    parser.add_argument("--no-stats", action="store_true", help="Don't show statistics")
    parser.add_argument("--no-sample", action="store_true", help="Don't show sample data")
    parser.add_argument("--max-items", type=int, default=10, help="Max number of items to show stats for")
    
    args = parser.parse_args()
    
    view_npz(
        args.file,
        show_stats=not args.no_stats,
        show_sample=not args.no_sample,
        max_items=args.max_items
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode - show available NPZ files
        import glob
        npz_files = glob.glob("*.npz")
        if npz_files:
            print("Available NPZ files:")
            for i, f in enumerate(npz_files, 1):
                print(f"  {i}. {f}")
            print("\nUsage: python view_npz.py <filename>")
            if len(npz_files) == 1:
                print(f"\nViewing: {npz_files[0]}")
                view_npz(npz_files[0])
        else:
            print("No NPZ files found in current directory")
            print("Usage: python view_npz.py <filename>")
    else:
        main()

