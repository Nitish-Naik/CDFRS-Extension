import os
import glob

# Get all split files from the 100-file split
files = sorted(glob.glob('higgs_100_splits/HIGGS_part_*.csv'))

if not files:
    print("No split files found in higgs_100_splits/!")
else:
    print(f"\nFound {len(files)} split files in higgs_100_splits/:\n")
    print(f"{'File Name':<25} {'Size (MB)':>12} {'Size (GB)':>12}")
    print("-" * 52)
    
    total_size = 0
    for f in files:
        size_bytes = os.path.getsize(f)
        size_mb = size_bytes / (1024 ** 2)
        size_gb = size_bytes / (1024 ** 3)
        total_size += size_bytes
        
        print(f"{os.path.basename(f):<25} {size_mb:>12.2f} {size_gb:>12.3f}")
    
    # Show total
    total_mb = total_size / (1024 ** 2)
    total_gb = total_size / (1024 ** 3)
    print("-" * 52)
    print(f"{'TOTAL':<25} {total_mb:>12.2f} {total_gb:>12.3f}")
