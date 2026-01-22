import os
import csv

def split_higgs_dataset(input_file="HIGGS.csv", num_splits=100, output_prefix="HIGGS_part"):
    """
    Splits the HIGGS.csv file into equal-sized parts.
    
    Args:
        input_file: Path to the HIGGS.csv file
        num_splits: Number of files to split into (default: 100)
        output_prefix: Prefix for output files
    """
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return False
    
    print(f"Splitting {input_file} into {num_splits} files...")
    print("Step 1: Counting total rows...")
    
    # Count total rows
    total_rows = 0
    with open(input_file, 'r') as f:
        for _ in f:
            total_rows += 1
    
    print(f"Total rows: {total_rows:,}")
    
    # Calculate rows per file
    rows_per_file = total_rows // num_splits
    remainder = total_rows % num_splits
    
    print(f"Rows per file: ~{rows_per_file:,}")
    print(f"\nStep 2: Splitting files...")
    
    # Create output directory
    output_dir = "higgs_100_splits"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the file
    with open(input_file, 'r') as f_in:
        reader = csv.reader(f_in)
        
        current_file_num = 1
        current_row_count = 0
        output_file = None
        writer = None
        
        for row_num, row in enumerate(reader, 1):
            # Open new file when needed
            if current_row_count == 0:
                if output_file:
                    output_file.close()
                
                # Determine how many rows for this file
                # First 'remainder' files get one extra row
                file_size = rows_per_file + (1 if current_file_num <= remainder else 0)
                
                output_filename = os.path.join(output_dir, f"{output_prefix}_{current_file_num:03d}.csv")
                output_file = open(output_filename, 'w', newline='')
                writer = csv.writer(output_file)
                
                print(f"Creating {output_filename} ({file_size:,} rows)...")
            
            # Write row
            writer.writerow(row)
            current_row_count += 1
            
            # Show progress
            if row_num % 100000 == 0:
                print(f"  Progress: {row_num:,} / {total_rows:,} rows ({row_num*100/total_rows:.1f}%)")
            
            # Check if current file is complete
            file_size = rows_per_file + (1 if current_file_num <= remainder else 0)
            if current_row_count >= file_size:
                current_file_num += 1
                current_row_count = 0
        
        # Close last file
        if output_file:
            output_file.close()
    
    print(f"\n✓ Successfully split {input_file} into {num_splits} files!")
    print(f"✓ Output directory: {output_dir}/")
    
    # List first 5 and last 5 files with sizes
    print("\nCreated files (showing first 5 and last 5):")
    for i in list(range(1, 6)) + list(range(num_splits - 4, num_splits + 1)):
        filename = os.path.join(output_dir, f"{output_prefix}_{i:03d}.csv")
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"  {filename} ({size_mb:.1f} MB)")
    
    return True

if __name__ == "__main__":
    print("HIGGS Dataset Splitter - 100 Files")
    print("=" * 50)
    
    # Split into 100 files
    success = split_higgs_dataset(
        input_file="HIGGS.csv",
        num_splits=100,
        output_prefix="HIGGS_part"
    )
    
    if success:
        print("\n✓ Done!")
    else:
        print("\n✗ Failed to split dataset")
