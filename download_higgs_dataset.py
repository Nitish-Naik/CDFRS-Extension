import urllib.request
import os
import gzip
import shutil

def download_higgs_dataset():
    """
    Downloads the HIGGS dataset from UCI Machine Learning Repository.
    The dataset is approximately 2.6 GB compressed (7.5 GB uncompressed).
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    gz_filename = "HIGGS.csv.gz"
    csv_filename = "HIGGS.csv"
    
    # Check if already downloaded
    if os.path.exists(csv_filename):
        print(f"{csv_filename} already exists. Skipping download.")
        return csv_filename
    
    if os.path.exists(gz_filename):
        print(f"{gz_filename} exists. Skipping download step.")
    else:
        print(f"Downloading HIGGS dataset from {url}")
        print("This may take a while (approximately 2.6 GB)...")
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end='')
        
        try:
            urllib.request.urlretrieve(url, gz_filename, reporthook=report_progress)
            print("\nDownload complete!")
        except Exception as e:
            print(f"\nError downloading file: {e}")
            return None
    
    # Decompress the file
    if not os.path.exists(csv_filename):
        print(f"\nDecompressing {gz_filename}...")
        try:
            with gzip.open(gz_filename, 'rb') as f_in:
                with open(csv_filename, 'wb') as f_out:
                    # Decompress in chunks to show progress
                    chunk_size = 1024 * 1024  # 1 MB chunks
                    total_read = 0
                    while True:
                        chunk = f_in.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        total_read += len(chunk)
                        print(f"\rDecompressed: {total_read / (1024 * 1024):.1f} MB", end='')
            print(f"\nDecompression complete! File saved as {csv_filename}")
        except Exception as e:
            print(f"\nError decompressing file: {e}")
            return None
    
    # Optionally remove the .gz file to save space
    if os.path.exists(gz_filename):
        print(f"\nRemoving compressed file {gz_filename} to save space...")
        os.remove(gz_filename)
    
    return csv_filename

if __name__ == "__main__":
    print("HIGGS Dataset Downloader")
    print("=" * 50)
    result = download_higgs_dataset()
    if result:
        print(f"\n✓ Dataset ready at: {result}")
        # Show file size
        size_bytes = os.path.getsize(result)
        size_gb = size_bytes / (1024 ** 3)
        print(f"✓ File size: {size_gb:.2f} GB")
        
        # Count lines (approximately)
        print("\nCounting rows (this may take a moment)...")
        with open(result, 'r') as f:
            num_lines = sum(1 for _ in f)
        print(f"✓ Total rows: {num_lines:,}")
    else:
        print("\n✗ Failed to download dataset")
