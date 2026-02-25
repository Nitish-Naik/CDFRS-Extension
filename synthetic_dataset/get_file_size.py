import os
import argparse
import sys

def get_file_size(filepath):
    """
    Returns the file size in bytes.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return os.path.getsize(filepath)

def human_readable_size(size_bytes):
    """
    Converts bytes to a human-readable format (KB, MB, GB, etc.).
    """
    if size_bytes == 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while size_bytes >= 1024 and index < len(units) - 1:
        size_bytes /= 1024.0
        index += 1
    return f"{size_bytes:.2f} {units[index]}"

def main():
    parser = argparse.ArgumentParser(description="Calculate file size from a given file path.")
    parser.add_argument("filepath", type=str, help="Path to the file")
    parser.add_argument(
        "-hr", "--human-readable",
        action="store_true",
        help="Display size in human-readable format"
    )
    
    args = parser.parse_args()

    try:
        size = get_file_size(args.filepath)
        if args.human_readable:
            print(f"File size: {human_readable_size(size)}")
        else:
            print(f"File size: {size} bytes")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except PermissionError:
        print(f"Permission denied: {args.filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

