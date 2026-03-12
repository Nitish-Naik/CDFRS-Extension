import time
import json
import csv
from kafka import KafkaProducer
import os
import argparse

def stream_from_higgs_file(
    producer,
    topic_name,
    filename="2000_higgs.csv",
    events_per_second=0,
    repeat=False,
    target_bytes=None,
):
    """
    Streams data from the HIGGS CSV file to Kafka.
    
    Args:
        producer: KafkaProducer instance
        topic_name: Name of the Kafka topic
        filename: Path to 2000_higgs.csv file
        events_per_second: Rate of data streaming. Use <= 0 for no throttle.
        repeat: Loop file continuously when True
        target_bytes: Stop after sending at least this many bytes when set
    """
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        print("Please run 'python download_higgs_dataset.py' first to download the dataset.")
        return False
    
    print(f"Starting to stream from {filename}...")
    sleep_time = None
    if events_per_second and events_per_second > 0:
        sleep_time = 1.0 / events_per_second
        print(f"Rate limit enabled: {events_per_second} events/second")
    else:
        print("Rate limit disabled: sending at max throughput")
    if target_bytes is not None:
        print(f"Target payload size: {target_bytes / (1024 ** 3):.2f} GB")
    if repeat:
        print("Repeat mode enabled: file will loop until target is reached or interrupted")
    
    try:
        count = 0
        bytes_sent = 0
        while True:
            sent_in_pass = 0
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # 2000_higgs format: first column is label, remaining 28 are features
                    if len(row) != 29:
                        continue
                    
                    try:
                        label = float(row[0])
                        features = {f"x{i}": float(row[i]) for i in range(1, 29)}
                        
                        record = {"label": label}
                        record.update(features)
                        
                        producer.send(topic_name, record)
                        count += 1
                        sent_in_pass += 1
                        bytes_sent += len(json.dumps(record).encode('utf-8'))
                        
                        if count % 1000 == 0:
                            if target_bytes is not None:
                                progress_pct = min(100.0, (bytes_sent / target_bytes) * 100)
                                print(
                                    f"\rStreamed {count:,} records ({bytes_sent / (1024 ** 3):.2f} GB, {progress_pct:.1f}% target)...",
                                    end='',
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\rStreamed {count:,} records ({bytes_sent / (1024 ** 2):.1f} MB)...",
                                    end='',
                                    flush=True,
                                )

                        if target_bytes is not None and bytes_sent >= target_bytes:
                            print(
                                f"\n\nReached target payload size after {count:,} records "
                                f"({bytes_sent / (1024 ** 3):.2f} GB)."
                            )
                            return True
                        
                        if sleep_time is not None:
                            time.sleep(sleep_time)
                        
                    except (ValueError, IndexError):
                        # Skip malformed rows
                        continue

            if sent_in_pass == 0:
                print("\nNo valid rows found to stream from input file.")
                return False

            if not repeat:
                break
                    
    except KeyboardInterrupt:
        print(
            f"\n\nStopped after streaming {count:,} records "
            f"({bytes_sent / (1024 ** 3):.2f} GB)."
        )
        return True
    
    print(
        f"\n\nFinished streaming {count:,} records from the dataset"
        f"({bytes_sent / (1024 ** 3):.2f} GB)."
    )
    return True

def main():
    parser = argparse.ArgumentParser(description="Stream HIGGS data to Kafka")
    parser.add_argument(
        "--topic",
        default="2000_higgs_stream",
        help="Kafka topic name"
    )
    parser.add_argument(
        "--bootstrap-server",
        default="localhost:9092",
        help="Kafka bootstrap server"
    )
    parser.add_argument(
        "--input-file",
        default="2000_higgs.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--events-per-second",
        type=float,
        default=0,
        help="Streaming rate limit. Use 0 for max throughput (default)."
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Loop the input file continuously until stopped."
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        default=0,
        help="Stop after sending at least this much payload (GB). Enables repeat mode automatically."
    )
    args = parser.parse_args()

    if args.target_gb < 0:
        print("Error: --target-gb cannot be negative.")
        return

    target_bytes = None
    repeat_mode = args.repeat
    if args.target_gb > 0:
        target_bytes = int(args.target_gb * (1024 ** 3))
        repeat_mode = True

    topic_name = args.topic
    bootstrap_servers = [args.bootstrap_server]
    
    print(f"Connecting to Kafka at {bootstrap_servers}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print("Connected successfully.")
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        return

    print(f"Starting stream to topic '{topic_name}'...")
    print("Press Ctrl+C to stop streaming.\n")
    
    # Stream from the actual HIGGS dataset
    success = stream_from_higgs_file(
        producer,
        topic_name,
        args.input_file,
        events_per_second=args.events_per_second,
        repeat=repeat_mode,
        target_bytes=target_bytes,
    )
    
    if success:
        producer.flush()
        print("All messages sent successfully.")
    
    producer.close()

if __name__ == "__main__":
    main()
