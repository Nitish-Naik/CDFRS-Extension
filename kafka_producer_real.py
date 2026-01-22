import time
import json
import csv
from kafka import KafkaProducer
import os

def stream_from_higgs_file(producer, topic_name, filename="2000_higgs.csv", events_per_second=10):
    """
    Streams data from the HIGGS CSV file to Kafka.
    
    Args:
        producer: KafkaProducer instance
        topic_name: Name of the Kafka topic
        filename: Path to 2000_higgs.csv file
        events_per_second: Rate of data streaming
    """
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        print("Please run 'python download_2000_higgs_dataset.py' first to download the dataset.")
        return False
    
    print(f"Starting to stream from {filename}...")
    sleep_time = 1.0 / events_per_second
    
    try:
        count = 0
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
                    
                    if count % 100 == 0:
                        print(f"\rStreamed {count:,} records...", end='', flush=True)
                    
                    time.sleep(sleep_time)
                    
                except (ValueError, IndexError) as e:
                    # Skip malformed rows
                    continue
                    
    except KeyboardInterrupt:
        print(f"\n\nStopped after streaming {count:,} records.")
        return True
    
    print(f"\n\nFinished streaming all {count:,} records from the dataset.")
    return True

def main():
    topic_name = "2000_higgs_stream"
    bootstrap_servers = ['localhost:9092']
    
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
    success = stream_from_higgs_file(producer, topic_name, "2000_higgs.csv", events_per_second=10)
    
    if success:
        producer.flush()
        print("All messages sent successfully.")
    
    producer.close()

if __name__ == "__main__":
    main()
