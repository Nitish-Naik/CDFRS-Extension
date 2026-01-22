import json
from kafka import KafkaConsumer

def main():
    topic_name = "2000_higgs_stream"
    bootstrap_servers = ['localhost:9092']
    
    print(f"Connecting to Kafka at {bootstrap_servers}...")
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',  
            enable_auto_commit=True,
            group_id='higgs-consumer-group'
        )
        print("Connected successfully.")
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        return

    print(f"Listening to topic '{topic_name}'...")
    print("Press Ctrl+C to stop.\n")
    
    try:
        count = 0
        for message in consumer:
            data = message.value
            count += 1
            
            if count % 10 == 0:
                print(f"\n--- Record #{count} ---")
                print(f"Label: {data.get('label')}")
                print(f"Features: x1={data.get('x1'):.4f}, x2={data.get('x2'):.4f}, ..., x28={data.get('x28'):.4f}")
            
            if count % 100 == 0:
                print(f"\rProcessed {count:,} records...", end='', flush=True)
                
    except KeyboardInterrupt:
        print(f"\n\nStopped. Total records processed: {count:,}")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
