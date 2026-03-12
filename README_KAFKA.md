# Kafka Streaming Setup for CDFRS POC

This setup provides a Kafka streaming service compatible with the HIGGS classification dataset used in `cdfrs_poc.ipynb`.

## components
1. **Kafka (KRaft mode)**: A modern setup without Zookeeper using `bitnami/kafka`.
2. **Kafka UI**: Web interface to view topics and messages at [http://localhost:8080](http://localhost:8080).
3. **Producer**: `kafka_producer_real.py` streams HIGGS classification data.
4. **Consumer**: `kafka_spark_consumer.py` demonstrates reading the stream with PySpark.

## Prerequisites
- Docker & Docker Compose
- Python 3.x
- Java (for Spark)

## Setup and Run

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Kafka
Start the services in the background:
```bash
docker-compose up -d
```
Check if it's running:
```bash
docker-compose ps
```
Wait a few seconds for Kafka to initialize. You can verify it's working by visiting [http://localhost:8080](http://localhost:8080) in your browser.

### 3. Start the Data Stream (Producer)
Run the producer in a separate terminal.
```bash
python kafka_producer_real.py --events-per-second 0
```

Producer rate control:
- `--events-per-second 0` sends at max throughput (default).
- `--events-per-second 10000` caps the producer at 10k msg/sec.
- `--target-gb <N>` streams until at least N GB payload is sent.

Send at least 1 to 2 GB payload:
- 1 GB:
	`python kafka_producer_real.py --events-per-second 0 --target-gb 1`
- 2 GB:
	`python kafka_producer_real.py --events-per-second 0 --target-gb 2`

### 3.1 Benchmark Maximum Throughput (Recommended)
Use the benchmark helper to estimate your machine-specific Kafka limits:

```bash
python benchmark_kafka_throughput.py --auto-start
```

The script prints:
- Practical max throughput (`acks=1`)
- Raw peak throughput (`acks=0`)
- A recommended producer cap to use with `kafka_producer_real.py`

### 4. Run the Spark Consumer

use this 
```bash 
python kafka_consumer_simple.py

Run the consumer in another terminal to see the data invocation.
**Note**: You may need to adjust the `--packages` version if your Spark version differs from 3.5.0.

```bash
# Check your spark version
pyspark --version

# Run the consumer (example for Spark 3.5.0, Scala 2.12)
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 kafka_spark_consumer.py
```

## Integration with cdfrs_poc
The schema used in the stream matches your notebook:
- `label`: Double (0 or 1)
- `x1` to `x28`: Double (features)

You can adapt the logic in `kafka_spark_consumer.py` to perform the CDFRS sampling logic on micro-batches or simply use it to ingest data into a sink for your existing batch notebook to process.
