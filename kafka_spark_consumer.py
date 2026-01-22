from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
import argparse
import os

# Define the schema matching HIGGS dataset (label + 28 features)
schema = StructType([
    StructField("label", DoubleType(), True)
] + [StructField(f"x{i}", DoubleType(), True) for i in range(1, 29)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sink", choices=["console", "file"], default="console", help="Output sink for the stream")
    parser.add_argument("--outdir", default="stream_output", help="Output directory when using file sink")
    parser.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"), help="Kafka bootstrap servers")
    args = parser.parse_args()

    if args.sink == "file":
        os.makedirs(args.outdir, exist_ok=True)
    # Initialize Spark Session with Kafka package
    # Note: When running this, make sure to include the package:
    # spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 kafka_spark_consumer.py
    # (Adjust version 3.5.0 to match your local Spark version)
    
    spark = SparkSession.builder \
        .appName("HiggsKafkaStream") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", args.bootstrap) \
        .option("subscribe", "2000_higgs_stream") \
        .option("startingOffsets", "earliest") \
        .load()

    # Parse JSON data
    # Kafka value is binary, cast to string then parse
    parsed_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    # Print schema to console to verify
    parsed_df.printSchema()

    # Write to chosen sink to verify flow
    writer = parsed_df.writeStream.outputMode("append")

    if args.sink == "console":
        query = writer.format("console").option("truncate", "false").start()
    else:
        # Write JSON files and use a checkpoint for fault-tolerance
        checkpoint = os.path.join(args.outdir, "_checkpoint")
        query = writer.format("json") \
            .option("path", args.outdir) \
            .option("checkpointLocation", checkpoint) \
            .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()
