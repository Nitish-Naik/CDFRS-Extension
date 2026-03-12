#!/usr/bin/env python3

"""Benchmark Kafka producer throughput for this machine and project setup."""

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class PerfResult:
    scenario: str
    topic: str
    partitions: int
    acks: int
    records_per_sec: float
    mb_per_sec: float
    avg_latency_ms: float


def run_command(command, check=True):
    """Run a command from project root and return combined output."""
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    combined_output = result.stdout
    if result.stderr:
        combined_output += result.stderr

    if check and result.returncode != 0:
        print("Command failed:", " ".join(command), file=sys.stderr)
        if combined_output.strip():
            print(combined_output.strip(), file=sys.stderr)
        raise RuntimeError("Command execution failed")

    return combined_output


def parse_perf_output(output):
    """Parse kafka-producer-perf-test summary line."""
    pattern = re.compile(
        r"([\d,]+)\s+records sent,\s+([\d.]+)\s+records/sec\s+"
        r"\(([\d.]+)\s+MB/sec\),\s+([\d.]+)\s+ms avg latency"
    )

    for line in reversed(output.splitlines()):
        match = pattern.search(line)
        if match:
            return {
                "records": int(match.group(1).replace(",", "")),
                "records_per_sec": float(match.group(2)),
                "mb_per_sec": float(match.group(3)),
                "avg_latency_ms": float(match.group(4)),
            }

    raise ValueError("Could not parse producer perf output")


def ensure_services_running(auto_start):
    running = run_command(
        ["docker", "compose", "ps", "--services", "--filter", "status=running"],
        check=False,
    )
    active = {line.strip() for line in running.splitlines() if line.strip()}

    if {"kafka", "zookeeper"}.issubset(active):
        return

    if not auto_start:
        raise RuntimeError(
            "Kafka/Zookeeper are not running. Start them with 'docker compose up -d' "
            "or re-run this script with --auto-start."
        )

    print("Starting Kafka stack...")
    run_command(["docker", "compose", "up", "-d", "zookeeper", "kafka"])


def wait_for_kafka(container, bootstrap_server, timeout_sec=180):
    start = time.time()
    while time.time() - start < timeout_sec:
        output = run_command(
            [
                "docker",
                "exec",
                container,
                "kafka-topics",
                "--bootstrap-server",
                bootstrap_server,
                "--list",
            ],
            check=False,
        )

        if "Connection to node" not in output and "Exception" not in output:
            return
        time.sleep(2)

    raise TimeoutError("Kafka did not become ready in time")


def ensure_topic(container, bootstrap_server, topic, partitions):
    run_command(
        [
            "docker",
            "exec",
            container,
            "kafka-topics",
            "--bootstrap-server",
            bootstrap_server,
            "--create",
            "--if-not-exists",
            "--topic",
            topic,
            "--partitions",
            str(partitions),
            "--replication-factor",
            "1",
        ]
    )


def run_perf_test(
    container,
    bootstrap_server,
    topic,
    scenario,
    partitions,
    acks,
    num_records,
    record_size,
):
    ensure_topic(container, bootstrap_server, topic, partitions)
    print(f"Running {scenario}...")

    output = run_command(
        [
            "docker",
            "exec",
            container,
            "kafka-producer-perf-test",
            "--topic",
            topic,
            "--num-records",
            str(num_records),
            "--record-size",
            str(record_size),
            "--throughput",
            "-1",
            "--producer-props",
            f"bootstrap.servers={bootstrap_server}",
            f"acks={acks}",
            "linger.ms=5",
            "batch.size=65536",
            "compression.type=none",
        ]
    )
    parsed = parse_perf_output(output)

    return PerfResult(
        scenario=scenario,
        topic=topic,
        partitions=partitions,
        acks=acks,
        records_per_sec=parsed["records_per_sec"],
        mb_per_sec=parsed["mb_per_sec"],
        avg_latency_ms=parsed["avg_latency_ms"],
    )


def print_results(results, recommendation_factor):
    print("\nBenchmark results")
    print("-" * 86)
    print(
        f"{'Scenario':30} {'Partitions':>10} {'Acks':>6} "
        f"{'Msg/sec':>14} {'MB/sec':>10} {'Avg Lat(ms)':>12}"
    )
    print("-" * 86)
    for result in results:
        print(
            f"{result.scenario:30} {result.partitions:>10} {result.acks:>6} "
            f"{result.records_per_sec:>14.2f} {result.mb_per_sec:>10.2f} "
            f"{result.avg_latency_ms:>12.2f}"
        )
    print("-" * 86)

    practical_results = [result for result in results if result.acks == 1]
    best_practical = max(practical_results, key=lambda item: item.records_per_sec)
    best_raw = max(results, key=lambda item: item.records_per_sec)
    suggested_producer_rate = int(best_practical.records_per_sec * recommendation_factor)

    print("\nSuggested limits")
    print(
        f"- Practical max (acks=1): {best_practical.records_per_sec:.0f} msg/sec "
        f"on '{best_practical.scenario}'"
    )
    print(
        f"- Raw max (fastest): {best_raw.records_per_sec:.0f} msg/sec "
        f"on '{best_raw.scenario}'"
    )
    print(
        f"- Recommended producer cap (~{int(recommendation_factor * 100)}% of practical): "
        f"{suggested_producer_rate} events/sec"
    )
    print(
        "\nUse this in producer:\n"
        f"python kafka_producer_real.py --events-per-second {suggested_producer_rate}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Kafka throughput for this machine")
    parser.add_argument("--container", default="kafka", help="Kafka container name")
    parser.add_argument(
        "--bootstrap-server",
        default="localhost:9092",
        help="Kafka bootstrap server reachable from inside the Kafka container",
    )
    parser.add_argument("--topic-prefix", default="perf_topic", help="Topic prefix")
    parser.add_argument("--num-records", type=int, default=500000, help="Messages per test")
    parser.add_argument("--record-size", type=int, default=512, help="Message size in bytes")
    parser.add_argument(
        "--high-partitions",
        type=int,
        default=6,
        help="Partition count for high-throughput tests",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start Kafka/Zookeeper automatically if not running",
    )
    parser.add_argument(
        "--stop-after",
        action="store_true",
        help="Stop Kafka/Zookeeper after benchmark",
    )
    parser.add_argument(
        "--recommendation-factor",
        type=float,
        default=0.8,
        help="Factor applied to practical max for suggested producer cap",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.recommendation_factor <= 0 or args.recommendation_factor > 1:
        raise ValueError("--recommendation-factor must be > 0 and <= 1")

    ensure_services_running(auto_start=args.auto_start)
    wait_for_kafka(args.container, args.bootstrap_server)

    results = []
    scenarios = [
        ("1 partition, durable", 1, 1),
        (f"{args.high_partitions} partitions, durable", args.high_partitions, 1),
        (f"{args.high_partitions} partitions, raw peak", args.high_partitions, 0),
    ]

    for scenario_name, partitions, acks in scenarios:
        topic = f"{args.topic_prefix}_{partitions}p_a{acks}"
        results.append(
            run_perf_test(
                container=args.container,
                bootstrap_server=args.bootstrap_server,
                topic=topic,
                scenario=scenario_name,
                partitions=partitions,
                acks=acks,
                num_records=args.num_records,
                record_size=args.record_size,
            )
        )

    print_results(results, recommendation_factor=args.recommendation_factor)

    if args.stop_after:
        run_command(["docker", "compose", "stop", "kafka", "zookeeper"], check=False)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TimeoutError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
