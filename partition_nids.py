"""
NF-UQ-NIDS-v2 Dataset Partitioner
===================================
Partitions the NF-UQ-NIDS-v2 dataset into streaming-ready windows that
preserve natural concept drift boundaries via the `Dataset` source column.

Strategy
--------
1. PRIMARY:  Partition by `Dataset` column (7 sub-datasets = 7 natural drift
             boundaries, each with a distinct attack distribution).
2. SECONDARY: Within each Dataset shard, sub-partition into fixed-size
             time-ordered chunks so each window is manageable for YARN.
3. OUTPUT:   HDFS parquet partitions, one directory per window, ordered
             chronologically / by dataset source, ready for sequential
             streaming in the ensemble pipeline.

NF-UQ-NIDS-v2 columns of interest
-----------------------------------
  Label        — string ("Benign" / attack name)
  label        — binary int (0 benign, 1 attack) [must be derived]
  Dataset      — source sub-dataset identifier  ← drift boundary
  IPV4_SRC_ADDR, IPV4_DST_ADDR — raw IPs → converted to /24 subnet ints
  L4_SRC_PORT, L4_DST_PORT, PROTOCOL, ...
  (43 flow features total after dropping IP strings)
"""

import os

os.environ["SPARK_LOCAL_IP"] = "127.0.1.1"
os.environ["HADOOP_CONF_DIR"] = "/usr/local/hadoop/etc/hadoop"
os.environ["YARN_CONF_DIR"] = "/usr/local/hadoop/etc/hadoop"

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, IntegerType
from pyspark.sql.window import Window

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HDFS_INPUT  = "hdfs://localhost:9000/data/nids/NF-UQ-NIDS-v2.csv"
HDFS_OUTPUT = "hdfs://localhost:9000/data/nids_windows"

# Hard ceiling on rows per window.
# This is the primary control knob — it guarantees no window ever exceeds
# this size regardless of which sub-dataset it comes from.
# Larger sub-datasets (e.g. ToN-IoT at ~37M rows) are split into as many
# windows as needed to stay under this cap.
#
# Sizing guidance for a 2-executor YARN cluster:
#   3_000_000 rows × 41 float64 cols ≈ 984 MB raw; Spark adds ~2-3× overhead
#   during shuffles/joins → ~2-3 GB executor memory per window.
#   If you still OOM, halve this value.
MAX_ROWS_PER_WINDOW = 3_000_000

# Hard floor: windows smaller than this are merged upward.
# Needed so tiny sub-datasets don't produce windows too small for SWD.
MIN_ROWS_PER_WINDOW = 200_000

# Derived: target number of sub-windows per dataset is computed dynamically
# from (shard_total / MAX_ROWS_PER_WINDOW), so every window is bounded above
# AND below. SUBWINDOWS_PER_DATASET is no longer a fixed constant.

# Columns to DROP from the feature set.
# Raw IP strings are high-cardinality strings — we keep subnet aggregates instead.
DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack", "Dataset"]

# Columns to KEEP for SWD drift detection (excludes label).
# These 39 numeric flow features are the SWD feature set.
SWD_FEATURE_EXCLUDE = {"label", "src_subnet", "dst_subnet"}  # exclude from SWD if desired

# ─────────────────────────────────────────────────────────────────────────────
# IP → /24 Subnet Helper
# ─────────────────────────────────────────────────────────────────────────────

def ip_to_subnet24(ip_col: str) -> F.Column:
    """
    Convert a dotted-decimal IPv4 string column to a /24 subnet integer.
    E.g. "192.168.1.42" → 192*65536 + 168*256 + 1 = 12625921
    This collapses host variation while retaining network-level structure.
    """
    parts = F.split(F.col(ip_col), r"\.")
    return (
        parts.getItem(0).cast(LongType()) * 65536
        + parts.getItem(1).cast(LongType()) * 256
        + parts.getItem(2).cast(LongType())
    ).cast(LongType())


# ─────────────────────────────────────────────────────────────────────────────
# Schema Prep
# ─────────────────────────────────────────────────────────────────────────────

def prepare_schema(df: DataFrame) -> DataFrame:
    """
    1. Derive binary `label` from the string `Label` column (0=Benign, 1=Attack).
    2. Convert IP strings to /24 subnet integers.
    3. Cast all remaining feature columns to float (handles nulls gracefully).
    4. Drop raw string columns.
    """
    # Binary label
    df = df.withColumn(
        "label",
        F.when(F.lower(F.col("Label")) == "benign", 0).otherwise(1).cast(IntegerType()),
    )

    # /24 subnet aggregates
    df = df.withColumn("src_subnet", ip_to_subnet24("IPV4_SRC_ADDR"))
    df = df.withColumn("dst_subnet", ip_to_subnet24("IPV4_DST_ADDR"))

    # Cast all numeric-looking columns to double (skip string cols)
    string_cols = {"Label", "Attack", "Dataset", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"}
    for c in df.columns:
        if c not in string_cols and c not in {"label", "src_subnet", "dst_subnet"}:
            df = df.withColumn(c, F.col(c).cast("double"))

    # Drop raw strings
    df = df.drop(*[c for c in DROP_COLS if c in df.columns])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-Level Partitioner  (primary drift boundaries)
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_order(df: DataFrame) -> list:
    """
    Returns the list of Dataset source names ordered by row count (ascending),
    so smaller datasets come first and the pipeline warms up on lighter windows.

    Ordering rationale: starting small → large ensures the ensemble has a
    stable reference before it sees the high-volume shards, which reduces
    false-positive drift flags from sample-size asymmetry.
    """
    counts = (
        df.groupBy("Dataset")
        .agg(F.count("*").alias("cnt"))
        .orderBy("cnt")
        .collect()
    )
    order = [row["Dataset"] for row in counts]
    print("[Partitioner] Dataset order (asc row count):")
    for i, (row, name) in enumerate(zip(counts, order)):
        print(f"  [{i}] {name}  ({row['cnt']:,} rows)")
    return order


# ─────────────────────────────────────────────────────────────────────────────
# Sub-Window Splitter  (secondary partitioning within each Dataset shard)
# ─────────────────────────────────────────────────────────────────────────────

def write_subwindows(
    df: DataFrame,
    dataset_name: str,
    dataset_idx: int,
    window_counter: list,
    max_rows: int = MAX_ROWS_PER_WINDOW,
    min_rows: int = MIN_ROWS_PER_WINDOW,
) -> None:
    """
    Split a Dataset shard into windows bounded by [min_rows, max_rows].

    The number of windows is derived from the shard size and max_rows:
        n = ceil(total / max_rows)   (at least 1)

    This guarantees no window ever exceeds max_rows regardless of which
    sub-dataset is being processed. Large sub-datasets (ToN-IoT ~37M rows)
    are split into more windows; small ones produce fewer but always stay
    above min_rows (merged to 1 window if needed).

    window_counter[0] is incremented per window written.
    """
    total = df.count()

    # Compute window count from ceiling division, then enforce min_rows floor
    import math as _math
    n_raw = _math.ceil(total / max_rows)
    effective_n = max(1, n_raw)

    # If splitting would produce windows smaller than min_rows, reduce count
    if effective_n > 1 and (total // effective_n) < min_rows:
        effective_n = max(1, total // min_rows)

    rows_per_window = total // effective_n

    print(
        f"[Partitioner] Dataset '{dataset_name}' → {total:,} rows → "
        f"{effective_n} window(s) of ~{rows_per_window:,} rows each "
        f"(cap={max_rows:,})"
    )

    df_idx = df.withColumn(
        "_subwindow_id",
        (F.monotonically_increasing_id() % effective_n).cast(IntegerType()),
    )

    for sw in range(effective_n):
        global_window_idx = window_counter[0]
        out_path = f"{HDFS_OUTPUT}/window_{global_window_idx:04d}"

        shard = df_idx.filter(F.col("_subwindow_id") == sw).drop("_subwindow_id")
        shard.coalesce(8).write.mode("overwrite").parquet(out_path)

        actual_count = shard.count()
        print(
            f"  → window_{global_window_idx:04d}  "
            f"(dataset_idx={dataset_idx}, subwindow={sw})  "
            f"rows={actual_count:,}  path={out_path}"
        )
        window_counter[0] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Write Manifest
# ─────────────────────────────────────────────────────────────────────────────

def write_manifest(dataset_order: list, window_counter: list, spark: SparkSession) -> None:
    """
    Write a small manifest CSV to HDFS so the ensemble pipeline can discover
    window paths without globbing the entire output directory.
    """
    total_windows = window_counter[0]
    rows = [(f"{HDFS_OUTPUT}/window_{i:04d}", i) for i in range(total_windows)]
    manifest_df = spark.createDataFrame(rows, ["path", "window_idx"])
    manifest_path = f"{HDFS_OUTPUT}/manifest"
    manifest_df.coalesce(1).write.mode("overwrite").option("header", True).csv(manifest_path)
    print(f"[Partitioner] Manifest written → {manifest_path}  ({total_windows} windows)")


# ─────────────────────────────────────────────────────────────────────────────
# Class Distribution Reporter
# ─────────────────────────────────────────────────────────────────────────────

def report_class_distribution(df: DataFrame, dataset_name: str) -> None:
    """Print label balance for a shard — useful for spotting extreme imbalance."""
    counts = df.groupBy("label").count().collect()
    total = sum(r["count"] for r in counts)
    for r in sorted(counts, key=lambda x: x["label"]):
        pct = 100.0 * r["count"] / total if total > 0 else 0
        print(f"    label={r['label']}  count={r['count']:,}  ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("NIDS_Partitioner")
        .master("yarn")
        .config(
            "spark.yarn.appMasterEnv.PYSPARK_PYTHON",
            "/home/cwc/Major_Project/major-project/bin/python3",
        )
        .config(
            "spark.executorEnv.PYSPARK_PYTHON",
            "/home/cwc/Major_Project/major-project/bin/python3",
        )
        # Reading ~76M rows — give the shuffle plenty of room
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


    # ── 1. Load raw CSV ───────────────────────────────────────────────────────
    print(f"[Partitioner] Reading raw CSV from {HDFS_INPUT} ...")
    raw_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(HDFS_INPUT)
    )
    print(f"[Partitioner] Raw schema ({len(raw_df.columns)} cols):")
    raw_df.printSchema()

    # ── 2. Single-pass schema prep — keep Dataset for partitioning ───────────
    # Label is already a 0/1 integer per inferSchema — alias directly.
    # Attack is a string attack-type name — dropped along with raw IP strings.
    # ── 2. Single-pass schema prep — keep Dataset for partitioning ───────────
    print("[Partitioner] Preparing schema ...")

    STRING_COLS = {"Label", "Attack", "Dataset", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"}

    df_with_tag = (
        raw_df
        .withColumnRenamed("Label", "label") # Rename it instead of creating a duplicate
        .withColumn("label", F.col("label").cast(IntegerType())) # Ensure it's an int
        .withColumn("src_subnet", ip_to_subnet24("IPV4_SRC_ADDR"))
        .withColumn("dst_subnet", ip_to_subnet24("IPV4_DST_ADDR"))
    )

    # Cast every remaining non-string column to double for uniformity
    for c in df_with_tag.columns:
        if c not in STRING_COLS and c not in {"label", "src_subnet", "dst_subnet"}:
            df_with_tag = df_with_tag.withColumn(c, F.col(c).cast("double"))

    # Drop all raw string cols EXCEPT Dataset (needed for shard partitioning)
    # 🚨 REMOVED "Label" from this list so the "label" column isn't deleted
    df_with_tag = df_with_tag.drop("Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR")

    df_with_tag = df_with_tag.cache()
    total_rows = df_with_tag.count()   # single materialization pass
    print(f"[Partitioner] Total rows after prep: {total_rows:,}")
    print(f"[Partitioner] Feature columns: {[c for c in df_with_tag.columns if c != 'Dataset']}")

    # ── 3. Ordered Dataset list (natural drift boundaries) ───────────────────
    dataset_order = get_dataset_order(df_with_tag)

    # ── 4. Per-dataset sub-windowing ─────────────────────────────────────────
    window_counter = [0]

    for ds_idx, ds_name in enumerate(dataset_order):
        print(f"\n[Partitioner] ── Processing Dataset [{ds_idx}]: '{ds_name}' ──")
        shard = df_with_tag.filter(F.col("Dataset") == ds_name).drop("Dataset")
        report_class_distribution(shard, ds_name)
        write_subwindows(shard, ds_name, ds_idx, window_counter,
                         MAX_ROWS_PER_WINDOW, MIN_ROWS_PER_WINDOW)

    # ── 5. Manifest ───────────────────────────────────────────────────────────
    write_manifest(dataset_order, window_counter, spark)

    df_with_tag.unpersist()

    print(f"\n[Partitioner] Done. {window_counter[0]} windows written to {HDFS_OUTPUT}")
    spark.stop()