"""
CDFRS Sampling — NF-UQ-NIDS-v2 Edition
========================================
Adapted from cdfrs_higgs.py.

Key changes vs HIGGS version
-----------------------------
- Feature columns are the 39 numeric flow features of NF-UQ-NIDS-v2
  (excludes label, src_subnet, dst_subnet which are handled separately).
- Column set is passed in at call time rather than hard-coded as x1..x28.
- Everything else (CDFRS algorithm, KS-distance criterion, HDFS caching)
  is unchanged.
"""

import math
import os
import random

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType

os.environ["SPARK_LOCAL_IP"] = "127.0.1.1"
os.environ["HADOOP_CONF_DIR"] = "/usr/local/hadoop/etc/hadoop"
os.environ["YARN_CONF_DIR"] = "/usr/local/hadoop/etc/hadoop"

# ─────────────────────────────────────────────────────────────────────────────
# NF-UQ-NIDS-v2 feature columns
# (43 total columns minus label, src_subnet, dst_subnet = 40 numeric features)
# Update this list if you add/remove engineered columns.
# ─────────────────────────────────────────────────────────────────────────────
NIDS_FEATURE_COLS = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
    # Engineered subnet features (kept as numeric for SWD)
    "src_subnet", "dst_subnet",
]

# Subset used for SWD drift detection — exclude high-cardinality identifiers
# that dominate projections without carrying distributional drift signal.
SWD_COLS = [c for c in NIDS_FEATURE_COLS if c not in {"src_subnet", "dst_subnet"}]


# ─────────────────────────────────────────────────────────────────────────────
# KS Distance (unchanged from cdfrs_higgs.py)
# ─────────────────────────────────────────────────────────────────────────────

def ks_distance(df1: DataFrame, df2: DataFrame, col_name: str, num_points: int = 50) -> float:
    quantiles = df1.approxQuantile(col_name, [i / (num_points - 1.0) for i in range(num_points)], 0.01)

    n1 = df1.count()
    n2 = df2.count()

    max_diff = 0.0
    for v in quantiles:
        cdf1 = df1.filter(F.col(col_name) <= v).count() / float(n1)
        cdf2 = df2.filter(F.col(col_name) <= v).count() / float(n2)
        diff = abs(cdf1 - cdf2)
        if diff > max_diff:
            max_diff = diff
    return max_diff


# ─────────────────────────────────────────────────────────────────────────────
# CDFRS — NF-UQ-NIDS-v2 edition
# ─────────────────────────────────────────────────────────────────────────────

# Hard cap on rows collected to driver for SWD reference/detection samples.
# At 41 float64 cols: 10k rows ≈ 3.3 MB — trivial for driver heap.
# SWD permutation test subsamples to max_samples=5000 anyway, so collecting
# more than ~2x that buys nothing statistically.
CDFRS_COLLECT_CAP = 10_000


def cdfrs_nids(
    nids_df: DataFrame,
    spark: SparkSession,
    feature_cols: list = None,
    epsilon: float = 0.05,
    alpha: float = 0.10,
    M: int = 50,
) -> DataFrame:
    """
    CDFRS-inspired representative sampling for NF-UQ-NIDS-v2 windows.

    Produces a small, statistically representative subsample of a streaming
    window DataFrame, suitable as input to the SWD permutation test.

    Design rationale
    ----------------
    The full CDFRS algorithm (write blocks to HDFS, cache, KS convergence loop)
    caused two sequential OOM failures on this cluster:
      - Executor OOM (exit 143): cached RDDs from 30+ calls accumulated in
        executor memory and YARN killed the containers.
      - Driver OOM: collecting the full coarse subset D (~200k rows for a 1M
        row window) exhausted the driver JVM heap.

    Both problems share the same root: we were moving too much data. The SWD
    permutation test only ever uses max_samples=5000 rows. CDFRS's theoretical
    sample size K (with epsilon=0.05, alpha=0.10) is ~530 rows. There is no
    reason to collect or cache anything larger than CDFRS_COLLECT_CAP rows.

    Algorithm
    ---------
    1. Compute the CDFRS theoretical sample size K.
    2. Use Spark's .sample() to draw a fraction targeting CDFRS_COLLECT_CAP
       rows from the window — done entirely on workers, no shuffle.
    3. Collect those rows to the driver (guaranteed small).
    4. Run the KS block-convergence check in pure NumPy on the driver.
    5. Return a Spark DataFrame created from the chosen rows — no executor
       caching, no HDFS I/O, no persistent RDDs.

    Args:
        nids_df:      Input Spark DataFrame (one streaming window).
        spark:        Active SparkSession.
        feature_cols: Numeric columns for KS convergence check.
        epsilon:      CDFRS epsilon (distributional tolerance).
        alpha:        CDFRS alpha (failure probability).
        M:            Number of fine blocks for KS convergence check.

    Returns:
        A small representative subsample DataFrame (≤ CDFRS_COLLECT_CAP rows).
    """
    import numpy as np
    from collections import defaultdict

    if feature_cols is None:
        feature_cols = SWD_COLS

    # ── 1. Theoretical sample size K ─────────────────────────────────────────
    K = int(math.log(2 / alpha) / (2 * epsilon ** 2))
    # K ≈ 530 for defaults. We collect up to CDFRS_COLLECT_CAP for a margin.

    # ── 2. Fraction sample on workers — no collect of large data ─────────────
    total = nids_df.count()
    if total == 0:
        raise ValueError("[CDFRS] Input window is empty.")

    target = min(CDFRS_COLLECT_CAP, max(K * 4, 2000))
    fraction = min(1.0, (target * 1.5) / total)   # 50% oversample, then limit

    sampled = (
        nids_df
        .sample(withReplacement=False, fraction=fraction, seed=random.randint(0, 9999))
        .limit(target)
    )

    # ── 3. Assign block IDs and collect — guaranteed ≤ CDFRS_COLLECT_CAP rows ─
    k_cdf = min(M, 20)   # number of blocks for KS check; 20 is plenty
    all_cols = feature_cols + list({c for c in nids_df.columns
                                    if c not in feature_cols})
    sampled_with_block = sampled.withColumn(
        "_bid", (F.monotonically_increasing_id() % k_cdf).cast(IntegerType())
    )

    rows = sampled_with_block.collect()

    if not rows:
        raise ValueError("[CDFRS] Sampled DataFrame is empty.")

    # ── 4. KS block-convergence check on driver (pure NumPy) ─────────────────
    blocks = defaultdict(list)
    for row in rows:
        blocks[int(row["_bid"])].append(row)
    block_ids = sorted(blocks.keys())

    def get_arrays(up_to_t):
        combined = []
        for bid in block_ids[:up_to_t]:
            combined.extend(blocks[bid])
        out = {}
        for col in feature_cols:
            vals = [float(r[col]) for r in combined if r[col] is not None]
            out[col] = np.array(vals, dtype=np.float64) if vals else np.array([])
        return out

    def ks_1d(a, b):
        if len(a) == 0 or len(b) == 0:
            return 0.0
        combined_vals = np.unique(np.concatenate([a, b]))
        cdf_a = np.searchsorted(np.sort(a), combined_vals, side="right") / len(a)
        cdf_b = np.searchsorted(np.sort(b), combined_vals, side="right") / len(b)
        return float(np.max(np.abs(cdf_a - cdf_b)))

    epsilon_A2 = 0.02
    T_max = min(len(block_ids), 20)
    t, chosen_t = 2, None

    while t <= T_max:
        arrs_prev = get_arrays(t - 1)
        arrs_curr = get_arrays(t)
        delta_max = max(
            (ks_1d(arrs_prev[c], arrs_curr[c]) for c in feature_cols),
            default=0.0,
        )
        print(f"  [CDFRS] t={t}  KS-max={delta_max:.4f}")
        if delta_max <= epsilon_A2:
            chosen_t = t - 1
            break
        t += 1

    if chosen_t is None:
        chosen_t = T_max - 1

    # ── 5. Build final sample from chosen blocks ──────────────────────────────
    chosen_rows = []
    for bid in block_ids[:chosen_t]:
        chosen_rows.extend(blocks[bid])

    if not chosen_rows:
        chosen_rows = list(rows[:min(2000, len(rows))])

    # Strip the _bid helper column before creating the DataFrame
    schema = nids_df.schema
    final_rows = [{c: row[c] for c in schema.fieldNames()} for row in chosen_rows]

    print(f"  [CDFRS] Sample: {len(final_rows)} rows from {len(rows)} collected "
          f"(window total: {total:,})")

    return spark.createDataFrame(final_rows, schema=schema)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("CDFRS_NIDS_Test")
        .master("yarn")
        .config(
            "spark.yarn.appMasterEnv.PYSPARK_PYTHON",
            "/home/cwc/Major_Project/major-project/bin/python3",
        )
        .config(
            "spark.executorEnv.PYSPARK_PYTHON",
            "/home/cwc/Major_Project/major-project/bin/python3",
        )
        .getOrCreate()
    )

    window_df = spark.read.parquet("hdfs://localhost:9000/data/nids_windows/window_0000")
    print(f"Window rows: {window_df.count()}")
    print(f"Window schema: {window_df.columns}")

    sample = cdfrs_nids(window_df, spark)
    print(f"CDFRS sample rows: {sample.count()}")
    spark.stop()