"""
cdfrs.py — Column-agnostic CDFRS sampler
=========================================
Generalised version of cdfrs_higgs. Works with any PySpark DataFrame
by accepting feature_cols and label_col as parameters instead of
hardcoding HIGGS column names.
"""

import math
import random
import uuid

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from typing import List


def ks_distance(df1: DataFrame, df2: DataFrame, col_name: str, num_points: int = 50) -> float:
    """Approximate KS distance between two DataFrames on a single column."""
    quantiles = df1.approxQuantile(col_name, [i / (num_points - 1.0) for i in range(num_points)], 0.01)

    n1 = df1.count()
    n2 = df2.count()
    if n1 == 0 or n2 == 0:
        return 0.0

    max_diff = 0.0
    for v in quantiles:
        cdf1 = df1.filter(F.col(col_name) <= v).count() / float(n1)
        cdf2 = df2.filter(F.col(col_name) <= v).count() / float(n2)
        diff = abs(cdf1 - cdf2)
        if diff > max_diff:
            max_diff = diff

    return max_diff


def cdfrs(
    df: DataFrame,
    spark: SparkSession,
    feature_cols: List[str],
    epsilon: float = 0.05,
    alpha: float = 0.10,
    M: int = 50,
    k_cdf: int = 50,
    epsilon_A2: float = 0.02,
    T_max: int = 20,
    seed: int = 42,
) -> DataFrame:
    """
    Column-agnostic CDFRS adaptive sampler.

    Generalised from cdfrs_higgs — works with any DataFrame by accepting
    feature_cols explicitly instead of hardcoding column names.

    Args:
        df:           Input PySpark DataFrame.
        spark:        Active SparkSession.
        feature_cols: List of numeric feature column names to use for KS stopping criterion.
        epsilon:      CDFRS approximation error bound.
        alpha:        Confidence level for sample size calculation.
        M:            Number of repartition blocks.
        k_cdf:        Number of CDF blocks for convergence check.
        epsilon_A2:   KS distance threshold for convergence.
        T_max:        Maximum number of CDF blocks to consider.
        seed:         Random seed.

    Returns:
        A PySpark DataFrame — the adaptively chosen sample.
    """
    K = int(math.log(2 / alpha) / (2 * epsilon ** 2))
    s = max(1, M // max(1, K))

    df_blocks = df.repartition(M)
    df_with_subset = df_blocks.withColumn(
        "subset_id", (F.rand(seed=seed) * s).cast(IntegerType())
    )

    chosen_subset_id = random.randint(0, s - 1)
    D = df_with_subset.filter(F.col("subset_id") == chosen_subset_id).drop("subset_id")

    D_shuffled = D.orderBy(F.rand(seed=seed + 1))

    # Use monotonically_increasing_id to avoid global Window sort (no WindowExec warning)
    D_with_block = D_shuffled.withColumn(
        "cdfrs_block_id",
        (F.monotonically_increasing_id().cast("long") % k_cdf).cast(IntegerType()),
    )

    # Write to a unique HDFS path so concurrent calls don't overwrite each other
    run_id = uuid.uuid4().hex
    output_path = f"hdfs:///data/cdfrs_blocks_{run_id}"

    D_with_block.write.partitionBy("cdfrs_block_id").mode("overwrite").parquet(output_path)

    blocks_df = spark.read.parquet(output_path).cache()
    blocks_df.count()  # force materialization

    block_ids = sorted([
        row.cdfrs_block_id
        for row in blocks_df.select("cdfrs_block_id").distinct().collect()
    ])

    effective_T_max = min(T_max, len(block_ids))

    def get_S_t(t: int) -> DataFrame:
        ids = block_ids[:t]
        return blocks_df.filter(F.col("cdfrs_block_id").isin(ids))

    t = 2
    chosen_t = None

    while t <= effective_T_max:
        S_prev = get_S_t(t - 1).cache()
        S_curr = get_S_t(t).cache()

        deltas = []
        for col_name in feature_cols:
            d = ks_distance(S_prev.select(col_name), S_curr.select(col_name), col_name)
            deltas.append(d)

        S_prev.unpersist()
        S_curr.unpersist()

        delta_max = max(deltas) if deltas else 0.0
        print(f"[CDFRS] t={t}, KS max = {delta_max:.6f}")

        if delta_max <= epsilon_A2:
            chosen_t = t - 1
            break

        t += 1

    if chosen_t is None:
        chosen_t = effective_T_max - 1

    print(f"[CDFRS] Chosen sample: {chosen_t} blocks")

    final_sample = get_S_t(chosen_t)
    final_sample = final_sample.drop("cdfrs_block_id").cache()
    final_sample.count()

    return final_sample