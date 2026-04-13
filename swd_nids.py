"""
SWD Drift Detection — NF-UQ-NIDS-v2 Edition
=============================================
Adapted from swd4.py.

Changes vs HIGGS version
--------------------------
- Imports cdfrs_nids instead of cdfrs_higgs.
- Default feature_cols point to NIDS_FEATURE_COLS / SWD_COLS.
- No structural changes to SWD math or permutation test.
- Smoke-test __main__ reads from nids_windows partitions.
"""

import numpy as np
from pyspark.sql import SparkSession, DataFrame
from typing import List, Optional

from cdfrs_nids import cdfrs_nids, SWD_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Data Extraction Helpers  (unchanged from swd4.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_to_numpy(df: DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Pull a PySpark DataFrame into a NumPy array exactly once."""
    rows = df.select(*feature_cols).dropna().collect()
    arr = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)
    if arr.shape[0] == 0:
        raise ValueError("DataFrame has no non-null rows for the given columns.")
    return arr


def extract_sample_to_numpy(
    df: DataFrame,
    feature_cols: List[str],
    max_samples: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample up to max_samples rows from a Spark DataFrame and collect to NumPy.
    Sampling is done on workers BEFORE collecting to driver.
    Returns a **standardized** array (zero mean, unit variance per feature).
    Standardization is mandatory for SWD: without it, high-magnitude columns
    (throughput, DNS TTL, port numbers) dominate random projections entirely,
    producing SWD values in the billions that are numerically meaningless.
    """
    total_rows = df.count()
    if total_rows == 0:
        raise ValueError("DataFrame is empty.")

    if total_rows <= max_samples:
        sampled_df = df
    else:
        fraction = min(1.0, (max_samples * 1.2) / total_rows)
        sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=seed).limit(max_samples)

    rows = sampled_df.select(*feature_cols).dropna().collect()
    arr = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)

    # Per-feature standardization: (x - mean) / std
    # Columns with zero variance (constant features) are zeroed out rather
    # than causing a divide-by-zero — they carry no drift signal anyway.
    means = arr.mean(axis=0)
    stds  = arr.std(axis=0)
    stds[stds == 0] = 1.0          # avoid divide-by-zero on constant cols
    arr = (arr - means) / stds

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Core SWD  (unchanged from swd4.py)
# ─────────────────────────────────────────────────────────────────────────────

def sliced_wasserstein_distance_np(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 50,
    seed: Optional[int] = None,
) -> float:
    if seed is not None:
        np.random.seed(seed)

    dim = X.shape[1]
    directions = np.random.randn(n_projections, dim)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    X_proj = directions @ X.T
    Y_proj = directions @ Y.T

    swd_sum = 0.0
    for p in range(n_projections):
        xp = np.sort(X_proj[p])
        yp = np.sort(Y_proj[p])
        n, m = len(xp), len(yp)
        all_probs = np.unique(
            np.clip(
                np.concatenate([np.arange(1, n + 1) / n, np.arange(1, m + 1) / m]),
                0, 1,
            )
        )
        swd_sum += np.mean(np.abs(np.quantile(xp, all_probs) - np.quantile(yp, all_probs)))

    return float(swd_sum / n_projections)


# ─────────────────────────────────────────────────────────────────────────────
# Permutation Test  (unchanged from swd4.py — both variants kept)
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test_swd_ultra_fast(
    X_ref: np.ndarray,
    X_det: np.ndarray,
    max_samples: int = 5000,
    n_projections: int = 50,
    n_permutations: int = 500,
    seed: int = 42,
    significance_level: float = 0.05,
):
    """
    Ultra-fast SWD permutation test with optional subsampling.
    Pre-projects combined array once; permutes index arrays in the null loop.
    """
    rng = np.random.default_rng(seed)

    n_samples = min(X_ref.shape[0], X_det.shape[0])
    combined = np.vstack([X_ref[:n_samples], X_det[:n_samples]])

    dim = combined.shape[1]
    directions = rng.standard_normal((n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projected = directions @ combined.T  # (n_proj, 2*n_samples)

    def fast_w1(proj_row, idx_a, idx_b):
        return float(np.mean(np.abs(np.sort(proj_row[idx_a]) - np.sort(proj_row[idx_b]))))

    ref_idx = np.arange(n_samples)
    det_idx = np.arange(n_samples, 2 * n_samples)

    observed_swd = sum(
        fast_w1(projected[p], ref_idx, det_idx) for p in range(n_projections)
    ) / n_projections

    all_indices = np.arange(2 * n_samples)
    null_dist = np.zeros(n_permutations)

    for i in range(n_permutations):
        rng.shuffle(all_indices)
        null_dist[i] = sum(
            fast_w1(projected[p], all_indices[:n_samples], all_indices[n_samples:])
            for p in range(n_projections)
        ) / n_projections

    p_value = (np.sum(null_dist > observed_swd) + 1) / (n_permutations + 1)
    return observed_swd, p_value, null_dist


# ─────────────────────────────────────────────────────────────────────────────
# Reporting  (unchanged from swd4.py)
# ─────────────────────────────────────────────────────────────────────────────

def print_swd_report(
    observed_swd: float,
    p_value: float,
    null_dist: np.ndarray,
    significance_level: float = 0.05,
):
    z_score = (observed_swd - null_dist.mean()) / (null_dist.std() + 1e-12)
    drift_detected = p_value < significance_level

    print("\n" + "=" * 50)
    print("         SWD DRIFT DETECTION REPORT")
    print("=" * 50)
    print(f"  Observed SWD     : {observed_swd:.6f}")
    print(f"  Null mean        : {null_dist.mean():.6f}")
    print(f"  Null std         : {null_dist.std():.6f}")
    print(f"  Z-score          : {z_score:.2f} std devs above null")
    print(f"  p-value          : {p_value:.4f}")
    print(f"  Significance (α) : {significance_level}")
    print(f"  Drift detected   : {'YES ⚠️' if drift_detected else 'NO ✓'}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("SWD_NIDS_Test")
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

    w0 = spark.read.parquet("hdfs://localhost:9000/data/nids_windows/window_0000")
    w1 = spark.read.parquet("hdfs://localhost:9000/data/nids_windows/window_0001")

    s0 = cdfrs_nids(w0, spark)
    s1 = cdfrs_nids(w1, spark)

    X_ref = extract_sample_to_numpy(s0, SWD_COLS, max_samples=5000)
    X_det = extract_sample_to_numpy(s1, SWD_COLS, max_samples=5000)

    print(f"X_ref shape: {X_ref.shape}")
    print(f"X_det shape: {X_det.shape}")

    obs_swd, p_val, null_dist = permutation_test_swd_ultra_fast(
        X_ref=X_ref,
        X_det=X_det,
        n_projections=50,
        n_permutations=500,
        seed=42,
    )
    print_swd_report(obs_swd, p_val, null_dist)
    spark.stop()