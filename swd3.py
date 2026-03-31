import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from typing import List, Optional
import random
from scipy.stats import wasserstein_distance
from cdfrs_higgs import cdfrs_higgs


def extract_to_numpy(df: DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Helper to pull PySpark DataFrame into a NumPy array exactly once."""
    rows = df.select(*feature_cols).dropna().collect()
    arr = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)
    if arr.shape[0] == 0:
        raise ValueError("DataFrame has no non-null rows for the given columns.")
    return arr

def sliced_wasserstein_distance_np(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 50,
    seed: Optional[int] = None,
) -> float:
    """
    Pure NumPy implementation of SWD. No PySpark dependencies.
    """
    if seed is not None:
        np.random.seed(seed)

    dim = X.shape[1]
    
    # Random unit-vector projections: uniform on S^(dim-1)
    directions = np.random.randn(n_projections, dim)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Project: (P, N) and (P, M)
    X_proj = directions @ X.T
    Y_proj = directions @ Y.T

    swd_sum = 0.0
    for p in range(n_projections):
        xp = np.sort(X_proj[p])  # (N,)
        yp = np.sort(Y_proj[p])  # (M,)

        n, m = len(xp), len(yp)
        all_probs = np.concatenate([
            np.arange(1, n + 1) / n,
            np.arange(1, m + 1) / m,
        ])
        all_probs = np.unique(np.clip(all_probs, 0, 1))

        xp_q = np.quantile(xp, all_probs)
        yp_q = np.quantile(yp, all_probs)

        swd_sum += np.mean(np.abs(xp_q - yp_q))

    return float(swd_sum / n_projections)

def permutation_test_swd_fast(
    X_ref_df: DataFrame, 
    X_det_df: DataFrame, 
    feature_cols: List[str],
    n_projections: int = 50, 
    n_permutations: int = 500, 
    seed: int = 42
):
    """
    Optimized Permutation test: Collects to NumPy once, loops locally.
    """
    print("Collecting DataFrames to driver memory... (This happens exactly once)")
    X_ref = extract_to_numpy(X_ref_df, feature_cols)
    X_det = extract_to_numpy(X_det_df, feature_cols)
    
    n_ref = X_ref.shape[0]
    
    # 1. Calculate observed SWD
    print("Calculating observed SWD...")
    observed_swd = sliced_wasserstein_distance_np(
        X_ref, X_det, n_projections=n_projections, seed=seed
    )
    
    # 2. Combine datasets locally in NumPy
    combined_X = np.vstack([X_ref, X_det])
    null_distribution = np.zeros(n_permutations)
    
    # Setup random generator for fast shuffling
    rng = np.random.default_rng(seed)
    
    print(f"Starting {n_permutations} permutations locally...")
    # 3. Fast Local Permutation Loop
    for i in range(n_permutations):
        #if i % 50 == 0:
        print(f"  ...Permutation {i}/{n_permutations}")
            
        # In-place shuffle is incredibly fast in NumPy
        rng.shuffle(combined_X) 
        
        # Exact split based on original sample sizes
        perm_ref = combined_X[:n_ref, :]
        perm_det = combined_X[n_ref:, :]
        
        # Compute metric for the permutation (pass a new seed or let it float)
        perm_swd = sliced_wasserstein_distance_np(
            perm_ref, perm_det, n_projections=n_projections
        )
        null_distribution[i] = perm_swd
        
    p_value = np.mean(null_distribution >= observed_swd)

    return observed_swd, p_value, null_distribution


if __name__ == "__main__":
    spark = SparkSession.builder \
    .appName('CDFRS Sample') \
    .master('yarn') \
    .getOrCreate()
    
    num_features = 28
    cols = ["label"] + [f"x{i}" for i in range(1, num_features+1)]

    higgs_df1 = spark.read.parquet(
        "hdfs://localhost:9000/data/higgs_parquet/part-00000-cbe60f1e-6bdd-4c21-a969-c413a2625549-c000.snappy.parquet",
        header = False,
        inferSchema = True
    ).toDF(*cols)

    sample1 = cdfrs_higgs(higgs_df1, spark)

    higgs_df2 = spark.read.parquet(
        "hdfs://localhost:9000/data/higgs_parquet/part-00006-cbe60f1e-6bdd-4c21-a969-c413a2625549-c000.snappy.parquet",
        header = False,
        inferSchema = True
    ).toDF(*cols)

    sample2 = cdfrs_higgs(higgs_df2, spark)

    feature_columns = [f"x{i}" for i in range(1, num_features+1)]

    # Look how clean the call is now!
    swd, p_val, null_dist = permutation_test_swd_fast(
        X_ref_df=sample1, 
        X_det_df=sample2, 
        feature_cols=feature_columns,
        n_projections=50,      # Reduced from 200 for speed, adjust as needed
        n_permutations=500,
        seed=42
    )