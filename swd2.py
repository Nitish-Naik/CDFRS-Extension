import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from typing import List, Optional
import random
from scipy.stats import wasserstein_distance
from cdfrs_higgs import cdfrs_higgs

def swd_permutation_test(df1, df2, feature_cols, n_projections=100, n_permutations=99, seed=42):
    """
    Returns (observed_swd, p_value).
    p_value < 0.05 means the two datasets are significantly different.
    """

    np.random.seed(seed)

    def _to_numpy(df):
        rows = df.select(*feature_cols).dropna().limit(10000).collect()
        return np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)

    X = _to_numpy(df1)
    Y = _to_numpy(df2)

    def _swd_numpy(A, B, n_proj):
        dim = A.shape[1]
        directions = np.random.randn(n_proj, dim)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        A_proj = directions @ A.T
        B_proj = directions @ B.T
        total = 0.0
        for p in range(n_proj):
            ap = np.sort(A_proj[p])
            bp = np.sort(B_proj[p])
            n, m = len(ap), len(bp)
            probs = np.unique(np.clip(np.concatenate([
                np.arange(1, n+1)/n, np.arange(1, m+1)/m
            ]), 0, 1))
            total += np.mean(np.abs(np.quantile(ap, probs) - np.quantile(bp, probs)))
        return total / n_proj

    # Observed SWD
    observed = _swd_numpy(X, Y, n_projections)

    # Null distribution: shuffle combined data and split into two random halves
    combined = np.vstack([X, Y])
    n = len(X)
    null_swds = []

    for _ in range(n_permutations):
        idx = np.random.permutation(len(combined))
        A_perm = combined[idx[:n]]
        B_perm = combined[idx[n:]]
        null_swds.append(_swd_numpy(A_perm, B_perm, n_projections))

    null_swds = np.array(null_swds)
    p_value = float(np.mean(null_swds >= observed))

    print(f"Observed SWD : {observed:.6f}")
    print(f"Null SWD     : mean={null_swds.mean():.6f}, std={null_swds.std():.6f}")
    print(f"p-value      : {p_value:.4f}")
    print(f"Significant  : {p_value < 0.05}")

    return observed, p_value

def sliced_wasserstein_distance(
    df1: DataFrame,
    df2: DataFrame,
    feature_cols: List[str] = [f"x{i}" for i in range(1, 28+1)],
    n_projections: int = 50,
    seed: Optional[int] = None,
) -> float:
    """
    Sliced Wasserstein Distance between two PySpark DataFrames.

    Args:
        df1, df2:      PySpark DataFrames to compare.
        feature_cols:  Numeric columns to use as feature dimensions.
        n_projections: Number of random 1D projections.
        seed:          Random seed for reproducibility.

    Returns:
        Estimated Sliced Wasserstein Distance (float).
    """
    if seed is not None:
        np.random.seed(seed)

    dim = len(feature_cols)
    if dim == 0:
        raise ValueError("feature_cols must be non-empty.")
    
    # Compute per-feature std on df1 as reference scale
    stats = df1.select([F.stddev(c).alias(c) for c in feature_cols]).collect()[0]
    stds = np.array([stats[c] for c in feature_cols], dtype=np.float64)
    stds[stds == 0] = 1.0  # avoid division by zero

    def _to_numpy(df: DataFrame) -> np.ndarray:
        rows = df.select(*feature_cols).dropna().collect()
        arr = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)
        if arr.shape[0] == 0:
            raise ValueError("DataFrame has no non-null rows for the given columns.")
        return arr

    X = _to_numpy(df1)  # (N, dim)
    Y = _to_numpy(df2)  # (M, dim)

    print(f"[SWD] X shape: {X.shape}, Y shape: {Y.shape}")

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

        # ── FIX: use quantile-based comparison instead of interpolation ──
        # Evaluate both quantile functions at the same M+N-1 probability levels
        # This is the exact 1D Wasserstein via the quantile (inverse CDF) formula
        n, m = len(xp), len(yp)
        all_probs = np.concatenate([
            np.arange(1, n + 1) / n,
            np.arange(1, m + 1) / m,
        ])
        all_probs = np.unique(np.clip(all_probs, 0, 1))

        xp_q = np.quantile(xp, all_probs)
        yp_q = np.quantile(yp, all_probs)

        swd_sum += np.mean(np.abs(xp_q - yp_q))

    # After collecting your data, run these checks:

    # 1. Check shapes and a few raw values
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("X first 3 rows:\n", X[:3])
    print("Y first 3 rows:\n", Y[:3])

    # 2. Check if arrays are literally identical
    print("Arrays identical?", np.array_equal(X, Y))
    print("Max abs diff between X and Y:", np.max(np.abs(X - Y)) if X.shape == Y.shape else "different shapes")

    # 3. Check a single projection manually
    np.random.seed(42)
    dim = X.shape[1]
    d = np.random.randn(dim)
    d /= np.linalg.norm(d)

    xp = np.sort(X @ d)
    yp = np.sort(Y @ d)
    print("xp[:5]:", xp[:5])
    print("yp[:5]:", yp[:5])
    print("xp[-5:]:", xp[-5:])
    print("yp[-5:]:", yp[-5:])

    n, m = len(xp), len(yp)
    all_probs = np.unique(np.clip(np.concatenate([
        np.arange(1, n + 1) / n,
        np.arange(1, m + 1) / m,
    ]), 0, 1))

    xp_q = np.quantile(xp, all_probs)
    yp_q = np.quantile(yp, all_probs)
    print("W1 for this projection:", np.mean(np.abs(xp_q - yp_q)))

    # 4. Check variance of projections across directions
    diffs = []
    for _ in range(20):
        d = np.random.randn(dim)
        d /= np.linalg.norm(d)
        xp = np.sort(X @ d)
        yp = np.sort(Y @ d)
        n, m = len(xp), len(yp)
        all_probs = np.unique(np.clip(np.concatenate([
            np.arange(1, n+1)/n,
            np.arange(1, m+1)/m,
        ]), 0, 1))
        diffs.append(np.mean(np.abs(np.quantile(xp, all_probs) - np.quantile(yp, all_probs))))

    print("W1 across 20 projections:", diffs)
    print("Mean:", np.mean(diffs), "Std:", np.std(diffs))

    return float(swd_sum / n_projections)


#Gemini
def sliced_wasserstein_distance_normalized(
    X: np.ndarray, 
    Y: np.ndarray, 
    n_projections: int = 50, 
    seed: int = None
) -> float:
    """
    Computes the Sliced Wasserstein Distance between two datasets with built-in Z-score normalization.
    
    Args:
        X: NumPy array of shape (N, features)
        Y: NumPy array of shape (M, features)
        n_projections: Number of random 1D projections to approximate the integral.
        seed: Random seed for reproducibility.
        
    Returns:
        Estimated Sliced Wasserstein Distance (float).
    """
    if seed is not None:
        np.random.seed(seed)
        
    # ---------------------------------------------------------
    # 1. Normalization (Z-score Standardization)
    # We use the combined distribution to find the global mean 
    # and std to ensure both datasets are scaled uniformly.
    # ---------------------------------------------------------
    combined = np.vstack([X, Y])
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0)
    
    # Prevent division by zero for constant (zero-variance) features
    std[std == 0] = 1.0 
    
    X_norm = (X - mean) / std
    Y_norm = (Y - mean) / std
    
    dim = X.shape[1]
    
    # ---------------------------------------------------------
    # 2. Random Projections
    # Generate random directions uniformly on the unit sphere S^(dim-1)
    # ---------------------------------------------------------
    directions = np.random.randn(n_projections, dim)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    
    # Project data onto the random directions using matrix multiplication
    X_proj = X_norm @ directions.T  # Shape: (N, n_projections)
    Y_proj = Y_norm @ directions.T  # Shape: (M, n_projections)
    
    # ---------------------------------------------------------
    # 3. 1D Wasserstein Computation via Quantiles
    # ---------------------------------------------------------
    swd_sum = 0.0
    n, m = X_proj.shape[0], Y_proj.shape[0]
    
    # Pre-compute probability levels to evaluate the inverse CDF (quantile function)
    # This aligns the discrete distributions regardless of differing sample sizes.
    all_probs = np.concatenate([
        np.arange(1, n + 1) / n,
        np.arange(1, m + 1) / m,
    ])
    all_probs = np.unique(np.clip(all_probs, 0, 1))
    
    for p in range(n_projections):
        xp = np.sort(X_proj[:, p])
        yp = np.sort(Y_proj[:, p])
        
        # Evaluate both quantile functions at the same probability levels
        xp_q = np.quantile(xp, all_probs)
        yp_q = np.quantile(yp, all_probs)
        
        # W1 distance for this 1D slice
        swd_sum += np.mean(np.abs(xp_q - yp_q))
        
    return float(swd_sum / n_projections)

def permutation_test_swd(X_ref, X_det, n_projections=200, n_permutations=500, seed=42):
    observed_swd = sliced_wasserstein_distance(X_ref, X_det, n_projections=n_projections, seed=seed)
    
    combined = np.vstack([X_ref, X_det])
    n_ref = X_ref.count()
    
    null_distribution = []
    rng = np.random.RandomState(seed)
    
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        perm_ref = combined[perm[:n_ref]]
        perm_det = combined[perm[n_ref:]]
        null_distribution.append(
            sliced_wasserstein_distance(perm_ref, perm_det, n_projections=n_projections, seed=seed)
        )
    
    null_distribution = np.array(null_distribution)
    p_value = np.mean(null_distribution >= observed_swd)

    print(observed_swd)
    print(p_value)
    print(null_distribution)
    
    return observed_swd, p_value, null_distribution

import numpy as np
from pyspark.sql.functions import rand, row_number, col
from pyspark.sql.window import Window

def permutation_test_swd_pyspark(X_ref, X_det, n_projections=200, n_permutations=500, seed=42, exact_split=False):
    """
    PySpark refactor of SWD permutation test.
    
    Args:
        X_ref (DataFrame): Reference PySpark DataFrame.
        X_det (DataFrame): Detection PySpark DataFrame.
        swd_func (callable): A function that calculates SWD taking two PySpark DataFrames.
        n_projections (int): Number of projections for SWD.
        n_permutations (int): Number of permutation iterations.
        seed (int): Random seed.
        exact_split (bool): If True, strictly enforces identical row counts for permutations (slower). 
                            If False, uses randomSplit for high-performance approximations (faster).
    """
    # 1. Calculate observed SWD
    # Assumption: swd_func is refactored to accept PySpark DataFrames
    observed_swd = sliced_wasserstein_distance(X_ref, X_det, n_projections=n_projections, seed=seed)
    
    n_ref = X_ref.count()
    n_det = X_det.count()
    total_n = n_ref + n_det
    
    # 2. Combine datasets
    # unionByName is safer than union() if columns happen to be out of order
    # .cache() is CRITICAL here so Spark doesn't re-read the source data 500 times
    combined_df = X_ref.unionByName(X_det, allowMissingColumns=True).cache()
    
    # Force the cache to materialize immediately
    combined_df.count() 
    
    null_distribution = []
    
    for i in range(n_permutations):
        print(i)
        current_seed = seed + i
        
        if not exact_split:
            # OPTION A: Scalable PySpark Approach (Highly Recommended)
            # randomSplit is highly parallelized and avoids global sorting. 
            # Sizes will be binomially distributed around n_ref, but very close.
            fraction = n_ref / total_n
            perm_ref, perm_det = combined_df.randomSplit([fraction, 1.0 - fraction], seed=current_seed)
        
        else:
            # OPTION B: Exact Size Permutation (Slower)
            # Uses a global Window to assign row numbers. This forces data into fewer 
            # partitions and requires heavy shuffling.
            window_spec = Window.orderBy(rand(seed=current_seed))
            shuffled_df = combined_df.withColumn("row_num", row_number().over(window_spec))
            
            perm_ref = shuffled_df.filter(col("row_num") <= n_ref).drop("row_num")
            perm_det = shuffled_df.filter(col("row_num") > n_ref).drop("row_num")
            
        # 3. Compute metric for the permutation
        # Make sure swd_func triggers a Spark Action (like .collect() or .sum()) internally.
        # If it doesn't, Spark's lazy evaluation will build a DAG 500 iterations deep and crash.
        perm_swd = sliced_wasserstein_distance(perm_ref, perm_det, n_projections=n_projections, seed=current_seed)
        null_distribution.append(perm_swd)
        
    null_distribution = np.array(null_distribution)
    p_value = np.mean(null_distribution >= observed_swd)

    print(f"Observed SWD: {observed_swd}")
    print(f"P-Value: {p_value}")
    
    # Free up memory
    combined_df.unpersist()
    
    return observed_swd, p_value, null_distribution




if __name__ == "__main__":
    spark = SparkSession.builder \
    .appName('CDFRS Sample') \
    .master('yarn') \
    .getOrCreate()
    
    num_features = 28
    cols = ["label"] + [f"x{i}" for i in range(1, num_features+1)]

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    higgs_df1 = spark.read.parquet(
        "hdfs://localhost:9000/data/higgs_parquet/part-00000-cbe60f1e-6bdd-4c21-a969-c413a2625549-c000.snappy.parquet",
        header = False,
        inferSchema = True
    ).toDF(*cols)

    sample1 = cdfrs_higgs(higgs_df1, spark)

    higgs_df2 = spark.read.parquet(
        "hdfs://localhost:9000/data/higgs_parquet/part-00002-cbe60f1e-6bdd-4c21-a969-c413a2625549-c000.snappy.parquet",
        header = False,
        inferSchema = True
    ).toDF(*cols)

    sample2 = cdfrs_higgs(higgs_df2, spark)

    #print(sliced_wasserstein_distance(sample1, sample2, [f"x{i}" for i in range(1, num_features+1)]))

    #swd_permutation_test(sample1, sample2, [f"x{i}" for i in range(1, num_features+1)])
    #print(sliced_wasserstein_distance_normalized(sample1, sample2))
    swd, p_val, null_dist = permutation_test_swd_pyspark(sample1, sample2)
    #swd, p_val, null_dist = permutation_test_swd(sample1, sample2)
    print(f"SWD: {swd:.4f}, p-value: {p_val:.4f}")

    ALPHA = 0.05
    if p_val < ALPHA:
        print("⚠️  Drift detected!")
    else:
        print("✅  No significant drift")
