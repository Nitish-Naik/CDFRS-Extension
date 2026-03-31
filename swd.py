import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from typing import List, Optional
import random
from scipy.stats import wasserstein_distance
from cdfrs_higgs import cdfrs_higgs


def sliced_wasserstein_distance(
    df1: DataFrame,
    df2: DataFrame,
    feature_cols: List[str],
    n_projections: int = 50,
    seed: Optional[int] = None,
) -> float:
    """
    Compute the Sliced Wasserstein Distance (SWD) between two PySpark DataFrames.

    SWD approximates the Wasserstein distance by averaging 1D Wasserstein distances
    over random projections of the data onto unit vectors.

    Args:
        df1:           First PySpark DataFrame.
        df2:           Second PySpark DataFrame.
        feature_cols:  List of numeric column names to use as features.
        n_projections: Number of random 1D projections (more = more accurate, slower).
        seed:          Random seed for reproducibility.

    Returns:
        Scalar float — the estimated Sliced Wasserstein Distance.

    Example:
        swd = sliced_wasserstein_distance(
            df1, df2,
            feature_cols=["age", "income", "score"],
            n_projections=100,
            seed=42,
        )
        print(f"SWD: {swd:.4f}")
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    dim = len(feature_cols)
    if dim == 0:
        raise ValueError("feature_cols must be non-empty.")

    # ── 1. Pull feature vectors from Spark to driver as numpy arrays ──────────
    def _to_numpy(df: DataFrame) -> np.ndarray:
        """Select feature columns and collect as a (N, dim) float64 array."""
        rows = df.select(*feature_cols).dropna().collect()
        return np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)

    X = _to_numpy(df1)  # shape (N, dim)
    Y = _to_numpy(df2)  # shape (M, dim)

    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("One or both DataFrames have no non-null rows for the given columns.")

    # ── 2. Generate random unit-vector projections ────────────────────────────
    # Sample from a standard normal and normalize each row → uniform on S^(d-1)
    directions = np.random.randn(n_projections, dim)          # (P, dim)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)  # normalize

    # ── 3. Project both datasets onto each direction ──────────────────────────
    # X_proj[p, :] = X @ directions[p]  →  shape (P, N)
    X_proj = (directions @ X.T)  # (P, N)
    Y_proj = (directions @ Y.T)  # (P, M)

    # ── 4. Compute 1D Wasserstein distance per projection ─────────────────────
    # W1 in 1D = mean |F^{-1}(u) - G^{-1}(u)| over u ~ Uniform[0,1]
    #          = mean |sort(X_proj) - sort(Y_proj)| after interpolation
    swd_sum = 0.0
    for p in range(n_projections):
        xp = np.sort(X_proj[p])
        yp = np.sort(Y_proj[p])

        # Interpolate to a common grid so sizes don't need to match
        n_common = max(len(xp), len(yp))
        u = np.linspace(0, 1, n_common)
        xp_interp = np.interp(u, np.linspace(0, 1, len(xp)), xp)
        yp_interp = np.interp(u, np.linspace(0, 1, len(yp)), yp)

        swd_sum += np.mean(np.abs(xp_interp - yp_interp))

    return float(swd_sum / n_projections)


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

    print(sliced_wasserstein_distance(sample1, sample2, [f"x{i}" for i in range(1, num_features+1)]))