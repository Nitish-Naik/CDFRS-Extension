"""
ensemble_pipeline.py — Generic Drift-Aware Weighted Ensemble Pipeline
======================================================================
Works with ANY dataset. Configure via EnsembleConfig + DataConfig.

Window sources supported:
  - "directory" : all parquet files in an HDFS/local directory
  - "split"     : one large parquet file split into N equal row-count windows
  - "stream"    : list of pre-defined paths (manual / Kafka-fed paths)

Base models : Decision Tree, Random Forest, GBT, Logistic Regression
Task        : Binary classification
Weights     : Recency-based exponential decay
Drift det.  : SWD + permutation test
Viz         : EnsembleVisualizer (visualizations.py)
"""

import os
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterator

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
    LogisticRegression,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from cdfrs import cdfrs
from swd4 import extract_sample_to_numpy, permutation_test_swd_ultra_fast, print_swd_report
from visualizations import EnsembleVisualizer


# ─────────────────────────────────────────────────────────────────────────────
# Data Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """
    Everything dataset-specific lives here.

    Window source modes
    -------------------
    "directory" : Read every parquet file in `source_path` as one window each.
    "split"     : Read one large file at `source_path`, split into `n_splits` windows.
    "stream"    : `window_paths` is a manually provided list of file paths.

    Column naming
    -------------
    label_col   : Binary label column (must contain 0.0 / 1.0).
    feature_cols: Numeric feature column names for training and drift detection.

    Schema renaming
    ---------------
    col_rename  : Optional {old_name: new_name} applied after reading.
                  Useful for headerless CSVs e.g. {"_c0": "label", "_c1": "x1"}.
    """
    window_mode:      str       = "directory"
    source_path:      str       = ""
    n_splits:         int       = 6
    window_paths:     List[str] = field(default_factory=list)
    label_col:        str       = "label"
    feature_cols:     List[str] = field(default_factory=list)
    col_rename:       dict      = field(default_factory=dict)
    file_format:      str       = "parquet"
    csv_header:       bool      = True
    csv_infer_schema: bool      = True

    def validate(self):
        if not self.label_col:
            raise ValueError("DataConfig.label_col must be set.")
        if not self.feature_cols:
            raise ValueError("DataConfig.feature_cols must be non-empty.")
        if self.window_mode == "stream" and not self.window_paths:
            raise ValueError("window_paths must be non-empty when window_mode='stream'.")
        if self.window_mode in ("directory", "split") and not self.source_path:
            raise ValueError("source_path must be set for 'directory' or 'split' mode.")


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleConfig:
    max_ensemble_size: int   = 5
    recency_decay:     float = 0.5
    drift_alpha:       float = 0.05
    n_projections:     int   = 100
    n_permutations:    int   = 500
    max_samples:       int   = 5000
    seed:              int   = 42


# ─────────────────────────────────────────────────────────────────────────────
# Window Source
# ─────────────────────────────────────────────────────────────────────────────

class WindowSource:
    def __init__(self, data_config: DataConfig, spark: SparkSession):
        self.cfg   = data_config
        self.spark = spark

    def _read(self, path: str) -> DataFrame:
        cfg = self.cfg
        if cfg.file_format == "csv":
            df = self.spark.read.csv(path, header=cfg.csv_header,
                                     inferSchema=cfg.csv_infer_schema)
        else:
            df = self.spark.read.parquet(path)
        for old, new in cfg.col_rename.items():
            if old in df.columns:
                df = df.withColumnRenamed(old, new)
        return df

    def _list_parquet_files(self, directory: str) -> List[str]:
        try:
            jvm  = self.spark._jvm
            conf = self.spark._jsc.hadoopConfiguration()
            fs   = jvm.org.apache.hadoop.fs.FileSystem.get(conf)
            path = jvm.org.apache.hadoop.fs.Path(directory)
            files = sorted([
                f.getPath().toString()
                for f in fs.listStatus(path)
                if f.getPath().getName().endswith(".parquet")
            ])
        except Exception:
            files = sorted([
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith(".parquet")
            ])
        if not files:
            raise RuntimeError(f"No parquet files found in: {directory}")
        return files

    def windows(self) -> Iterator[Tuple[int, DataFrame]]:
        cfg = self.cfg

        if cfg.window_mode == "directory":
            paths = self._list_parquet_files(cfg.source_path)
            print(f"[WindowSource] Directory mode: {len(paths)} files")
            for i, path in enumerate(paths):
                print(f"[WindowSource] Window {i}: {os.path.basename(path)}")
                df = self._read(path).cache()
                df.count()
                yield i, df
                df.unpersist()

        elif cfg.window_mode == "split":
            print(f"[WindowSource] Split mode → {cfg.n_splits} windows")
            full_df = self._read(cfg.source_path)
            full_indexed = full_df.withColumn(
                "_window_id",
                (F.monotonically_increasing_id() % cfg.n_splits).cast("int"),
            )
            for i in range(cfg.n_splits):
                df = full_indexed.filter(F.col("_window_id") == i).drop("_window_id").cache()
                count = df.count()
                print(f"[WindowSource] Window {i}: {count} rows")
                yield i, df
                df.unpersist()

        elif cfg.window_mode == "stream":
            print(f"[WindowSource] Stream mode: {len(cfg.window_paths)} paths")
            for i, path in enumerate(cfg.window_paths):
                print(f"[WindowSource] Window {i}: {path}")
                df = self._read(path).cache()
                df.count()
                yield i, df
                df.unpersist()

        else:
            raise ValueError(f"Unknown window_mode '{cfg.window_mode}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Model Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(feature_cols, label_col, classifier_type="dt", seed=42):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features",
                                handleInvalid="skip")
    classifiers = {
        "dt":  DecisionTreeClassifier(labelCol=label_col, featuresCol="features",
                                      maxDepth=10, seed=seed),
        "rf":  RandomForestClassifier(labelCol=label_col, featuresCol="features",
                                      numTrees=50, maxDepth=10, seed=seed),
        "gbt": GBTClassifier(labelCol=label_col, featuresCol="features",
                             maxIter=20, maxDepth=6, seed=seed),
        "lr":  LogisticRegression(labelCol=label_col, featuresCol="features",
                                  maxIter=100, regParam=0.01),
    }
    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier_type '{classifier_type}'.")
    return Pipeline(stages=[assembler, classifiers[classifier_type]])


def evaluate_model(model, test_df, label_col):
    preds = model.transform(test_df)
    auc = BinaryClassificationMetrics(
        preds.select("rawPrediction", label_col)
             .rdd.map(lambda r: (float(r["rawPrediction"][1]), float(r[label_col])))
    ).areaUnderROC
    f1 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    ).evaluate(preds)
    return auc, f1


# ─────────────────────────────────────────────────────────────────────────────
# Recency Weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_recency_weights(k, decay):
    raw   = [decay ** (k - 1 - i) for i in range(k)]
    total = sum(raw)
    return [w / total for w in raw]


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Member
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleMember:
    model_id:   str
    model:      PipelineModel
    weight:     float
    train_auc:  float
    train_f1:   float
    window_idx: int
    model_type: str = "?"       # "dt" | "rf" | "gbt" | "lr" — used by visualizer


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class WeightedEnsemble:
    """Equation (2): ŷ(x) = Σ wi * P(class=1 | Mi(x))"""

    def __init__(self, config: EnsembleConfig):
        self.config  = config
        self.members: List[EnsembleMember] = []

    def add_model(self, model, auc, f1, window_idx, model_type="?"):
        m = EnsembleMember(
            model_id=uuid.uuid4().hex[:8], model=model,
            weight=0.0, train_auc=auc, train_f1=f1,
            window_idx=window_idx, model_type=model_type,
        )
        self.members.append(m)
        if len(self.members) > self.config.max_ensemble_size:
            evicted = self.members.pop(0)
            print(f"[Ensemble] Evicted {evicted.model_id} (window {evicted.window_idx})")
        self._recompute_weights()
        print(f"[Ensemble] Added {m.model_id} ({model_type}) | "
              f"AUC={auc:.4f} F1={f1:.4f} | size={len(self.members)}")
        self._print_weights()

    def _recompute_weights(self):
        weights = compute_recency_weights(len(self.members), self.config.recency_decay)
        for m, w in zip(self.members, weights):
            m.weight = w

    def _print_weights(self):
        print("[Ensemble] Weights:")
        for m in self.members:
            print(f"  {m.model_id} ({m.model_type})  win={m.window_idx}  "
                  f"w={m.weight:.4f}  AUC={m.train_auc:.4f}  F1={m.train_f1:.4f}")

    def predict(self, df: DataFrame) -> DataFrame:
        if not self.members:
            raise RuntimeError("Ensemble is empty.")

        df_indexed = df.withColumn("_row_id", F.monotonically_increasing_id())
        df_indexed = df_indexed.cache()
        df_indexed.count()

        result_df = df_indexed.select("_row_id")

        for i, member in enumerate(self.members):
            preds = member.model.transform(df_indexed)
            weighted_col = f"_w{i}"
            prob_df = preds.selectExpr(
                "_row_id",
                f"CAST(probability[1] AS DOUBLE) * {member.weight} AS {weighted_col}",
            )
            result_df = result_df.join(prob_df, on="_row_id", how="inner")

        weighted_cols = [f"_w{i}" for i in range(len(self.members))]
        result_df = result_df \
            .withColumn("ensemble_score", sum(F.col(c) for c in weighted_cols)) \
            .withColumn("ensemble_prediction",
                        (F.col("ensemble_score") >= 0.5).cast(DoubleType()))

        final_df = df_indexed \
            .join(result_df.select("_row_id", "ensemble_score", "ensemble_prediction"),
                  on="_row_id", how="inner") \
            .drop("_row_id")

        df_indexed.unpersist()
        return final_df

    @property
    def is_empty(self):
        return len(self.members) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Drift Detector
# ─────────────────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Wraps CDFRS + SWD + permutation test.
    Exposes last_swd, last_p_value, last_null_dist for the visualizer.
    """

    def __init__(self, ensemble_config, data_config, spark):
        self.ecfg      = ensemble_config
        self.dcfg      = data_config
        self.spark     = spark
        self.reference = None

        # Exposed for visualizer
        self.last_swd       = 0.0
        self.last_p_value   = 1.0
        self.last_null_dist = np.zeros(10)

    def _sample(self, df):
        sampled = cdfrs(df, self.spark, feature_cols=self.dcfg.feature_cols,
                        seed=self.ecfg.seed)
        return extract_sample_to_numpy(
            sampled, self.dcfg.feature_cols, max_samples=self.ecfg.max_samples
        )

    def set_reference(self, df):
        self.reference = self._sample(df)
        print(f"[Drift] Reference set: {self.reference.shape[0]} rows, "
              f"{self.reference.shape[1]} features")

    def check(self, df) -> Tuple[bool, float, float]:
        if self.reference is None:
            raise RuntimeError("Call set_reference() first.")
        X_det = self._sample(df)
        swd, p_val, null_dist = permutation_test_swd_ultra_fast(
            X_ref=self.reference, X_det=X_det,
            n_projections=self.ecfg.n_projections,
            n_permutations=self.ecfg.n_permutations,
            seed=self.ecfg.seed,
        )
        print_swd_report(swd, p_val, null_dist, self.ecfg.drift_alpha)

        # Cache for visualizer
        self.last_swd       = swd
        self.last_p_value   = p_val
        self.last_null_dist = null_dist

        return p_val < self.ecfg.drift_alpha, swd, p_val


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DriftAwareEnsemblePipeline:
    CLASSIFIER_TYPES = ["dt", "rf", "gbt", "lr"]

    def __init__(self, ensemble_config, data_config, spark):
        data_config.validate()
        self.ecfg       = ensemble_config
        self.dcfg       = data_config
        self.spark      = spark
        self.ensemble   = WeightedEnsemble(ensemble_config)
        self.detector   = DriftDetector(ensemble_config, data_config, spark)
        self.source     = WindowSource(data_config, spark)
        self.window_idx = 0

    def _train_best_model(self, train_df, val_df) -> EnsembleMember:
        print(f"[Train] Window {self.window_idx} — {len(self.CLASSIFIER_TYPES)} models...")
        best = None
        for ctype in self.CLASSIFIER_TYPES:
            print(f"  Training {ctype.upper()}...")
            pl    = build_pipeline(self.dcfg.feature_cols, self.dcfg.label_col,
                                   ctype, self.ecfg.seed)
            model = pl.fit(train_df)
            auc, f1 = evaluate_model(model, val_df, self.dcfg.label_col)
            print(f"    {ctype.upper()} → AUC={auc:.4f}  F1={f1:.4f}")
            c = EnsembleMember(uuid.uuid4().hex[:8], model, 0.0, auc, f1,
                               self.window_idx, model_type=ctype)
            if best is None or auc > best.train_auc:
                best = c
        print(f"[Train] Best: {best.model_type.upper()}  "
              f"AUC={best.train_auc:.4f}  F1={best.train_f1:.4f}")
        return best

    def process_window(self, window_df) -> bool:
        train_df, val_df = window_df.randomSplit([0.8, 0.2], seed=self.ecfg.seed)
        train_df = train_df.cache(); train_df.count()
        val_df   = val_df.cache();   val_df.count()
        retrained = False

        if self.window_idx == 0:
            print("[Pipeline] Window 0: cold start.")
            self.detector.set_reference(window_df)
            best = self._train_best_model(train_df, val_df)
            self.ensemble.add_model(best.model, best.train_auc, best.train_f1,
                                    0, model_type=best.model_type)
            retrained = True
        else:
            print(f"\n[Pipeline] Window {self.window_idx}: checking drift...")
            drifted, swd, p_val = self.detector.check(window_df)
            if drifted:
                print(f"[Pipeline] ⚠️  Drift detected (SWD={swd:.4f}, p={p_val:.4f})")
                best = self._train_best_model(train_df, val_df)
                self.ensemble.add_model(best.model, best.train_auc, best.train_f1,
                                        self.window_idx, model_type=best.model_type)
                self.detector.set_reference(window_df)
                retrained = True
            else:
                print(f"[Pipeline] ✓  No drift (SWD={swd:.4f}, p={p_val:.4f})")

        train_df.unpersist()
        val_df.unpersist()
        self.window_idx += 1
        return retrained

    def predict(self, df) -> DataFrame:
        if self.ensemble.is_empty:
            raise RuntimeError("No models trained yet.")
        return self.ensemble.predict(df)

    def evaluate(self, df) -> dict:
        preds = self.predict(df)
        auc = BinaryClassificationMetrics(
            preds.select("ensemble_score", self.dcfg.label_col)
                 .rdd.map(lambda r: (float(r["ensemble_score"]),
                                     float(r[self.dcfg.label_col])))
        ).areaUnderROC
        f1 = MulticlassClassificationEvaluator(
            labelCol=self.dcfg.label_col,
            predictionCol="ensemble_prediction", metricName="f1",
        ).evaluate(preds)
        accuracy = MulticlassClassificationEvaluator(
            labelCol=self.dcfg.label_col,
            predictionCol="ensemble_prediction", metricName="accuracy",
        ).evaluate(preds)
        metrics = {"auc": auc, "f1": f1, "accuracy": accuracy}
        print("\n[Evaluate] Ensemble metrics:")
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v:.4f}")
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("DriftAwareEnsemble") \
        .master("yarn") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executorEnv.PYSPARK_PYTHON",
                "/home/cwc/Major_Project/major-project/bin/python3") \
        .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON",
                "/home/cwc/Major_Project/major-project/bin/python3") \
        .getOrCreate()

    # ── CONFIGURE YOUR DATASET ────────────────────────────────────────────────

    # PRESET 1: HIGGS — directory of parquet partitions
    higgs_data = DataConfig(
        window_mode  = "directory",
        source_path  = "hdfs://localhost:9000/data/higgs_parquet",
        label_col    = "label",
        feature_cols = [f"x{i}" for i in range(1, 29)],
    )

    # PRESET 2: NF-UQ-NIDS-v2 — single large file split into time windows
    # nids_data = DataConfig(
    #     window_mode  = "split",
    #     source_path  = "hdfs://localhost:9000/data/nf_uq_nids_v2.parquet",
    #     n_splits     = 10,
    #     label_col    = "Label",
    #     feature_cols = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", ...],
    # )

    # PRESET 3: CSV stream
    # csv_data = DataConfig(
    #     window_mode      = "stream",
    #     window_paths     = ["hdfs:///data/jan.csv", "hdfs:///data/feb.csv"],
    #     label_col        = "attack",
    #     feature_cols     = ["col1", "col2", "col3"],
    #     col_rename       = {"_c0": "attack"},
    #     file_format      = "csv",
    # )

    ensemble_cfg = EnsembleConfig(
        max_ensemble_size = 5,
        recency_decay     = 0.5,
        drift_alpha       = 0.05,
        n_projections     = 100,
        n_permutations    = 500,
        max_samples       = 5000,
        seed              = 42,
    )

    # ← swap data config here to change datasets
    pipeline = DriftAwareEnsemblePipeline(ensemble_cfg, higgs_data, spark)
    viz      = EnsembleVisualizer()

    results = []
    for window_idx, window_df in pipeline.source.windows():
        print(f"\n{'='*60}\n  WINDOW {window_idx}\n{'='*60}")

        retrained = pipeline.process_window(window_df)
        metrics   = pipeline.evaluate(window_df)

        # Collect y_true / y_score for ROC curves
        preds_df = pipeline.predict(window_df)
        label_col = pipeline.dcfg.label_col
        y_true  = np.array([r[0] for r in
                             preds_df.select(label_col).collect()], dtype=float)
        y_score = np.array([r[0] for r in
                             preds_df.select("ensemble_score").collect()], dtype=float)

        # Log to visualizer
        viz.log_window(
            window_idx     = window_idx,
            swd            = pipeline.detector.last_swd,
            p_value        = pipeline.detector.last_p_value,
            null_dist      = pipeline.detector.last_null_dist,
            drift_detected = retrained and window_idx > 0,
            metrics        = metrics,
            ensemble       = pipeline.ensemble,
            retrained      = retrained,
            y_true         = y_true,
            y_score        = y_score,
        )

        results.append({"window": window_idx, "retrained": retrained, **metrics})
        print(f"[Window {window_idx}] retrained={retrained}  "
              f"AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}")

    print(f"\n{'='*60}\n  FINAL SUMMARY\n{'='*60}")
    for r in results:
        print(f"  Window {r['window']:2d} | retrained={str(r['retrained']):5s} | "
              f"AUC={r['auc']:.4f}  F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}")

    print("\n[Done] Final ensemble:")
    pipeline.ensemble._print_weights()

    # ── Render all plots ──────────────────────────────────────────────────────
    viz.plot_all()
    # For paper export (uncomment one):
    # viz.plot_all(save_dir="./figures", fmt="pdf")   # LaTeX
    # viz.plot_all(save_dir="./figures", fmt="png")   # Word / slides