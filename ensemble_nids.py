"""
Weighted Ensemble Training Pipeline — NF-UQ-NIDS-v2 Edition
=============================================================
Adapted from ensemble_higgs.py.

Changes vs HIGGS version
--------------------------
- EnsembleConfig.feature_cols now defaults to NIDS_FEATURE_COLS (41 cols).
- EnsembleConfig.swd_cols added — the subset of feature_cols used for SWD
  drift detection (excludes subnet identifiers).
- DriftDetector uses cdfrs_nids and swd_nids instead of cdfrs_higgs/swd4.
- Partition paths now read from the HDFS manifest written by partition_nids.py
  rather than being hard-coded.
- Label column is "label" (int 0/1) — same convention as HIGGS, so
  all evaluator code is identical.
- EnsemblePlotter.generate_all_plots() labels updated for NIDS context.
"""

import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
    LogisticRegression,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from cdfrs_nids import cdfrs_nids, NIDS_FEATURE_COLS, SWD_COLS
from swd_nids import (
    extract_sample_to_numpy,
    permutation_test_swd_ultra_fast,
    print_swd_report,
)
from academic_plots_nids import EnsemblePlotter


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleConfig:
    max_ensemble_size: int   = 5
    recency_decay:     float = 0.5
    drift_alpha:       float = 0.05
    n_projections:     int   = 100
    n_permutations:    int   = 500
    max_samples:       int   = 5000
    label_col:         str   = "label"

    # ML feature columns — excludes src_subnet/dst_subnet which are
    # IP-derived identifiers that cause trivial perfect separation (AUC=1)
    # when train and test come from the same network capture environment.
    feature_cols: List[str] = field(
        default_factory=lambda: [c for c in NIDS_FEATURE_COLS
                                 if c not in {"src_subnet", "dst_subnet"}]
    )

    # Subset used for SWD (excludes high-cardinality subnet IDs)
    swd_cols: List[str] = field(default_factory=lambda: list(SWD_COLS))

    model_save_base: str = "hdfs:///models/ensemble_nids"
    seed:            int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Member  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleMember:
    model_id:   str
    model:      PipelineModel
    weight:     float
    train_auc:  float
    train_f1:   float
    window_idx: int


# ─────────────────────────────────────────────────────────────────────────────
# Model Builders  (unchanged — label/feature col names are config-driven)
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    feature_cols: List[str],
    label_col: str,
    classifier_type: str = "dt",
    seed: int = 42,
) -> Pipeline:
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,   # center to zero mean
        withStd=True,    # scale to unit variance
    )
    # Note: StandardScaler produces DenseVector output when withMean=True,
    # which is compatible with DT/RF/GBT/LR classifiers.
    classifiers = {
        "dt": DecisionTreeClassifier(
            labelCol=label_col, featuresCol="features", maxDepth=10, seed=seed,
        ),
        "rf": RandomForestClassifier(
            labelCol=label_col, featuresCol="features", numTrees=50, maxDepth=10, seed=seed,
        ),
        "gbt": GBTClassifier(
            labelCol=label_col, featuresCol="features", maxIter=20, maxDepth=6, seed=seed,
        ),
        "lr": LogisticRegression(
            labelCol=label_col, featuresCol="features", maxIter=100, regParam=0.01,
        ),
    }
    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier_type '{classifier_type}'.")
    return Pipeline(stages=[assembler, scaler, classifiers[classifier_type]])


def evaluate_model(
    model: PipelineModel,
    test_df: DataFrame,
    label_col: str,
) -> Tuple[float, float, float]:
    """Returns (AUC-ROC, F1, Accuracy)."""
    preds = model.transform(test_df)
    auc = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC",
    ).evaluate(preds)
    f1 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1",
    ).evaluate(preds)
    accuracy = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="accuracy",
    ).evaluate(preds)
    return auc, f1, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Recency Weights  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_recency_weights(k: int, decay: float) -> List[float]:
    raw = [decay ** (k - 1 - i) for i in range(k)]
    total = sum(raw)
    return [w / total for w in raw]


# ─────────────────────────────────────────────────────────────────────────────
# WeightedEnsemble  (unchanged from ensemble_higgs.py)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedEnsemble:
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.members: List[EnsembleMember] = []

    def add_model(self, model, train_auc, train_f1, window_idx):
        member = EnsembleMember(
            model_id=uuid.uuid4().hex[:8], model=model, weight=0.0,
            train_auc=train_auc, train_f1=train_f1, window_idx=window_idx,
        )
        self.members.append(member)
        if len(self.members) > self.config.max_ensemble_size:
            evicted = self.members.pop(0)
            print(f"[Ensemble] Evicted model {evicted.model_id} (window {evicted.window_idx})")
        self._recompute_weights()
        print(
            f"[Ensemble] Added {member.model_id} | "
            f"AUC={train_auc:.4f} F1={train_f1:.4f} | size={len(self.members)}"
        )
        self._print_weights()

    def _recompute_weights(self):
        weights = compute_recency_weights(len(self.members), self.config.recency_decay)
        for member, w in zip(self.members, weights):
            member.weight = w

    def _print_weights(self):
        for m in self.members:
            print(f"  model={m.model_id}  win={m.window_idx}  w={m.weight:.4f}  AUC={m.train_auc:.4f}")

    def predict(self, df: DataFrame) -> DataFrame:
        """
        Weighted ensemble prediction — transform-then-select, no cross-plan refs.

        The previous "single-pass" approach failed because column references like
        transformed["probability"] carry the query plan ID of `transformed`, but
        withColumn() tries to resolve them against `result`'s plan (df_cached).
        Spark raises MISSING_ATTRIBUTES because the two plans are disjoint.

        Fix: for each model, transform df_cached to get the probability column,
        immediately SELECT out a named scalar column (_wi) — resolving everything
        within that plan — then join back to result on a stable row key.
        We force SORT-MERGE join (hint="merge") so Spark never attempts a
        broadcast regardless of estimated table size.
        """
        if not self.members:
            raise RuntimeError("Ensemble is empty.")
        from pyspark.ml.functions import vector_to_array
        from pyspark.sql.types import DoubleType, LongType

        # Materialize input once with a stable monotonic row key
        df_keyed = df.withColumn("_rid", F.monotonically_increasing_id()).cache()
        df_keyed.count()

        # Build a DataFrame of just (_rid, _w0, _w1, ...) by joining each
        # model's weighted-prob column on _rid with sort-merge join
        scores_df = df_keyed.select("_rid")

        for i, member in enumerate(self.members):
            # Transform entirely within its own plan, select only what we need
            prob_col = (
                member.model.transform(df_keyed)
                .select(
                    F.col("_rid"),
                    (vector_to_array(F.col("probability")).getItem(1)
                     .cast(DoubleType()) * F.lit(float(member.weight)))
                    .alias(f"_w{i}"),
                )
            )
            # Sort-merge join — guaranteed no broadcast attempt
            scores_df = scores_df.hint("merge").join(prob_col, on="_rid", how="inner")

        # Ensemble score and prediction — pure column arithmetic, no join
        weighted_cols = [f"_w{i}" for i in range(len(self.members))]
        scores_df = (
            scores_df
            .withColumn("ensemble_score", sum(F.col(c) for c in weighted_cols))
            .withColumn("ensemble_prediction",
                        (F.col("ensemble_score") >= 0.5).cast(DoubleType()))
            .drop(*weighted_cols)
        )

        # Attach scores back to the original rows via sort-merge join on _rid
        result = (
            df_keyed
            .hint("merge")
            .join(scores_df, on="_rid", how="inner")
            .drop("_rid")
        )

        df_keyed.unpersist()
        return result

    @property
    def size(self):
        return len(self.members)

    @property
    def is_empty(self):
        return len(self.members) == 0


# ─────────────────────────────────────────────────────────────────────────────
# DriftDetector  — uses cdfrs_nids + swd_cols instead of cdfrs_higgs
# ─────────────────────────────────────────────────────────────────────────────

class DriftDetector:
    def __init__(self, config: EnsembleConfig, spark: SparkSession):
        self.config    = config
        self.spark     = spark
        self.reference: Optional[np.ndarray] = None
        self.last_swd       = 0.0
        self.last_p_value   = 1.0
        self.last_null_dist = None

    def set_reference(self, df: DataFrame) -> None:
        sample = cdfrs_nids(df, self.spark, feature_cols=self.config.swd_cols)
        self.reference = extract_sample_to_numpy(
            sample, self.config.swd_cols, max_samples=self.config.max_samples
        )
        print(f"[Drift] Reference set: {self.reference.shape[0]} rows, {self.reference.shape[1]} features")

    def check(self, df: DataFrame) -> Tuple[bool, float, float]:
        if self.reference is None:
            raise RuntimeError("Reference not set.")
        sample = cdfrs_nids(df, self.spark, feature_cols=self.config.swd_cols)
        X_det = extract_sample_to_numpy(
            sample, self.config.swd_cols, max_samples=self.config.max_samples
        )
        observed_swd, p_value, null_dist = permutation_test_swd_ultra_fast(
            X_ref=self.reference,
            X_det=X_det,
            n_projections=self.config.n_projections,
            n_permutations=self.config.n_permutations,
            seed=self.config.seed,
        )
        self.last_swd       = observed_swd
        self.last_p_value   = p_value
        self.last_null_dist = null_dist
        print_swd_report(observed_swd, p_value, null_dist, self.config.drift_alpha)
        return p_value < self.config.drift_alpha, observed_swd, p_value


# ─────────────────────────────────────────────────────────────────────────────
# DriftAwareEnsemblePipeline  (logic unchanged; imports differ)
# ─────────────────────────────────────────────────────────────────────────────

class DriftAwareEnsemblePipeline:
    CLASSIFIER_TYPES = ["dt", "rf", "gbt", "lr"]

    def __init__(self, config: EnsembleConfig, spark: SparkSession):
        self.config     = config
        self.spark      = spark
        self.ensemble   = WeightedEnsemble(config)
        self.detector   = DriftDetector(config, spark)
        self.window_idx = 0

    def _train_best_model(self, train_df, val_df) -> EnsembleMember:
        print(f"[Train] Window {self.window_idx} — training {len(self.CLASSIFIER_TYPES)} base models...")
        best: Optional[EnsembleMember] = None
        for ctype in self.CLASSIFIER_TYPES:
            print(f"  Training {ctype.upper()}...")
            pipeline = build_pipeline(
                self.config.feature_cols, self.config.label_col, ctype, self.config.seed
            )
            model = pipeline.fit(train_df)
            auc, f1, acc = evaluate_model(model, val_df, self.config.label_col)
            print(f"    {ctype.upper()} → AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
            candidate = EnsembleMember(
                model_id=uuid.uuid4().hex[:8], model=model, weight=0.0,
                train_auc=auc, train_f1=f1, window_idx=self.window_idx,
            )
            if best is None or auc > best.train_auc:
                best = candidate
        print(f"[Train] Best: AUC={best.train_auc:.4f}  F1={best.train_f1:.4f}")
        return best

    def process_window(self, window_df: DataFrame) -> bool:
        train_df, val_df = window_df.randomSplit([0.8, 0.2], seed=self.config.seed)
        train_df = train_df.cache(); train_df.count()
        val_df   = val_df.cache();   val_df.count()

        retrained = False
        if self.window_idx == 0:
            print("[Pipeline] Window 0: cold start.")
            self.detector.set_reference(window_df)
            best = self._train_best_model(train_df, val_df)
            self.ensemble.add_model(best.model, best.train_auc, best.train_f1, 0)
            retrained = True
        else:
            print(f"\n[Pipeline] Window {self.window_idx}: checking drift...")
            drift_detected, swd, p_val = self.detector.check(window_df)
            if drift_detected:
                print(f"[Pipeline] ⚠️  Drift (SWD={swd:.4f}, p={p_val:.4f}) — retraining.")
                best = self._train_best_model(train_df, val_df)
                self.ensemble.add_model(best.model, best.train_auc, best.train_f1, self.window_idx)
                self.detector.set_reference(window_df)
                retrained = True
            else:
                print(f"[Pipeline] ✓  No drift (SWD={swd:.4f}, p={p_val:.4f}).")

        train_df.unpersist(); val_df.unpersist()
        self.window_idx += 1
        return retrained

    def predict(self, df: DataFrame) -> DataFrame:
        if self.ensemble.is_empty:
            raise RuntimeError("No models trained yet.")
        return self.ensemble.predict(df)

    def evaluate(self, df: DataFrame) -> dict:
        """
        Evaluate ensemble predictions against ground truth.

        Uses BinaryClassificationEvaluator (DataFrame/SQL-native) for AUC
        instead of the MLlib RDD path. The old .rdd.map() approach called
        javaToPython() internally, which triggered a broadcast join attempt
        on the score/label table and OOMed on large windows.

        BinaryClassificationEvaluator expects rawPrediction as a 2-element
        vector [score_class0, score_class1]. We construct that from
        ensemble_score (which is already P(class=1)).
        """
        from pyspark.sql.types import DoubleType, ArrayType
        from pyspark.ml.linalg import VectorUDT
        import pyspark.ml.functions as mlF

        preds = self.predict(df).cache()
        preds.count()

        # Build a 2-element array column [1-p, p] as the rawPrediction proxy,
        # then wrap it as a vector so BinaryClassificationEvaluator accepts it.
        preds_with_raw = preds.withColumn(
            "_raw",
            F.array(
                (F.lit(1.0) - F.col("ensemble_score")).cast(DoubleType()),
                F.col("ensemble_score").cast(DoubleType()),
            ),
        )
        # array_to_vector is available in pyspark.ml.functions (Spark 3.1+)
        preds_with_raw = preds_with_raw.withColumn(
            "_raw_vec", mlF.array_to_vector(F.col("_raw"))
        )

        auc = BinaryClassificationEvaluator(
            labelCol=self.config.label_col,
            rawPredictionCol="_raw_vec",
            metricName="areaUnderROC",
        ).evaluate(preds_with_raw)

        f1 = MulticlassClassificationEvaluator(
            labelCol=self.config.label_col,
            predictionCol="ensemble_prediction",
            metricName="f1",
        ).evaluate(preds)

        accuracy = MulticlassClassificationEvaluator(
            labelCol=self.config.label_col,
            predictionCol="ensemble_prediction",
            metricName="accuracy",
        ).evaluate(preds)

        preds.unpersist()

        metrics = {"auc": auc, "f1": f1, "accuracy": accuracy}
        print("\n[Evaluate] Ensemble metrics:")
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v:.4f}")
        return metrics



# ─────────────────────────────────────────────────────────────────────────────
# Manifest Reader  — discovers window paths written by partition_nids.py
# ─────────────────────────────────────────────────────────────────────────────

def load_window_paths(spark: SparkSession, manifest_path: str) -> List[str]:
    """
    Read the CSV manifest produced by partition_nids.py and return window
    HDFS paths in sorted window_idx order.

    Uses an explicit schema and a glob on part-*.csv to skip the _SUCCESS
    and .crc marker files that coalesce(1).write produces in the directory.
    """
    from pyspark.sql.types import StructType, StructField, StringType as ST, IntegerType as IT
    schema = StructType([
        StructField("path",       ST(), nullable=False),
        StructField("window_idx", IT(), nullable=False),
    ])
    manifest_df = (
        spark.read
        .option("header", True)
        .schema(schema)
        .csv(f"{manifest_path}/")
        .orderBy("window_idx")
    )
    paths = [row["path"] for row in manifest_df.collect()]
    print(f"[Main] Loaded {len(paths)} window paths from manifest.")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ["SPARK_LOCAL_IP"] = "127.0.1.1"
    os.environ["HADOOP_CONF_DIR"] = "/usr/local/hadoop/etc/hadoop"
    os.environ["YARN_CONF_DIR"]   = "/usr/local/hadoop/etc/hadoop"

    spark = (
        SparkSession.builder.appName("DriftAwareEnsemble_NIDS")
        .master("yarn")
        .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON",
                "/home/cwc/Major_Project/major-project/bin/python3")
        .config("spark.executorEnv.PYSPARK_PYTHON",
                "/home/cwc/Major_Project/major-project/bin/python3")
        .config("spark.sql.shuffle.partitions", "200")
        # Disable broadcast joins — with large windows Spark misjudges table
        # sizes and tries to broadcast, causing executor OOM.
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        # Storage memory fraction: reduce from default 0.5 to give more room
        # to execution (shuffle, sort, aggregation). CDFRS no longer caches
        # RDDs in executors, so storage pressure is low.
        .config("spark.memory.storageFraction", "0.2")
        # Allow Spark to spill to disk rather than OOM when execution memory
        # is exhausted — important for sort-merge joins on large windows.
        .config("spark.memory.offHeap.enabled", "false")
        # Increase max task failures before aborting a stage — gives more
        # tolerance if a single executor is slow or briefly unavailable.
        .config("spark.task.maxFailures", "8")
        # Fetch retry settings to handle transient shuffle fetch failures
        # that cascade from executor deaths.
        .config("spark.shuffle.io.maxRetries", "10")
        .config("spark.shuffle.io.retryWait", "30s")
        .getOrCreate()
    )

    MANIFEST = "hdfs://localhost:9000/data/nids_windows/manifest"
    partition_paths = load_window_paths(spark, MANIFEST)

    config = EnsembleConfig(
        max_ensemble_size = 5,
        recency_decay     = 0.5,
        drift_alpha       = 0.05,
        n_projections     = 100,
        n_permutations    = 500,
        max_samples       = 5000,
        label_col         = "label",
        # feature_cols and swd_cols default to NIDS constants
        seed              = 42,
    )

    pipeline = DriftAwareEnsemblePipeline(config=config, spark=spark)
    plotter  = EnsemblePlotter()
    static_model = None

    for i, path in enumerate(partition_paths):
        print(f"\n{'='*60}")
        print(f"  WINDOW {i}: {path.split('/')[-1]}")
        print(f"{'='*60}")

        window_df = spark.read.parquet(path)

        # ── Static baseline: train once on window 0, never retrain ───────────
        if i == 0:
            train_df0, _ = window_df.randomSplit([0.8, 0.2], seed=42)
            static_pipeline = build_pipeline(config.feature_cols, config.label_col, "gbt")
            static_model = static_pipeline.fit(train_df0)

        # ── Dynamic ensemble ──────────────────────────────────────────────────
        retrained = pipeline.process_window(window_df)
        metrics   = pipeline.evaluate(window_df)

        print(f"[Window {i}] retrained={retrained}  AUC={metrics['auc']:.4f}  "
              f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")

        _, _, static_accuracy = evaluate_model(static_model, window_df, config.label_col)

        # ── Update plotter ────────────────────────────────────────────────────
        plotter.windows.append(i)
        plotter.metrics["auc"].append(metrics["auc"])
        plotter.metrics["f1"].append(metrics["f1"])
        plotter.metrics["accuracy"].append(metrics["accuracy"])
        plotter.static_metrics["accuracy"].append(static_accuracy)
        plotter.swd_scores.append(pipeline.detector.last_swd)
        plotter.drift_flags.append(False if i == 0 else retrained)

        if retrained and pipeline.detector.last_null_dist is not None:
            plotter.null_distributions[i] = (
                pipeline.detector.last_null_dist,
                pipeline.detector.last_swd,
            )

        current_weights = {"dt": 0.0, "rf": 0.0, "gbt": 0.0, "lr": 0.0}
        for member in pipeline.ensemble.members:
            model_str = str(member.model.stages[-1]).lower()
            for key in current_weights:
                if key in model_str:
                    current_weights[key] += member.weight
                    break
        for k in current_weights:
            plotter.weights[k].append(current_weights[k])

        preds = pipeline.predict(window_df).select(config.label_col, "ensemble_prediction")
        pd_preds = preds.toPandas()
        plotter.final_y_true.extend(pd_preds[config.label_col].tolist())
        plotter.final_y_pred.extend(pd_preds["ensemble_prediction"].tolist())

    print("\n[Done] Final ensemble state:")
    pipeline.ensemble._print_weights()

    print("\nGenerating Academic Plots...")
    plotter.generate_all_plots()

    spark.stop()