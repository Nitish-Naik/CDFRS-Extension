from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import math
import random
import uuid

import os
os.environ['SPARK_LOCAL_IP'] = "127.0.1.1"
os.environ['HADOOP_CONF_DIR'] = "/usr/local/hadoop/etc/hadoop"
os.environ['YARN_CONF_DIR'] = "/usr/local/hadoop/etc/hadoop"


def ks_distance(df1, df2, col_name, num_points=50):
    quantiles = df1.approxQuantile(col_name, [i/(num_points-1.0) for i in range(num_points)], 0.01)
    

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

def cdfrs_higgs(higgs_df, spark, epsilon=0.05, alpha = 0.10, M=50):

    K = int(((math.log(2/alpha))/(2*epsilon**2)))
    s = max(1, M // max(1, K))

    higgs_blocks = higgs_df.repartition(M)

    higgs_with_subset = higgs_blocks.withColumn("subset_id", (F.rand(seed=42)*s).cast(IntegerType()))

    chosen_subset_id = random.randint(0, s-1)

    D = higgs_with_subset.filter(F.col("subset_id") == chosen_subset_id)
    D = D.drop("subset_id")

    k_cdf = 50

    D_shuffled = D.orderBy(F.rand(seed=123))

    window_spec = Window.orderBy(F.monotonically_increasing_id())

    D_indexed = D_shuffled.withColumn(
        "row_idx",
        F.row_number().over(window_spec) - 1
    )

    D_with_block = D_indexed.withColumn("cdfrs_block_id", (F.col("row_idx") % k_cdf).cast(IntegerType()))

   # ✅ unique path per call so calls don't overwrite each other
    run_id = uuid.uuid4().hex
    cdfrs_output_path = f"hdfs:///data/higgs_cdfrs_blocks_{run_id}"

    D_with_block.write.partitionBy("cdfrs_block_id").mode("overwrite").parquet(cdfrs_output_path)

    cdfrs_blocks_df = spark.read.parquet(cdfrs_output_path).cache()
    cdfrs_blocks_df.count()  # ✅ force materialization immediately so path is locked in

    block_ids = [row.cdfrs_block_id for row in cdfrs_blocks_df.select("cdfrs_block_id").distinct().collect()]
    block_ids = sorted(block_ids)

    A = [f"x{i}" for i in range(1,29)]

    epsilon_A2 = 0.02
    T_max = min(20, len(block_ids))

    def get_S_t(t):
        ids = block_ids[:t]
        return cdfrs_blocks_df.filter(F.col("cdfrs_block_id").isin(ids))

    t = 2
    chosen_t_minus_1 = None

    while t <= T_max:
        S_t_minus_1 = get_S_t(t-1).cache()
        S_t_df = get_S_t(t).cache()

        deltas = []
        for col_name in A:
            print(col_name)
            d = ks_distance(S_t_minus_1.select(col_name), S_t_df.select(col_name), col_name)
            deltas.append(d)

        delta_max = max(deltas) if deltas else 0
        print(f"t={t}, K-S max over A = {delta_max}")
        
        if delta_max <= epsilon_A2:
            chosen_t_minus_1 = t - 1
            break
        
        t += 1

    if chosen_t_minus_1 == None:
        chosen_t_minus_1 = T_max - 1      

    print(f"Chosen sample size {chosen_t_minus_1}")

    final_sample = get_S_t(chosen_t_minus_1)
    print(final_sample)
    print("Finished")

    return final_sample

if __name__ == "__main__":
    spark = SparkSession.builder \
    .appName('CDFRS Sample') \
    .master('yarn') \
    .getOrCreate()
    
    num_features = 28
    cols = ["label"] + [f"x{i}" for i in range(1, num_features+1)]

    higgs_df = spark.read.parquet(
        "hdfs://localhost:9000/data/higgs_parquet/part-00000-cbe60f1e-6bdd-4c21-a969-c413a2625549-c000.snappy.parquet",
        header = False,
        inferSchema = True
    ).toDF(*cols)

    final_sample = cdfrs_higgs(higgs_df)

