import yaml
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType
from utils import parse_array

parse_udf = F.udf(parse_array, ArrayType(ArrayType(DoubleType())))

def summarize_vector(col_arr):
    return {
        "len": F.size(col_arr),
        "min": F.array_min(col_arr),
        "max": F.array_max(col_arr),
        "mean": F.aggregate(col_arr, F.lit(0.0), lambda acc,x: acc + x) / F.when(F.size(col_arr) > 0, F.size(col_arr)).otherwise(F.lit(1.0)),
        "std": F.sqrt(F.when(F.size(col_arr) > 1, 
                             (F.aggregate(col_arr, F.lit(0.0), lambda acc,x: acc + (x * x)) 
                              - (F.aggregate(col_arr, F.lit(0.0), lambda acc,x: acc + x) ** 2) / F.size(col_arr)) 
                             / (F.size(col_arr) - 1)).otherwise(F.lit(0.0))),
        "first": F.element_at(col_arr, 1),
        "last": F.element_at(col_arr, -1),
        "sum": F.aggregate(col_arr, F.lit(0.0), lambda acc,x: acc + x),
        "slope": (F.element_at(col_arr, -1) - F.element_at(col_arr, 1))
    }

def summarize_ts_column(df, colname):
    parsed = df.withColumn(f"{colname}__parsed", parse_udf(F.col(colname)))
    exploded = parsed.withColumn(f"{colname}__ch", F.explode_outer(F.col(f"{colname}__parsed")))
    feats = summarize_vector(F.col(f"{colname}__ch"))
    agg = exploded.groupBy(*[c for c in df.columns if c != colname]).agg(
        F.avg(feats["len"]).alias(f"{colname}_len_mean"),
        F.max(feats["len"]).alias(f"{colname}_len_max"),
        F.avg(feats["min"]).alias(f"{colname}_min_mean"),
        F.avg(feats["max"]).alias(f"{colname}_max_mean"),
        F.avg(feats["mean"]).alias(f"{colname}_mean_mean"),
        F.avg(feats["std"]).alias(f"{colname}_std_mean"),
        F.avg(feats["first"]).alias(f"{colname}_first_mean"),
        F.avg(feats["last"]).alias(f"{colname}_last_mean"),
        F.avg(feats["sum"]).alias(f"{colname}_sum_mean"),
        F.avg(feats["slope"]).alias(f"{colname}_slope_mean")
    )
    return agg

def main():
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)

    ts_cols = cfg["time_series_columns"]
    id_col = cfg["id_column"]
    cyc_col = cfg["cycle_number_column"]
    ts_time = cfg["timestamp_column"]
    good_bad = cfg.get("good_bad_label_column", None)
    csv_path = cfg["csv_path"]
    raw_out = cfg["raw_parquet_path"]
    feat_out = cfg["features_parquet_path"]

    spark = SparkSession.builder.appName("CycleTimeETL").getOrCreate()

    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df = df.withColumn(ts_time, F.to_timestamp(F.col(ts_time)))

    # ---- Time-based split (train/val/test by global timestamp percentiles) ----
    q_row = df.select(F.percentile_approx(F.col(ts_time), [0.70, 0.85], 1000).alias("q")).first()
    t_train, t_val = q_row.q[0], q_row.q[1]
    df = df.withColumn(
        "split",
        F.when(F.col(ts_time) <= F.lit(t_train), F.lit("train"))
         .when(F.col(ts_time) <= F.lit(t_val), F.lit("val"))
         .otherwise(F.lit("test"))
    )

    # Compute cycle time (seconds) = lead(timestamp)-timestamp within id
    order_col = F.when(F.col(cyc_col).isNotNull(), F.col(cyc_col)).otherwise(F.col(ts_time))
    w = Window.partitionBy(id_col).orderBy(order_col)
    df = df.withColumn("next_ts", F.lead(F.col(ts_time)).over(w))
    df = df.withColumn("CycleTime_sec", F.when(F.col("next_ts").isNotNull(),
                                               F.col("next_ts").cast("long") - F.col(ts_time).cast("long")).otherwise(None).cast("double"))
    df.write.mode("overwrite").parquet(raw_out)

    # Summarize all time-series columns and join features
    feat_df = df
    for c in ts_cols:
        if c in feat_df.columns:
            feat_df = summarize_ts_column(feat_df, c)
        else:
            print(f"WARNING: time-series column '{c}' not found in CSV.")

    # Remove raw TS string columns from output
    feat_cols = [c for c in feat_df.columns if c not in ts_cols]
    final = feat_df.select(*[c for c in feat_cols if c in feat_df.columns])

    # Write features
    final.write.mode("overwrite").parquet(feat_out)

    # Data dictionary
    rows = [f"| {name} | {dtype} |" for name, dtype in final.dtypes]
    with open("outputs/data_dictionary.md","w") as f:
        f.write("\n".join(["# Data Dictionary","","| Column | Type |","|---|---|"] + rows))

    spark.stop()
    print(f"Wrote raw to {raw_out} and features to {feat_out}")

if __name__ == "__main__":
    main()
