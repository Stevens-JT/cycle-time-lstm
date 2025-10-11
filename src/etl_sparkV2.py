# src/etl_spark.py
import yaml
import importlib.util
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType


def summarize_vector(col_arr):
    # vector-wise summaries for a single channel (array<double>)
    return {
        "len": F.size(col_arr),
        "min": F.array_min(col_arr),
        "max": F.array_max(col_arr),
        "mean": (
            F.aggregate(col_arr, F.lit(0.0), lambda acc, x: acc + x)
            / F.when(F.size(col_arr) > 0, F.size(col_arr)).otherwise(F.lit(1.0))
        ),
        "std": F.sqrt(
            F.when(
                F.size(col_arr) > 1,
                (
                    F.aggregate(col_arr, F.lit(0.0), lambda acc, x: acc + (x * x))
                    - (
                        F.aggregate(col_arr, F.lit(0.0), lambda acc, x: acc + x) ** 2
                    )
                    / F.size(col_arr)
                )
                / (F.size(col_arr) - 1),
            ).otherwise(F.lit(0.0))
        ),
        "first": F.element_at(col_arr, 1),
        "last": F.element_at(col_arr, -1),
        "sum": F.aggregate(col_arr, F.lit(0.0), lambda acc, x: acc + x),
        "slope": (F.element_at(col_arr, -1) - F.element_at(col_arr, 1)),
    }


def summarize_ts_column(df, colname, parse_udf):
    # parse -> explode channels -> aggregate summaries (mean across channels)
    parsed = df.withColumn(f"{colname}__parsed", parse_udf(F.col(colname)))
    exploded = parsed.withColumn(
        f"{colname}__ch", F.explode_outer(F.col(f"{colname}__parsed"))
    )
    feats = summarize_vector(F.col(f"{colname}__ch"))
    # group by everything *except* the original raw TS column
    group_keys = [c for c in df.columns if c != colname]
    agg = exploded.groupBy(*group_keys).agg(
        F.avg(feats["len"]).alias(f"{colname}_len_mean"),
        F.max(feats["len"]).alias(f"{colname}_len_max"),
        F.avg(feats["min"]).alias(f"{colname}_min_mean"),
        F.avg(feats["max"]).alias(f"{colname}_max_mean"),
        F.avg(feats["mean"]).alias(f"{colname}_mean_mean"),
        F.avg(feats["std"]).alias(f"{colname}_std_mean"),
        F.avg(feats["first"]).alias(f"{colname}_first_mean"),
        F.avg(feats["last"]).alias(f"{colname}_last_mean"),
        F.avg(feats["sum"]).alias(f"{colname}_sum_mean"),
        F.avg(feats["slope"]).alias(f"{colname}_slope_mean"),
    )
    return agg


def main():
    # -------------------
    # Load config
    # -------------------
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ts_cols = cfg["time_series_columns"]            # 12 TS columns (arrays)
    id_col = cfg.get("id_column")                   # "A"
    cyc_col = cfg.get("cycle_number_column")        # "B"
    ts_time = cfg["timestamp_column"]               # "C"
    csv_path = cfg["csv_path"]
    raw_out = cfg["raw_parquet_path"]
    feat_out = cfg["features_parquet_path"]

    # -------------------
    # Spark session
    # -------------------
    spark = (
        SparkSession.builder.appName("CycleTimeETL")
        # .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    # Be lenient about casts (avoid ANSI hard failures)
    spark.conf.set("spark.sql.ansi.enabled", "false")

    # -------------------
    # Ship utils.py to workers & register UDFs *after* shipping
    # -------------------
    utils_path = Path(__file__).parent / "utils.py"
    spark.sparkContext.addPyFile(str(utils_path))

    spec = importlib.util.spec_from_file_location("utils", str(utils_path))
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    parse_udf = F.udf(utils.parse_array, ArrayType(ArrayType(DoubleType())))

    # -------------------
    # Read CSV
    # -------------------
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # -------------------
    # Robust timestamp parsing (US-style + ISO fallbacks)
    # -------------------
    formats = [
        "M/d/yyyy H:mm",
        "M/d/yyyy HH:mm",
        "M/d/yyyy H:mm:ss",
        "M/d/yyyy HH:mm:ss",
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd'T'HH:mm:ss",
    ]
    parsed_ts = [F.to_timestamp(F.col(ts_time), fmt) for fmt in formats]
    df = df.withColumn(ts_time, F.coalesce(*parsed_ts))

    # Ensure we have at least some parsed timestamps
    df_nonnull = df.filter(F.col(ts_time).isNotNull())
    if df_nonnull.rdd.isEmpty():
        raise ValueError(
            f"No timestamps could be parsed in column '{ts_time}'. "
            "Update the timestamp formats in etl_spark.py or add them in config.yaml."
        )

    # -------------------
    # Time-based split via global percentiles
    # -------------------
    q_row = df_nonnull.select(
        F.percentile_approx(F.col(ts_time), [0.70, 0.85], 1000).alias("q")
    ).first()
    t_train, t_val = q_row.q[0], q_row.q[1]

    df = df.withColumn(
        "split",
        F.when(F.col(ts_time) <= F.lit(t_train), F.lit("train"))
        .when(F.col(ts_time) <= F.lit(t_val), F.lit("val"))
        .otherwise(F.lit("test")),
    )

    # -------------------
    # Cycle time (seconds) from next timestamp, ordered by cycle number if present else timestamp
    # -------------------
    order_expr = F.when(
        F.col(cyc_col).isNotNull(), F.col(cyc_col).cast("long")
    ).otherwise(F.col(ts_time).cast("long"))

    # single machine => global order
   # w = Window.orderBy(order_expr)
    w = Window.orderBy(order_expr, F.col(ts_time).cast("long"), F.col(id_col if id_col else "A"))

    # lookahead columns
    next_ts  = F.lead(F.col(ts_time)).over(w)
    next_cyc = F.lead(F.col(cyc_col).cast("long")).over(w)

    # STRICT: only valid if next cycle is exactly +1 (no time-only fallback here)
    valid_sequential = (
        F.col(cyc_col).isNotNull() &
        next_cyc.isNotNull() &
        ((next_cyc - F.col(cyc_col).cast("long")) == 1)
    )

    df = df.withColumn("next_ts", next_ts)

    # compute delta ONLY for valid sequential pairs
    df = df.withColumn(
        "CycleTime_sec",
        F.when(
            valid_sequential & F.col("next_ts").isNotNull(),
            F.col("next_ts").cast("long") - F.col(ts_time).cast("long")
        ).otherwise(F.lit(None).cast("double"))
    )

    # cap to single-cycle durations
    max_gap = int(cfg.get("max_gap_sec", 90))  # adjust in config.yaml
    df = df.withColumn(
        "CycleTime_sec",
        F.when(
            (F.col("CycleTime_sec") > 0) & (F.col("CycleTime_sec") <= F.lit(max_gap)),
            F.col("CycleTime_sec")
        ).otherwise(F.lit(None).cast("double"))
    )

    # (optional but recommended) drop rows without a valid target
    # so they can't leak into later steps
    df = df.filter(F.col("CycleTime_sec").isNotNull())

    print("DEBUG total rows:", df.count())
    print("DEBUG > max_gap rows:", df.filter(F.col("CycleTime_sec") > F.lit(max_gap)).count())
    print("DEBUG sample bad rows:")
    for r in df.filter(F.col("CycleTime_sec") > F.lit(max_gap))\
               .select("A","B","C","CycleTime_sec").limit(5).collect():
        print(r)


    '''
    order_expr = F.when(
        F.col(cyc_col).isNotNull(), F.col(cyc_col).cast("long")
    ).otherwise(F.col(ts_time).cast("long"))

    # Single-machine (no/empty id) => global order; else partition by id
    #id_name = (id_col or "").strip()
    #if id_name.lower() in ("", "none", "null"):
    #    w = Window.orderBy(order_expr)
    #else:
    #    w = Window.partitionBy(F.col(id_name)).orderBy(order_expr)

    #df = df.withColumn("next_ts", F.lead(F.col(ts_time)).over(w))

    # single machine => global order; else partition by machine id if you ever add multiple machines
    w = Window.orderBy(order_expr)

    next_ts  = F.lead(F.col(ts_time)).over(w)
    next_cyc = F.lead(F.col(cyc_col).cast("long")).over(w)

    # only valid if next cycle is exactly +1 when cycle numbers exist; if cycle numbers are null, we’ll allow it and let a time-gap cap (step 2) handle it
    valid_sequential = (
        (F.col(cyc_col).isNotNull()) & (next_cyc.isNotNull()) &
        ((next_cyc - F.col(cyc_col).cast("long")) == 1)
    ) #| (F.col(cyc_col).isNull())

    df = df.withColumn("next_ts", next_ts)

    df = df.withColumn(
        "CycleTime_sec",
        F.when(
            F.col("next_ts").isNotNull(),
            F.col("next_ts").cast("long") - F.col(ts_time).cast("long"),
        )
        #.otherwise(None)
        .otherwise(F.lit(None)
        .cast("double")),
    )

    max_gap = cfg.get("max_gap_sec", 90)  # eliminate outliers
    df = df.withColumn(
        "CycleTime_sec",
        F.when(
             (F.col("CycleTime_sec") > 0) & (F.col("CycleTime_sec") <= F.lit(max_gap)),
             F.col("CycleTime_sec"),
        ).otherwise(F.lit(None).cast("double")),
    )
    '''
    # -------------------
    # Persist the raw parquet
    # -------------------
    df.write.mode("overwrite").parquet(raw_out)

    # -------------------
    # Build feature table: summarize each TS column and join progressively
    # -------------------
    feat_df = df
    for c in ts_cols:
        if c in feat_df.columns:
            feat_df = summarize_ts_column(feat_df, c, parse_udf)
        else:
            # Keep going, just warn
            print(f"WARNING: time-series column '{c}' not found in CSV.")

    # Drop raw TS string columns from the final feature table
    out_cols = [c for c in feat_df.columns if c not in ts_cols]
    final = feat_df.select(*out_cols)

    # -------------------
    # Persist features + data dictionary
    # -------------------
    final.write.mode("overwrite").parquet(feat_out)

    rows = [f"| {name} | {dtype} |" for name, dtype in final.dtypes]
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/data_dictionary.md", "w") as f:
        f.write(
            "\n".join(
                ["# Data Dictionary", "", "| Column | Type |", "|---|---|"] + rows
            )
        )

    spark.stop()
    print(f"Wrote raw to {raw_out} and features to {feat_out}")


if __name__ == "__main__":
    main()
