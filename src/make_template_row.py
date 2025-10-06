import pandas as pd
from pathlib import Path

PARQUET = "outputs/features_spark.parquet"
TARGET = "CycleTime_sec"
OUT_CSV = "outputs/inference_template.csv"

def main():
    df = pd.read_parquet(PARQUET)
    # Prefer a recent TEST row; fall back to the last row
    if "split" in df.columns and (df["split"]=="test").any():
        df = df[df["split"]=="test"].copy()
    row = df.sort_values(by=[c for c in ["cycle_timestamp","cycle_number"] if c in df.columns]).tail(1)

    # Drop non-feature columns and the target so user can fill values or reuse as-is
    drop_cols = [c for c in row.columns if c.lower() in ["serial_id","cycle_number","cycle_timestamp","good_label","split", TARGET.lower()]]
    Xrow = row.drop(columns=[c for c in drop_cols if c in row.columns], errors="ignore")
    if TARGET in Xrow.columns:
        Xrow = Xrow.drop(columns=[TARGET])

    # Save as CSV template for baseline inference
    Xrow.to_csv(OUT_CSV, index=False)
    print(f"Wrote template to {OUT_CSV}. You can edit values and run:\n"
          f"  python src/inference.py --model baseline --input {OUT_CSV}")

if __name__ == "__main__":
    main()
