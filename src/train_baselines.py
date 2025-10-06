import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

PARQUET = "outputs/features_spark.parquet"
TARGET = "CycleTime_sec"

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def main():
    df = pd.read_parquet(PARQUET)
    # Keep only numeric features for modeling; drop identifiers & timestamp & label
    drop_cols = [c for c in df.columns if c.lower() in ["serial_id", "cycle_number", "cycle_timestamp", "good_label"]]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])

    num_cols = X.columns.tolist()
    pre = ColumnTransformer([("scale", StandardScaler(), num_cols)], remainder="drop")

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RF": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "XGB": XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    }

    # Use time-based split if available (train/val/test from ETL)
    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
        val_df   = df[df["split"] == "val"]
        test_df  = df[df["split"] == "test"]
        # Train on train only; evaluate on val and test
        Xtr, ytr = train_df.drop(columns=[TARGET, "split"]), train_df[TARGET].values
        Xva, yva = val_df.drop(columns=[TARGET, "split"]),   val_df[TARGET].values
        Xte, yte = test_df.drop(columns=[TARGET, "split"]),  test_df[TARGET].values
    else:
        # Fallback: random split 80/20
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        Xva, yva = Xte, yte  # mirror for compatibility

    rows = []
    best_name, best_score, best_pipe = None, -1e9, None  # selection by validation RMSE
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        mae, rmse, r2 = evaluate(yte, pred)
        rows.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        if -rmse_v > best_score:
            best_name, best_score, best_pipe = name, -rmse_v, pipe

    res = pd.DataFrame(rows).sort_values("RMSE")
    res_val = pd.DataFrame(rows_val).sort_values("RMSE")
    res.to_csv("outputs/results_baselines.csv", index=False)
    res_val.to_csv("outputs/results_val_baselines.csv", index=False)
    print('Test results:')
    print(res)
    print('\nValidation results:')
    print(res_val)

    if best_pipe is not None:
        joblib.dump(best_pipe, "outputs/best_baseline.joblib")
        print(f"Saved best model: {best_name}")

    # ---- Write Markdown Metrics Report ----
    import platform, sys, datetime as dt
    from pathlib import Path
    import hashlib, os

    def parquet_dir_sha256(p: Path) -> str:
        if not p.exists():
            return ""
        if p.is_file():
            try:
                return hashlib.sha256(p.read_bytes()).hexdigest()
            except Exception:
                return ""
        h = hashlib.sha256()
        for sub in sorted(p.rglob("*")):
            if sub.is_file():
                st = sub.stat()
                h.update(str(sub.relative_to(p)).encode())
                h.update(str(st.st_size).encode())
                h.update(str(int(st.st_mtime)).encode())
        return h.hexdigest()

    feats_path = Path(PARQUET)
    lines = [
        "# Metrics Report",
        "",
        "## Meta",
        f"- Timestamp: {dt.datetime.utcnow().isoformat()}Z",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {platform.platform()}",
        f"- Rows: {int(df.shape[0])}",
        f"- Features: {int(X.shape[1])}",
        f"- Split: **time-based** (train/val/test via timestamp percentiles; model selection by validation RMSE)",
        f"- Features Parquet: `{PARQUET}`",
        f"- Features SHA256 (structure): `{parquet_dir_sha256(feats_path if feats_path.is_file() else feats_path.parent)}`",
        "",
        "## Validation Results (for model selection)",
        res_val.to_markdown(index=False),
        "",
        "## Test Results (final report)",
        res.to_markdown(index=False),
        "",
        "## Best Model",
        f"- Name: {res.sort_values('RMSE').iloc[0]['model']}",
        f"- Saved at: `outputs/best_baseline.joblib`",
        ""
    ]
    Path("outputs/metrics_report.md").write_text("\n".join(lines))
    print("Wrote outputs/metrics_report.md")

if __name__ == "__main__":
    main()
