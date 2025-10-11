# src/train_baselines.py
import pandas as pd, numpy as np, joblib
from pathlib import Path
import platform, sys, datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PARQUET = "outputs/features_spark.parquet"
TARGET = "CycleTime_sec"

# Columns we definitely do NOT want to learn from, even if numeric
EXCLUDE_IF_PRESENT = ["A", "B", "C", "D", "next_ts", "split"]

#def evaluate(y_true, y_pred):
    #mae = mean_absolute_error(y_true, y_pred)
    #rmse = mean_squared_error(y_true, y_pred, squared=False)
    #r2 = r2_score(y_true, y_pred)
    #return mae, rmse, r2

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def select_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Keep numeric features only, then drop known meta columns and the target
    num_df = df.select_dtypes(include=[np.number, "bool"]).copy()
    drop_cols = [c for c in EXCLUDE_IF_PRESENT + [TARGET] if c in num_df.columns]
    return num_df.drop(columns=drop_cols, errors="ignore")

def main():
    df = pd.read_parquet(PARQUET)

    # -------- Diagnostics (light) --------
    print("Columns:", df.columns.tolist())
    print("Rows total:", len(df))
    print("Has 'split' + target:", set(["split", TARGET]).issubset(df.columns))
    if "split" in df.columns:
        print(df["split"].value_counts(dropna=False))
        nn = df.groupby("split")[TARGET].apply(lambda s: s.notna().sum())
        print("Target non-null per split:")
        print(nn)

    # Remove rows without target
    if TARGET not in df.columns:
        raise RuntimeError(f"Target column '{TARGET}' not found in {PARQUET}")
    df = df.dropna(subset=[TARGET])

    # -------- Build time-based split frames (preferred) --------
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        val_df   = df[df["split"] == "val"].copy()
        test_df  = df[df["split"] == "test"].copy()

        Xtr, ytr = select_feature_frame(train_df), train_df[TARGET].values
        Xva, yva = select_feature_frame(val_df),   val_df[TARGET].values
        Xte, yte = select_feature_frame(test_df),  test_df[TARGET].values

        if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
            raise RuntimeError(
                f"Empty split detected: lens -> train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}. "
                f"Check ETL split thresholds or data coverage."
            )
        split_info = "time-based (train/val/test from ETL)"
    else:
        # Fallback (shouldn’t be needed, but kept for robustness)
        from sklearn.model_selection import train_test_split
        Xall = select_feature_frame(df)
        yall = df[TARGET].values
        Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.2, random_state=42)
        # Mirror as "val" for selection; true production uses ETL split
        Xva, yva = Xte, yte
        split_info = "random 80/20 (fallback; ETL 'split' not found)"

    print("Train mean CycleTime_sec:", train_df["CycleTime_sec"].mean())

    # -------- Preprocess: Impute -> Scale --------
    num_cols = Xtr.columns.tolist()
    if not num_cols:
        raise RuntimeError("No numeric feature columns after exclusions. Check ETL features.")
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]), num_cols)
        ],
        remainder="drop"
    )

    # -------- Models --------
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RF": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "XGB": XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ),
    }

    rows_val, rows_test = [], []
    best_name, best_rmse, best_pipe = None, np.inf, None

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(Xtr, ytr)

        # Validation (for model selection)
        pred_val = pipe.predict(Xva)
        mae_v, rmse_v, r2_v = evaluate(yva, pred_val)
        rows_val.append({"model": name, "MAE": mae_v, "RMSE": rmse_v, "R2": r2_v})

        # Keep best by validation RMSE
        if rmse_v < best_rmse:
            best_name, best_rmse, best_pipe = name, rmse_v, pipe

        # Test (for reporting)
        pred_test = pipe.predict(Xte)
        mae_t, rmse_t, r2_t = evaluate(yte, pred_test)
        rows_test.append({"model": name, "MAE": mae_t, "RMSE": rmse_t, "R2": r2_t})

    res_val = pd.DataFrame(rows_val).sort_values("RMSE")
    res_test = pd.DataFrame(rows_test).sort_values("RMSE")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    res_val.to_csv("outputs/results_val_baselines.csv", index=False)
    res_test.to_csv("outputs/results_baselines.csv", index=False)

    print("\nValidation results (for selection):")
    print(res_val)
    print("\nTest results (final):")
    print(res_test)

    if best_pipe is not None:
        joblib.dump(best_pipe, "outputs/best_baseline.joblib")
        print(f"\nSaved best baseline: {best_name} (val RMSE={best_rmse:.4f}) -> outputs/best_baseline.joblib")

    # -------- Markdown report --------
    def parquet_dir_sha256(p: Path) -> str:
        # compute a simple structural hash of the parquet output directory
        if not p.exists():
            return ""
        if p.is_file():
            try:
                import hashlib
                return hashlib.sha256(p.read_bytes()).hexdigest()
            except Exception:
                return ""
        import hashlib
        h = hashlib.sha256()
        for sub in sorted(p.rglob("*")):
            if sub.is_file():
                st = sub.stat()
                h.update(str(sub.relative_to(p)).encode())
                h.update(str(st.st_size).encode())
                h.update(str(int(st.st_mtime)).encode())
        return h.hexdigest()

    lines = [
        "# Baselines — Metrics Report",
        "",
        "## Meta",
        f"- Timestamp: {dt.datetime.utcnow().isoformat()}Z",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {platform.platform()}",
        f"- Rows used (after target dropna): {len(df)}",
        f"- Feature count (after selection): {len(num_cols)}",
        f"- Split strategy: **{split_info}**",
        f"- Features Parquet: `{PARQUET}` (hash: `{parquet_dir_sha256(Path(PARQUET))}`)",
        "",
        "## Validation (used for model selection)",
        res_val.to_markdown(index=False),
        "",
        "## Test (final report)",
        res_test.to_markdown(index=False),
        "",
        "## Best Model",
        f"- Name: {best_name}",
        f"- Saved at: `outputs/best_baseline.joblib`",
        f"- Best Val RMSE: {best_rmse:.6f}",
        "",
    ]
    Path("outputs/metrics_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote outputs/metrics_report.md")

if __name__ == "__main__":
    main()
