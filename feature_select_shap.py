# feature_select_shap.py
import argparse, numpy as np, pandas as pd, shap
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TARGET = "CycleTime_sec"
ID = "serial_id"
ORDER = "cycle_number"
TS = "cycle_timestamp"

def main():
    p = argparse.ArgumentParser(description="Pre-training SHAP feature selection (Ridge + LinearExplainer).")
    p.add_argument("--features", default="outputs/features_spark.parquet", help="Path to features parquet")
    p.add_argument("--target", default=TARGET)
    p.add_argument("--topk", type=int, default=50, help="Top-K features to keep")
    p.add_argument("--sample", type=int, default=5000, help="Max rows for SHAP background/eval")
    args = p.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load features
    df = pd.read_parquet(args.features)

    # ---- Use train split only for selection (avoid leakage)
    if "split" in df.columns:
        df = df[df["split"] == "train"].copy()

    # ---- Choose numeric candidate features (drop identifiers, time, target)
    drop_cols = {c for c in [ID, ORDER, TS, "split", args.target] if c in df.columns}
    num_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    if len(num_cols) == 0:
        raise SystemExit("No numeric feature columns found for SHAP selection.")

    # ---- X, y
    y = df[args.target].astype("float32").values
    X = df[num_cols].astype("float32").values

    # ---- Simple mean impute (train-only) + scale (for stable ridge)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---- Subsample for SHAP to keep it quick (background + eval)
    n = Xs.shape[0]
    if n > args.sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=args.sample, replace=False)
        Xs_bg = Xs[idx]
    else:
        Xs_bg = Xs

    # ---- Fit a simple ridge regressor
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(Xs, y)

    # ---- SHAP (linear explainer suits linear models)
    explainer = shap.LinearExplainer(model, Xs_bg, feature_perturbation="interventional")
    # evaluate SHAP on a held subset as well (reuse bg if small)
    if n > args.sample:
        eval_idx = idx
    else:
        eval_idx = np.arange(n)
    shap_vals = explainer.shap_values(Xs[eval_idx])
    # mean |SHAP| per feature
    mean_abs = np.abs(shap_vals).mean(axis=0)

    # ---- Rank & persist
    order = np.argsort(-mean_abs)
    ranked = [(num_cols[i], float(mean_abs[i])) for i in order]
    top = ranked[: args.topk]

    # CSV: feature, mean_abs_shap
    pd.DataFrame(top, columns=["feature", "mean_abs_shap"]).to_csv(out_dir / "top_features.csv", index=False)

    # TXT: just the feature names (one per line) — this feeds training/inference/SHAP
    (out_dir / "lstm_features.txt").write_text("\n".join([f for f, _ in top]) + "\n", encoding="utf-8")

    print(f"[OK] Wrote: {out_dir/'top_features.csv'} and {out_dir/'lstm_features.txt'}")
    print(f"[INFO] Selected top-{args.topk} / {len(num_cols)} numeric features.")

if __name__ == "__main__":
    main()
