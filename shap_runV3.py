#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Runner for Cycle-Time LSTM Project
---------------------------------------
Usage:
  # Pre-training feature selection
  python3 shap_run.py --window 10 --mode pre

  # Post-training explainability
  python3 shap_run.py --window 10 --mode post --serial-id 1234
"""

import os, sys, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Optional dependency
try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class LSTMRegMTL(nn.Module):
    def __init__(self, in_dim, hidden=64, reg_width=64, has_cls=False, cls_width=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head_reg = nn.Sequential(
            nn.Linear(hidden, reg_width),
            nn.ReLU(),
            nn.Linear(reg_width, 1),
        )
        self.has_cls = has_cls
        if has_cls:
            self.head_cls = nn.Sequential(
                nn.Linear(hidden, cls_width),
                nn.ReLU(),
                nn.Linear(cls_width, 1),
            )

    def forward(self, x, task="reg"):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y_reg = self.head_reg(h)
        if task == "reg" or not self.has_cls:
            return y_reg
        y_cls = self.head_cls(h)
        return (y_reg, y_cls) if task == "both" else y_cls


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_features_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features parquet not found: {path}")
    return pd.read_parquet(path)


def load_scaler_artifact(path):
    obj = joblib.load(path)
    if hasattr(obj, "transform") and hasattr(obj, "fit"):
        return obj, None, None
    if isinstance(obj, dict):
        mu = np.asarray(obj.get("mean"), dtype=np.float32)
        std = np.asarray(obj.get("std"), dtype=np.float32)
        return None, mu, std
    raise TypeError(f"Unsupported scaler artifact type: {type(obj)}")


def get_initial_feat_cols(df: pd.DataFrame, TARGET, ID, ORDER) -> list:
    exclude = {TARGET, ID, ORDER, "split", "cycle_timestamp", "timestamp"}
    present = [c for c in df.columns if c not in exclude]
    numeric = list(df[present].select_dtypes(include=[np.number]).columns)
    if len(numeric) == 0:
        raise ValueError("No numeric feature columns found.")
    return numeric

'''
def window_lastW(groups: dict, feat_cols: list, window: int, ORDER, ID, TARGET):
    Xs, ys = [], []
    for sid, g in groups.items():
        g = g.sort_values(ORDER) if ORDER in g.columns else g.sort_values(ID)
        if len(g) < window + 1:
            continue
        vals = g[feat_cols].values.astype("float32")
        tgt = g[TARGET].values.astype("float32")
        for i in range(len(g) - window):
            Xs.append(vals[i:i + window])
            ys.append(tgt[i + window])
    if not Xs:
        raise ValueError(
            f"Not enough rows to form windows (need ≥ {window+1} per group). "
            f"Groups checked: {len(groups)}."
        )
    return np.stack(Xs, axis=0), np.asarray(ys, dtype="float32")
'''

def window_lastW(groups: dict, feat_cols: list, window: int, ORDER, ID, TARGET):
    Xs, ys = [], []
    for sid, g in groups.items():
        g = g.sort_values(ORDER)
        if len(g) < window + 1:
            continue
        vals = g[feat_cols].values.astype("float32")
        tgt = g[TARGET].values.astype("float32")
        for i in range(len(g) - window):
            Xs.append(vals[i:i + window])
            ys.append(tgt[i + window])
    if not Xs:
        raise ValueError(
            f"Not enough rows to form windows (need ≥ {window+1} per group). "
            f"Groups checked: {len(groups)}."
        )
    return np.stack(Xs, axis=0), np.asarray(ys, dtype="float32")

def per_timestep_impute_scale(Xseq, scaler=None, mu=None, std=None):
    N, T, F = Xseq.shape
    if scaler is not None:
        for t in range(T):
            Xt = Xseq[:, t, :]
            Xt[~np.isfinite(Xt)] = np.nan
            col_mean = np.nanmean(Xt, axis=0)
            inds = np.where(~np.isfinite(Xt))
            if inds[0].size:
                Xt[inds] = np.take(col_mean, inds[1])
            Xseq[:, t, :] = scaler.transform(Xt)
        return Xseq.astype(np.float32)
    if mu is None or std is None:
        raise ValueError("Need either sklearn scaler or (mu,std) arrays.")
    std_safe = np.maximum(std, 1e-8)
    MU = np.broadcast_to(mu, (N, T, F))
    STD = np.broadcast_to(std_safe, (N, T, F))
    X = Xseq.copy()
    X[~np.isfinite(X)] = np.nan
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = MU[nan_mask]
    X = (X - MU) / STD
    return X.astype(np.float32)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--serial-id", default=None)
    ap.add_argument("--mode", choices=["pre", "post"], default="pre",
                    help="SHAP mode: pre=feature selection, post=explainability")
    args = ap.parse_args()

    OUT_DIR = Path(f"outputs/shap_{args.mode}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FEATURES_PARQUET = Path("outputs/features_spark.parquet")
    SCALER_PATH = Path("outputs/lstm_scaler.joblib")
    MODEL_PATH = Path("outputs/lstm_cycle_time.pt")
    FEATS_TXT = Path("outputs/lstm_features.txt")

    TARGET, ID, ORDER = "CycleTime_sec", "serial_id", "cycle_number"

    df = load_features_df(FEATURES_PARQUET).copy()
    df = df.dropna(subset=[TARGET])

    # Auto-detect ID column
    if "serial_id" in df.columns:
        ID = "serial_id"
    else:
        # fallback: pick first non-numeric column or "A" if present
        obj_cols = list(df.select_dtypes(exclude=[np.number]).columns)
        if "A" in df.columns:
            ID = "A"
        elif len(obj_cols) > 0:
            ID = obj_cols[0]
        else:
            # as last resort, create dummy group
            ID = "_group"
            df[ID] = 0
        print(f"[INFO] Using '{ID}' as the grouping column for serial IDs.")

    feat_cols = get_initial_feat_cols(df, TARGET, ID, ORDER)
    tr_df = df[df["split"] == "train"] if "split" in df.columns else df

    # Detect valid serial_ids
    counts = df.groupby(ID).size()
    valid_serials = counts[counts >= args.window + 1]
    if args.mode == "post":
        print("\n📊 Valid serial_id candidates (>= W+1 rows):")
        print(valid_serials.sort_values(ascending=False).head(15))
        if len(valid_serials) == 0:
            raise SystemExit("❌ No serial_id groups have enough rows for post-training SHAP. Try lowering --window.")
        if args.serial_id is None:
            args.serial_id = valid_serials.index[0]
            print(f"\n[INFO] Using default serial_id={args.serial_id}\n")

    # Grouping
    groups = {}
    if args.serial_id is not None:
        groups[args.serial_id] = df[df[ID] == args.serial_id]
    else:
        for sid, g in df.groupby(ID):
            groups[sid] = g

    ID = "A"
    ORDER = "B"

    if ORDER not in df.columns:
        df[ORDER] = df.groupby(ID).cumcount()

    # Filter train or full data
    tr_df = df[df["split"] == "train"] if "split" in df.columns else df

    # Build groups by serial_id (A)
    groups = {}
    for sid, g in df.groupby(ID):
        g = g.sort_values(ORDER)
        groups[sid] = g

    print(f"[INFO] Grouped by '{ID}' with ORDER='{ORDER}' → {len(groups)} groups total.")

    scaler, mu_from_art, std_from_art = load_scaler_artifact(SCALER_PATH)
    if scaler is None and mu_from_art is not None and FEATS_TXT.exists():
        feat_cols = [ln.strip() for ln in FEATS_TXT.read_text().splitlines() if ln.strip()]

    Xseq, y = window_lastW(groups, feat_cols, args.window, ORDER, ID, TARGET)
    Xseq = per_timestep_impute_scale(Xseq, scaler, mu_from_art, std_from_art)
    X_last = Xseq[:, -1, :]

    # Load model (for post mode)
    state = torch.load(MODEL_PATH, map_location="cpu")
    hidden_ckpt = int(state["lstm.weight_ih_l0"].shape[0] // 4)
    input_ckpt = int(state["lstm.weight_ih_l0"].shape[1])
    has_cls = any(k.startswith("head_cls.") for k in state.keys())
    model = LSTMRegMTL(in_dim=input_ckpt, hidden=hidden_ckpt, has_cls=has_cls)
    state_pruned = {k: v for k, v in state.items() if not k.startswith("head_cls.")}
    model.load_state_dict(state_pruned, strict=False)
    model.eval()

    if not HAVE_SHAP:
        print("⚠️  SHAP not installed. Run: pip install shap matplotlib")
        sys.exit(0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge

    # ---------------- SHAP MODES ----------------
    if args.mode == "pre":
        print(f"\n[INFO] Running SHAP in PRE-TRAINING mode → feature selection")
        ridge = Ridge(alpha=1.0).fit(X_last, y)
        explainer = shap.KernelExplainer(ridge.predict, X_last)
        shap_values = explainer.shap_values(X_last, nsamples="auto")

    else:  # post
        print(f"\n[INFO] Running SHAP in POST-TRAINING mode → model explainability for serial_id={args.serial_id}")
        def model_predict(X_np):
            X_t = torch.tensor(X_np, dtype=torch.float32)
            with torch.no_grad():
                preds = model(X_t).numpy().flatten()
            return preds
        explainer = shap.KernelExplainer(model_predict, X_last)
        shap_values = explainer.shap_values(X_last, nsamples="auto")

    # ---------------- Plotting ----------------
    plt.figure()
    shap.summary_plot(shap_values, X_last, feature_names=feat_cols, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_last, feature_names=feat_cols, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n✅ SHAP {args.mode} artifacts saved to {OUT_DIR.resolve()}")
    print("Files:")
    print(" - shap_summary_bar.png")
    print(" - shap_beeswarm.png\n")


if __name__ == "__main__":
    main()
