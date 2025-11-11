#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drop-in SHAP runner for Cycle Time LSTM project.

Adds a GLOBAL sliding-window fallback so SHAP can run even when per-serial
groups are too small to form W+1-length sequences.

Usage:
    python3 shap_run.py --window 10 [--serial-id SERIAL]

Outputs (in outputs/ by default):
    - shap_summary_bar.png
    - shap_beeswarm.png
    - shap_dependence_1.png
    - shap_dependence_2.png
    - top_features.csv
    - dataset_stats.csv
"""

import os, sys, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Force CPU to avoid CUDA init warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler

# Try SHAP (soft dependency)
try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

# --------------------
# Paths / Constants
# --------------------
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PARQUET = OUT_DIR / "features_spark.parquet"
MODEL_PATH       = OUT_DIR / "lstm_cycle_time.pt"      # checkpoint produced by train_lstm.py
SCALER_PATH      = OUT_DIR / "lstm_scaler.joblib"      # can be sklearn scaler OR dict artifact
FEATS_TXT        = OUT_DIR / "lstm_features.txt"       # feature order from training (one per line)

TOP_FEATS_CSV    = OUT_DIR / "top_features.csv"
DATA_STATS_CSV   = OUT_DIR / "dataset_stats.csv"

FIG_BAR          = OUT_DIR / "shap_summary_bar.png"
FIG_BSWARM       = OUT_DIR / "shap_beeswarm.png"
FIG_DEP1         = OUT_DIR / "shap_dependence_1.png"
FIG_DEP2         = OUT_DIR / "shap_dependence_2.png"

TARGET = "CycleTime_sec"

# We will auto-detect ID/order but default to common names
POSSIBLE_ID    = ["serial_id", "A"]
POSSIBLE_ORDER = ["cycle_number", "B", "C"]  # C is timestamp in many configs


# --------------------
# Model (MTL-compatible)
# --------------------
class LSTMRegMTL(nn.Module):
    def __init__(self, in_dim, hidden=64, reg_width=64, has_cls=False, cls_width=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head_reg = nn.Sequential(nn.Linear(hidden, reg_width), nn.ReLU(), nn.Linear(reg_width, 1))
        self.has_cls = has_cls
        if has_cls:
            self.head_cls = nn.Sequential(nn.Linear(hidden, cls_width), nn.ReLU(), nn.Linear(cls_width, 1))

    def forward(self, x, task="reg"):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y_reg = self.head_reg(h)
        if task == "reg" or not self.has_cls:
            return y_reg
        y_cls = self.head_cls(h)
        return (y_reg, y_cls) if task == "both" else y_cls


# --------------------
# Utilities
# --------------------
def load_features_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features parquet not found: {path}")
    return pd.read_parquet(path)


def load_scaler_artifact(path: Path):
    """
    Returns a tuple (scaler, mu, std, seq_len_saved, feat_cols_saved):
      - If joblib file is an sklearn scaler -> (scaler, None, None, None, None)
      - If joblib file is a dict with {'mean','std','feature_cols','seq_len'} -> (None, mu, std, seq_len, feature_cols)
    """
    if not path.exists():
        return None, None, None, None, None
    obj = joblib.load(path)
    # sklearn scaler?
    if hasattr(obj, "transform") and hasattr(obj, "fit"):
        return obj, None, None, None, None
    # dict artifact?
    if isinstance(obj, dict):
        mu  = np.asarray(obj.get("mean"), dtype=np.float32) if obj.get("mean") is not None else None
        std = np.asarray(obj.get("std"),  dtype=np.float32) if obj.get("std")  is not None else None
        seq_len = int(obj.get("seq_len", 0)) if obj.get("seq_len") is not None else None
        fcols = obj.get("feature_cols")
        return None, mu, std, seq_len, fcols
    raise TypeError(f"Unsupported scaler artifact type: {type(obj)}")


def get_initial_feat_cols(df: pd.DataFrame) -> list:
    """Prefer training feature list; otherwise numeric columns only (excluding ID/order/split/target)."""
    exclude = {TARGET, "split", "cycle_timestamp", "timestamp", "time", "date", "datetime"}
    # we'll exclude ID/order later once we identify them
    present = [c for c in df.columns if c not in exclude]

    # If we have a training feature list, use its order and intersect with present & numeric
    if FEATS_TXT.exists():
        trained = [ln.strip() for ln in FEATS_TXT.read_text().splitlines() if ln.strip()]
        trained = [c for c in trained if c in present]
        num_trained = list(df[trained].select_dtypes(include=[np.number]).columns)
        if len(num_trained) == 0:
            raise ValueError("lstm_features.txt exists but none of its columns are numeric/present.")
        return num_trained

    numeric = list(df[present].select_dtypes(include=[np.number]).columns)
    if len(numeric) == 0:
        raise ValueError("No numeric feature columns found after exclusions.")
    return numeric


def window_lastW(groups: dict, feat_cols: list, window: int, order_col: str, id_col: str, tgt_col: str):
    """Build [N, W, F] windows and next-step target y (float), grouped by id_col."""
    Xs, ys = [], []
    for sid, g in groups.items():
        if order_col in g.columns:
            g = g.sort_values(order_col)
        elif id_col in g.columns:
            # fallback: stable sort by row order per group
            g = g.reset_index(drop=True)
        else:
            continue
        if len(g) < window + 1:
            continue
        vals = g[feat_cols].values.astype("float32")
        tgt  = g[tgt_col].values.astype("float32")
        for i in range(len(g) - window):
            Xs.append(vals[i:i+window])
            ys.append(tgt[i+window])
    if not Xs:
        # Collect tiny-group diagnostics for error message
        counts = {str(k): len(v) for k, v in groups.items()}
        preview = ", ".join([f"{k}:{counts[k]}" for k in list(counts.keys())[:10]])
        need = window + 1
        msg = [
            f"Not enough rows to form windows (need >= {need} per group). Groups checked: {len(groups)}.",
            f"Examples (sid:rows) → {preview}",
            "Try lowering --window or verifying ID/order columns."
        ]
        raise ValueError(" ".join(msg))
    return np.stack(Xs, axis=0), np.asarray(ys, dtype="float32")


def build_global_windows(df: pd.DataFrame, feat_cols: list, window: int, order_col: str, target_col: str):
    """Build [N, window, F] windows over the entire dataset, sorted by order_col (no grouping)."""
    if order_col not in df.columns:
        raise ValueError(f"Global windows need a sortable column; '{order_col}' not found.")
    g = df.sort_values(order_col).reset_index(drop=True)
    vals = g[feat_cols].values.astype("float32")
    tgt  = g[target_col].values.astype("float32")
    Xs, ys = [], []
    for i in range(len(g) - window):
        Xs.append(vals[i:i+window])
        ys.append(tgt[i+window])
    if not Xs:
        raise ValueError(f"Global fallback also failed (len={len(g)}, window={window}).")
    return np.stack(Xs, axis=0), np.asarray(ys, dtype="float32")


def per_timestep_impute_scale(Xseq, scaler=None, mu=None, std=None):
    """
    Xseq: [N, T, F]
    If scaler is provided (sklearn), use scaler.transform per timestep.
    Else use mu/std arrays (broadcasted) for imputation + standardization.
    """
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
        raise ValueError("per_timestep_impute_scale: need either sklearn scaler or (mu,std) arrays.")

    if mu.shape[-1] != F or std.shape[-1] != F:
        raise ValueError(f"Feature count mismatch: X has F={F}, mu/std have {mu.shape[-1]}/{std.shape[-1]}")

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


# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10, help="Sequence length (W)")
    ap.add_argument("--serial-id", default=None, help="Filter to a single serial/group (optional)")
    args = ap.parse_args()

    # 1) Load features
    df = load_features_df(FEATURES_PARQUET).copy()
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])

    # 2) Detect ID/ORDER columns
    id_col = next((c for c in POSSIBLE_ID if c in df.columns), None)
    order_col = next((c for c in POSSIBLE_ORDER if c in df.columns), None)

    # If missing, synthesize a simple order column
    if order_col is None:
        order_col = "__ord__"
        if id_col is None:
            df[order_col] = np.arange(len(df))
        else:
            df[order_col] = df.groupby(id_col).cumcount()

    # 3) Initial numeric feature list (exclude id/order/split/target)
    feat_cols = get_initial_feat_cols(df)
    exclude_cols = {TARGET, "split", id_col, order_col}
    feat_cols = [c for c in feat_cols if c not in exclude_cols]
    if len(feat_cols) == 0:
        raise ValueError("After excluding ID/order/target/split, no numeric feature columns remain.")

    # 4) Train subset for stats
    tr_df = df[df["split"] == "train"] if "split" in df.columns else df

    # 5) Load scaler artifact (supports sklearn or dict)
    scaler, mu_art, std_art, seq_from_art, fcols_from_art = load_scaler_artifact(SCALER_PATH)

    # If artifact carries feature order, align to it
    if fcols_from_art:
        fcols_from_art = [c for c in fcols_from_art if c in df.columns]
        fcols_from_art = list(df[fcols_from_art].select_dtypes(include=[np.number]).columns)
        if fcols_from_art:
            feat_cols = fcols_from_art

    # If a plain sklearn scaler, respect its feature count
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        f_scaler = int(scaler.n_features_in_)
        if len(feat_cols) != f_scaler:
            if FEATS_TXT.exists():
                trained = [ln.strip() for ln in FEATS_TXT.read_text().splitlines() if ln.strip()]
                trained = [c for c in trained if c in df.columns]
                trained = list(df[trained].select_dtypes(include=[np.number]).columns)
                feat_cols = trained[:f_scaler] if trained else feat_cols[:f_scaler]
            else:
                feat_cols = feat_cols[:f_scaler]

    # 6) Build groups/windows (try grouped first, then GLOBAL fallback)
    groups = {}
    if args.serial_id is not None and id_col is not None:
        groups[str(args.serial_id)] = df[df[id_col] == args.serial_id]
    elif id_col is not None:
        for sid, g in df.groupby(id_col):
            groups[str(sid)] = g
    else:
        groups = {}

    try:
        if groups:
            Xseq, y = window_lastW(groups, feat_cols, args.window, order_col, id_col or "", TARGET)  # [N, W, F]
        else:
            raise ValueError("No groups available; forcing global fallback.")
    except ValueError as e:
        print(f"[INFO] Grouped windows failed: {e}")
        print("[INFO] Falling back to GLOBAL windows (time-sorted, no per-serial grouping).")
        Xseq, y = build_global_windows(df, feat_cols, args.window, order_col, TARGET)

    # 7) Per-timestep impute + scale
    if scaler is not None:
        Xseq = per_timestep_impute_scale(Xseq, scaler=scaler)
    else:
        if (mu_art is None) or (std_art is None):
            tr_vals = tr_df[feat_cols].values.astype("float32")
            mu_art  = np.nanmean(tr_vals, axis=0).astype("float32")
            std_art = np.nanstd (tr_vals, axis=0).astype("float32")
        Xseq = per_timestep_impute_scale(Xseq, scaler=None, mu=mu_art, std=std_art)

    # 8) Align with checkpoint input size and load
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location="cpu")
    w_ih = state["lstm.weight_ih_l0"]             # [4*hidden, input_size]
    hidden_ckpt = int(w_ih.shape[0] // 4)
    input_ckpt  = int(w_ih.shape[1])

    if len(feat_cols) != input_ckpt:
        print(f"[INFO] Aligning feat_cols to checkpoint input size {input_ckpt} (was {len(feat_cols)}).")
        if FEATS_TXT.exists():
            trained = [ln.strip() for ln in FEATS_TXT.read_text().splitlines() if ln.strip()]
            trained = [c for c in trained if c in df.columns]
            trained = list(df[trained].select_dtypes(include=[np.number]).columns)
            feat_cols = trained[:input_ckpt] if trained else feat_cols[:input_ckpt]
        else:
            feat_cols = feat_cols[:input_ckpt]
        try:
            if groups:
                Xseq, y = window_lastW(groups, feat_cols, args.window, order_col, id_col or "", TARGET)
            else:
                raise ValueError("No groups available; forcing global fallback after realign.")
        except ValueError:
            Xseq, y = build_global_windows(df, feat_cols, args.window, order_col, TARGET)

        # Re-scale after rebuild
        if scaler is not None:
            Xseq = per_timestep_impute_scale(Xseq, scaler=scaler)
        else:
            if (mu_art is None) or (std_art is None):
                tr_vals = tr_df[feat_cols].values.astype("float32")
                mu_art  = np.nanmean(tr_vals, axis=0).astype("float32")
                std_art = np.nanstd (tr_vals, axis=0).astype("float32")
            Xseq = per_timestep_impute_scale(Xseq, scaler=None, mu=mu_art, std=std_art)

    has_cls = any(k.startswith("head_cls.") for k in state.keys())
    model = LSTMRegMTL(in_dim=len(feat_cols), hidden=hidden_ckpt, has_cls=has_cls)
    state_pruned = {k: v for k, v in state.items() if not k.startswith("head_cls.")}
    _missing, _unexpected = model.load_state_dict(state_pruned, strict=False)
    model.eval()

    # 9) Dataset statistics (train subset)
    try:
        num_stats = tr_df[feat_cols + ([TARGET] if TARGET in tr_df.columns else [])].describe().T
        num_stats.to_csv(DATA_STATS_CSV)
    except Exception as e:
        print(f"[WARN] dataset_stats.csv not written: {e}", file=sys.stderr)

    # 10) SHAP (or permutation) on last-timestep features
    X_last = Xseq[:, -1, :]        # [N, F]
    topN = min(20, X_last.shape[1])

    if HAVE_SHAP:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.linear_model import Ridge

        ridge = Ridge(alpha=1.0).fit(X_last, y)

        bg_n = min(200, len(X_last))
        bg_idx = np.random.choice(len(X_last), size=bg_n, replace=False)
        background = X_last[bg_idx]

        explainer = shap.KernelExplainer(ridge.predict, background)
        samp_idx = np.random.choice(len(X_last), size=min(1000, len(X_last)), replace=False)
        shap_values = explainer.shap_values(X_last[samp_idx], nsamples="auto")

        # Summary bar
        plt.figure()
        shap.summary_plot(shap_values, X_last[samp_idx], feature_names=feat_cols, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(FIG_BAR, dpi=200, bbox_inches="tight")
        plt.close()

        # Beeswarm
        plt.figure()
        shap.summary_plot(shap_values, X_last[samp_idx], feature_names=feat_cols, show=False)
        plt.tight_layout()
        plt.savefig(FIG_BSWARM, dpi=200, bbox_inches="tight")
        plt.close()

        # Top features CSV
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order = np.argsort(-mean_abs)[:topN]
        pd.DataFrame({
            "feature": np.array(feat_cols)[order],
            "mean_abs_shap": mean_abs[order]
        }).to_csv(TOP_FEATS_CSV, index=False)

        # Dependence for top-2
        if len(order) >= 2:
            try:
                plt.figure()
                shap.dependence_plot(order[0], shap_values, X_last[samp_idx], feature_names=feat_cols, interaction_index=None, show=False)
                plt.tight_layout()
                plt.savefig(FIG_DEP1, dpi=200, bbox_inches="tight")
                plt.close()

                plt.figure()
                shap.dependence_plot(order[1], shap_values, X_last[samp_idx], feature_names=feat_cols, interaction_index=None, show=False)
                plt.tight_layout()
                plt.savefig(FIG_DEP2, dpi=200, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"[WARN] Dependence plots skipped: {e}", file=sys.stderr)

        print("[OK] SHAP artifacts written.")
    else:
        from sklearn.linear_model import Ridge
        from sklearn.inspection import permutation_importance
        ridge = Ridge(alpha=1.0).fit(X_last, y)
        perm = permutation_importance(ridge, X_last, y, n_repeats=10, random_state=42)
        idx = np.argsort(-perm.importances_mean)[:topN]
        pd.DataFrame({
            "feature": np.array(feat_cols)[idx],
            "importance": perm.importances_mean[idx]
        }).to_csv(TOP_FEATS_CSV, index=False)
        print("[OK] Fallback importance written (install `shap` for SHAP plots).")

    print(f"[DONE] Outputs in: {OUT_DIR.resolve()}")
    print(f" - {TOP_FEATS_CSV.name}, {DATA_STATS_CSV.name}")
    print(f" - {FIG_BAR.name}, {FIG_BSWARM.name}, {FIG_DEP1.name}, {FIG_DEP2.name}")


if __name__ == "__main__":
    main()
