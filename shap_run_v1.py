#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drop-in SHAP runner for Cycle Time LSTM project.

Key features:
- CPU-only (avoids CUDA init issues)
- Loads outputs/lstm_cycle_time.pt (MTL-aware; prunes head_cls.* if present)
- Uses outputs/lstm_features.txt when present to lock training feature order
- Filters to numeric-only features; removes timestamp/ID/split columns
- Aligns feature set to StandardScaler.n_features_in_ and to checkpoint input size
- Imputes NaNs with train means; scales per timestep
- Produces SHAP bar + beeswarm + dependence plots, top_features.csv, dataset_stats.csv
- Fallback to permutation importance if SHAP is not installed

Usage:
    python3 shap_run.py --window 10
"""

import os, sys, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Force CPU to avoid CUDA warnings
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
MODEL_PATH       = OUT_DIR / "lstm_cycle_time.pt"      # your checkpoint filename
SCALER_PATH      = OUT_DIR / "lstm_scaler.joblib"
FEATS_TXT        = OUT_DIR / "lstm_features.txt"

TOP_FEATS_CSV    = OUT_DIR / "top_features.csv"
DATA_STATS_CSV   = OUT_DIR / "dataset_stats.csv"

FIG_BAR          = OUT_DIR / "shap_summary_bar.png"
FIG_BSWARM       = OUT_DIR / "shap_beeswarm.png"
FIG_DEP1         = OUT_DIR / "shap_dependence_1.png"
FIG_DEP2         = OUT_DIR / "shap_dependence_2.png"

TARGET = "CycleTime_sec"
ID     = "serial_id"
ORDER  = "cycle_number"


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

def load_scaler_artifact(path):
    """
    Returns (scaler, mu, std):
      - If joblib file is an sklearn scaler -> (scaler, None, None)
      - If joblib file is a dict with {'mean','std','feature_cols','seq_len'} -> (None, mu, std)
    """
    obj = joblib.load(path)
    # sklearn scaler?
    if hasattr(obj, "transform") and hasattr(obj, "fit"):
        return obj, None, None
    # dict artifact?
    if isinstance(obj, dict):
        mu = np.asarray(obj.get("mean"), dtype=np.float32)
        std = np.asarray(obj.get("std"), dtype=np.float32)
        return None, mu, std
    raise TypeError(f"Unsupported scaler artifact type: {type(obj)}")


def get_initial_feat_cols(df: pd.DataFrame) -> list:
    """Prefer training feature list; otherwise numeric columns only."""
    # Remove known non-features
    exclude = {TARGET, ID, ORDER, "split", "cycle_timestamp", "timestamp", "time", "date", "datetime"}
    present = [c for c in df.columns if c not in exclude]

    # If we have a training feature list, use its order and intersect with present
    if FEATS_TXT.exists():
        trained = [ln.strip() for ln in FEATS_TXT.read_text().splitlines() if ln.strip()]
        trained = [c for c in trained if c in present]
        # Numeric only
        num_trained = list(df[trained].select_dtypes(include=[np.number]).columns)
        if len(num_trained) == 0:
            raise ValueError("lstm_features.txt exists but none of its columns are numeric/present.")
        return num_trained

    # Else: numeric only from remaining
    numeric = list(df[present].select_dtypes(include=[np.number]).columns)
    if len(numeric) == 0:
        raise ValueError("No numeric feature columns found after exclusions.")
    return numeric

def window_lastW(groups: dict, feat_cols: list, window: int) -> tuple:
    """Build [N,W,F] windows and next-step target y (float)."""
    Xs, ys = [], []
    for sid, g in groups.items():
        g = g.sort_values(ORDER) if ORDER in g.columns else g.sort_values(ID)
        if len(g) < window + 1:
            continue
        vals = g[feat_cols].values.astype("float32")
        tgt  = g[TARGET].values.astype("float32")
        for i in range(len(g) - window):
            Xs.append(vals[i:i+window])
            ys.append(tgt[i+window])
    if not Xs:
        raise ValueError("Not enough rows to form windows. Check window size or grouping.")
    return np.stack(Xs, axis=0), np.asarray(ys, dtype="float32")

def impute_fit_scaler(tr_df: pd.DataFrame, feat_cols: list) -> tuple[StandardScaler, np.ndarray]:
    """Impute train-split NaNs with column means and fit a StandardScaler."""
    tr_vals = tr_df[feat_cols].values.astype("float32")
    means = tr_df[feat_cols].mean(numeric_only=True).astype("float32").values
    nan_mask = np.isnan(tr_vals)
    if nan_mask.any():
        r, c = np.where(nan_mask)
        tr_vals[r, c] = means[c]
    scaler = StandardScaler().fit(tr_vals)
    return scaler, means

'''
def per_timestep_impute_scale(Xseq: np.ndarray, scaler: StandardScaler, means: np.ndarray) -> np.ndarray:
    """Impute NaNs and apply scaler transform for each timestep independently."""
    N, W, F = Xseq.shape
    for t in range(W):
        Xt = Xseq[:, t, :]
        nan_mask = np.isnan(Xt)
        if nan_mask.any():
            r, c = np.where(nan_mask)
            Xt[r, c] = means[c]
        Xseq[:, t, :] = scaler.transform(Xt)
    return Xseq
'''
def per_timestep_impute_scale(Xseq, scaler=None, mu=None, std=None):
    """
    Xseq: [N, T, F]
    If scaler is provided (sklearn), use scaler.transform per timestep.
    Else use mu/std arrays (broadcasted) for imputation + standardization.
    """
    N, T, F = Xseq.shape

    if scaler is not None:
        # sklearn path
        for t in range(T):
            Xt = Xseq[:, t, :]
            Xt[~np.isfinite(Xt)] = np.nan
            # impute with column means estimated on train
            col_mean = np.nanmean(Xt, axis=0)
            inds = np.where(~np.isfinite(Xt))
            if inds[0].size:
                Xt[inds] = np.take(col_mean, inds[1])
            Xseq[:, t, :] = scaler.transform(Xt)
        return Xseq.astype(np.float32)

    # dict (mu/std) path
    if mu is None or std is None:
        raise ValueError("per_timestep_impute_scale: need either sklearn scaler or (mu,std) arrays.")

    if mu.shape[-1] != F or std.shape[-1] != F:
        raise ValueError(f"Feature count mismatch: X has F={F}, mu/std have {mu.shape[-1]}/{std.shape[-1]}")

    std_safe = np.maximum(std, 1e-8)
    # broadcast mu/std to [N,T,F]
    MU = np.broadcast_to(mu, (N, T, F))
    STD = np.broadcast_to(std_safe, (N, T, F))

    X = Xseq.copy()
    X[~np.isfinite(X)] = np.nan
    # impute with train means
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = MU[nan_mask]
    # standardize
    X = (X - MU) / STD
    return X.astype(np.float32)


# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10, help="Sequence length (W)")
    ap.add_argument("--serial-id", default=None, help="Filter to a single serial_id group (optional)")
    args = ap.parse_args()

    # 1) Load features
    df = load_features_df(FEATURES_PARQUET).copy()
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])

    # Ensure IDs/ORDER sane
    if ID not in df.columns:
        df[ID] = 0
    if ORDER not in df.columns:
        df[ORDER] = df.groupby(ID).cumcount()

    # 2) Initial numeric feature list
    feat_cols = get_initial_feat_cols(df)
    F = len(feat_cols)

    # 3) Train slice for stats
    tr_df = df[df["split"] == "train"] if "split" in df.columns else df

    '''
    # 4) Load or fit scaler (train means)
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        # Ensure numeric-only and present columns
        feat_cols = [c for c in feat_cols if c in df.columns]
        # Align to scaler.n_features_in_
        if hasattr(scaler, "n_features_in_"):
            f_scaler = int(scaler.n_features_in_)
            if len(feat_cols) != f_scaler:
                print(f"[INFO] Adjusting feat_cols to scaler n_features_in_={f_scaler} (was {len(feat_cols)}).")
                if FEATS_TXT.exists():
                    trained = [ln.strip() for ln in FEATS_TXT.read_text().splitlines() if ln.strip()]
                    # keep numeric-only + present
                    num_trained = list(df[trained].select_dtypes(include=[np.number]).columns)
                    feat_cols = [c for c in trained if c in num_trained][:f_scaler]
                else:
                    feat_cols = feat_cols[:f_scaler]
        # Compute means aligned to current feat_cols
        means = tr_df[feat_cols].mean(numeric_only=True).astype("float32").values
    else:
        scaler, means = impute_fit_scaler(tr_df, feat_cols)
        joblib.dump(scaler, SCALER_PATH)

    # 5) Build groups/windows (before any model alignment)
    groups = {}
    if args.serial_id is not None:
        groups[args.serial_id] = df[df[ID] == args.serial_id]
    else:
        for sid, g in df.groupby(ID):
            groups[sid] = g

    Xseq, y = window_lastW(groups, feat_cols, args.window)   # [N,W,F]
    '''

    scaler, mu_from_art, std_from_art = load_scaler_artifact(SCALER_PATH)

    # If we have a dict artifact, align feature list to its feature_cols
    if scaler is None and mu_from_art is not None:
        if FEATS_TXT.exists():
            feat_cols = [ln.strip() for ln in FEATS_TXT.read_text().splitlines() if ln.strip()]
        # ensure feat_cols length matches mu/std
        F_art = int(mu_from_art.shape[-1])
        if len(feat_cols) != F_art:
            # hard align: keep only features present and truncate/pad if needed
            feat_cols = [c for c in feat_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)][:F_art]

    # build Xseq ... (your existing windowing code)

    # now scale with either sklearn scaler OR mu/std
    Xseq = per_timestep_impute_scale(
        Xseq,
        scaler=scaler,
        mu=mu_from_art,
        std=std_from_art
    )


    # 6) Align with checkpoint input size
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location="cpu")
    w_ih = state["lstm.weight_ih_l0"]             # [4*hidden, input_size]
    hidden_ckpt = int(w_ih.shape[0] // 4)
    input_ckpt  = int(w_ih.shape[1])

    if len(feat_cols) != input_ckpt:
        print(f"[INFO] Aligning feat_cols to checkpoint input size {input_ckpt} (was {len(feat_cols)}).")
        # Prefer training feature order if available
        if FEATS_TXT.exists():
            trained = [ln.strip() for ln in FEATS_TXT.read_text().splitlines() if ln.strip()]
            # numeric-only + present
            num_trained = list(df[trained].select_dtypes(include=[np.number]).columns)
            feat_cols = [c for c in trained if c in num_trained][:input_ckpt]
        else:
            # already numeric-only; truncate deterministically
            feat_cols = feat_cols[:input_ckpt]

        # Rebuild windows & means to match new feat_cols
        means = tr_df[feat_cols].mean(numeric_only=True).astype("float32").values
        Xseq, y = window_lastW(groups, feat_cols, args.window)

    # 7) Per-timestep impute + scale (now that feat_cols are final)
    Xseq = per_timestep_impute_scale(Xseq, scaler, means)

    # 8) Rebuild model with checkpoint hidden size; prune classification head
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

    # 10) SHAP (or fallback) on last timestep features
    X_last = Xseq[:, -1, :]        # [N,F]
    topN = min(20, X_last.shape[1])

    if HAVE_SHAP:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.linear_model import Ridge

        # Surrogate for KernelExplainer (fast, stable)
        ridge = Ridge(alpha=1.0).fit(X_last, y)

        # Background sample
        bg_idx = np.random.choice(len(X_last), size=min(200, len(X_last)), replace=False)
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

        # Dependence plots for top-2
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
        # Fallback to permutation importance
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
