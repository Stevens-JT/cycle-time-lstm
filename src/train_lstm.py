#!/usr/bin/env python3
import argparse
import datetime as dt
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SeqDataset(Dataset):
    def __init__(self, X3, y_sec, y_long):
        self.X3 = X3.astype(np.float32)
        self.y_sec = y_sec.astype(np.float32)  # raw seconds (or log)
        self.y_long = y_long.astype(np.float32)  # 0/1
    def __len__(self):
        return self.X3.shape[0]
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X3[idx]),
            torch.from_numpy(self.y_sec[idx:idx+1]),
            torch.from_numpy(self.y_long[idx:idx+1]),
        )

class LSTMTwoHead(nn.Module):
    def __init__(self, n_feat, hidden=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=0.0 if num_layers==1 else dropout)
        self.head_reg = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.head_cls = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # last timestep
        logit_long = self.head_cls(h)
        pred_sec   = self.head_reg(h)
        return logit_long, pred_sec

def huber_loss(pred, target, beta=1.0, reduction="none"):
    # pred, target: [B, 1]
    err = pred - target
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, torch.tensor(beta, device=pred.device))
    lin  = abs_err - quad
    loss = 0.5 * quad**2 / beta + lin
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def bce_with_logits_focal(logits, targets, gamma=0.0, pos_weight=None):
    # logits/targets: [B,1]
    # standard BCE with logits
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    if gamma <= 0.0:
        return bce.mean()
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    fl = (1 - pt) ** gamma * bce
    return fl.mean()

def build_sequences(df, seq_len, feature_cols):
    # df is already split-filtered, sorted by C
    X, y = [], []
    for i in range(len(df) - seq_len + 1):
        window = df.iloc[i:i+seq_len]
        target_row = df.iloc[i+seq_len-1]
        X.append(window[feature_cols].values)
        y.append(target_row["CycleTime_sec"])
    X = np.array(X, dtype=np.float32)  # [N, T, F]
    y = np.array(y, dtype=np.float32)  # [N]
    return X, y


'''
def build_sequences_grouped(df, seq_len, feature_cols, id_col="A"):
    X, y = []
    for sid, g in df.sort_values(["C"]).groupby(id_col):
        g = g.sort_values("C")
        for i in range(len(g) - seq_len + 1):
            win = g.iloc[i:i+seq_len]
            tgt = g.iloc[i+seq_len-1]["CycleTime_sec"]
            X.append(win[feature_cols].values)
            y.append(tgt)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
'''

def standardize(train_arr, *others):
    mean = train_arr.mean(axis=(0,1), keepdims=True)
    std  = train_arr.std(axis=(0,1), keepdims=True) + 1e-8
    out = [(train_arr - mean) / std]
    for arr in others:
        out.append((arr - mean) / std)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--loss", type=str, default="huber", choices=["huber","l1","l2"])
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument("--log_target", action="store_true")
    p.add_argument("--long_thresh", type=float, default=90.0)
    p.add_argument("--alpha_cls", type=float, default=0.05, help="final weight for classifier after warm-up")
    p.add_argument("--alpha_reg", type=float, default=1.0)
    p.add_argument("--reg_w_long", type=float, default=3.0)
    p.add_argument("--reg_w_short", type=float, default=0.2)
    # New: warm-up & sampler
    p.add_argument("--warmup_epochs", type=int, default=5, help="epochs to ramp classifier from 0 -> alpha_cls")
    p.add_argument("--use_weighted_sampler", action="store_true")
    p.add_argument("--oversample_long", type=float, default=20.0, help="weight multiplier for long samples in sampler")
    p.add_argument("--focal_gamma", type=float, default=0.0, help=">0 to enable focal weighting for cls head")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_path = "outputs/features_spark.parquet"
    df = pd.read_parquet(features_path)
    print("Columns:", df.columns.tolist())
    print("Rows total:", len(df))

    # Make sure required columns exist
    req = {"A","B","C","split","CycleTime_sec"}
    assert req.issubset(df.columns), f"Missing required columns: {req - set(df.columns)}"

    df["C"] = pd.to_datetime(df["C"])
    df = df.sort_values(["C","A","B"]).reset_index(drop=True)

    print("Has 'split' + target:", {("split" in df.columns) and ("CycleTime_sec" in df.columns)})
    print(df["split"].value_counts())

    # Keep only rows with target present
    df = df.dropna(subset=["CycleTime_sec"])

    # Identify feature columns (exclude id/time/split/target)
    exclude = {"A","B","C","D","split","next_ts","CycleTime_sec"}
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    n_feat = len(feature_cols)
    print(f"# Feature columns: {n_feat}")

    feats_txt = Path("outputs/lstm_features.txt")
    if feats_txt.exists():
        selected = [ln.strip() for ln in feats_txt.read_text().splitlines() if ln.strip()]
        missing = [c for c in selected if c not in df.columns]
    if missing:
        print(f"[WARN] Selected features missing in dataframe: {missing}")
    feature_cols = [c for c in selected if c in df.columns]
    n_feat = len(feature_cols)
    print(f"[FS] Using {n_feat} selected features from outputs/lstm_features.txt")

    # Build sequences per split
    seq_len = args.seq_len
    train_df = df[df["split"]=="train"].copy().sort_values("C")
    val_df   = df[df["split"]=="val"].copy().sort_values("C")
    test_df  = df[df["split"]=="test"].copy().sort_values("C")

    Xtr, ytr = build_sequences(train_df, seq_len, feature_cols)
    Xva, yva = build_sequences(val_df,   seq_len, feature_cols)
    Xte, yte = build_sequences(test_df,  seq_len, feature_cols)
    #Xtr, ytr = build_sequences_grouped(train_df, seq_len, feature_cols)
    #Xva, yva = build_sequences_grouped(val_df,   seq_len, feature_cols)
    #Xte, yte = build_sequences_grouped(test_df,  seq_len, feature_cols)

    # Save raw y (seconds) for export metrics
    ytr_raw = ytr.copy()
    yva_raw = yva.copy()
    yte_raw = yte.copy()

    # Long/short labels (based on raw seconds)
    thr = args.long_thresh
    ytr_long = (ytr_raw >= thr).astype(np.float32)
    yva_long = (yva_raw >= thr).astype(np.float32)
    yte_long = (yte_raw >= thr).astype(np.float32)
    print(f"Train long-rate (>={thr}s): {ytr_long.mean():.4f}")

    # Standardize inputs based on TRAIN only
    #Xtr_s, Xva_s, Xte_s = standardize(Xtr, Xva, Xte)

    # --- Compute train-only mean/std and apply to all splits; also SAVE scaler + features ---
    '''
    mu = Xtr.mean(axis=(0, 1), keepdims=True)                           # shape [1,1,F]
    sigma = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8                  # shape [1,1,F]

    def apply_standardize(arr, mu, sigma):
        return (arr - mu) / sigma

    Xtr_s = apply_standardize(Xtr, mu, sigma)
    Xva_s = apply_standardize(Xva, mu, sigma)
    Xte_s = apply_standardize(Xte, mu, sigma)

    # Persist the scaler + feature list for inference/SHAP reproducibility
    Path("outputs").mkdir(parents=True, exist_ok=True)
    scaler_art = {
        "mean": mu.squeeze().astype(np.float32),   # shape [F]
        "std":  sigma.squeeze().astype(np.float32),# shape [F]
        "feature_cols": feature_cols,
        "seq_len": seq_len,
    }
    joblib.dump(scaler_art, "outputs/lstm_scaler.joblib")
    Path("outputs/lstm_features.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")
    print("[OK] Saved scaler -> outputs/lstm_scaler.joblib and feature list -> outputs/lstm_features.txt")
    '''

    # --- Impute NaNs using TRAIN means (per feature), then standardize ---
    # Compute per-feature means over train (across all timesteps/samples), ignoring NaNs
    train_means = np.nanmean(Xtr, axis=(0, 1), keepdims=True)  # shape [1,1,F]
    # If a feature is entirely NaN in train, fall back to 0
    train_means = np.where(np.isfinite(train_means), train_means, 0.0).astype(np.float32)

    def impute_with(train_means, arr):
        out = arr.copy()
        # Replace inf with nan so we can impute them too
        out[~np.isfinite(out)] = np.nan
        mask = np.isnan(out)
        if mask.any():
            # broadcast train_means to arr shape and plug values where mask
            out[mask] = np.broadcast_to(train_means, out.shape)[mask]
        return out

    Xtr = impute_with(train_means, Xtr)
    Xva = impute_with(train_means, Xva)
    Xte = impute_with(train_means, Xte)

    # Compute train-only mean/std AFTER imputation
    mu    = Xtr.mean(axis=(0, 1), keepdims=True)
    sigma = Xtr.std(axis=(0, 1), keepdims=True)
    sigma = np.where(sigma < 1e-8, 1e-8, sigma).astype(np.float32)

    def apply_standardize(arr, mu, sigma):
        return (arr - mu) / sigma

    Xtr_s = apply_standardize(Xtr, mu, sigma)
    Xva_s = apply_standardize(Xva, mu, sigma)
    Xte_s = apply_standardize(Xte, mu, sigma)

    # Persist the scaler + feature list for inference/SHAP
    Path("outputs").mkdir(parents=True, exist_ok=True)
    scaler_art = {
        "mean":  mu.squeeze().astype(np.float32),      # [F]
        "std":   sigma.squeeze().astype(np.float32),   # [F]
        "train_means": train_means.squeeze().astype(np.float32),  # [F], for completeness
        "feature_cols": feature_cols,
        "seq_len": seq_len,
    }
    joblib.dump(scaler_art, "outputs/lstm_scaler.joblib")
    Path("outputs/lstm_features.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")
    print("[OK] Saved scaler -> outputs/lstm_scaler.joblib and feature list -> outputs/lstm_features.txt")

    # Sanity checks
    assert np.isfinite(Xtr_s).all(), "Xtr_s contains non-finite values"
    assert np.isfinite(Xva_s).all(), "Xva_s contains non-finite values"
    assert np.isfinite(Xte_s).all(), "Xte_s contains non-finite values"

    # Optionally transform targets (log1p)
    if args.log_target:
        ytr_t = np.log1p(ytr_raw)
        yva_t = np.log1p(yva_raw)
        yte_t = np.log1p(yte_raw)
        def inv(x): return np.expm1(x)
    else:
        ytr_t, yva_t, yte_t = ytr_raw, yva_raw, yte_raw
        def inv(x): return x

    # Torch datasets
    tr_ds = SeqDataset(Xtr_s, ytr_t, ytr_long)
    va_ds = SeqDataset(Xva_s, yva_t, yva_long)
    te_ds = SeqDataset(Xte_s, yte_t, yte_long)

    # Weighted sampler to oversample LONG examples
    if args.use_weighted_sampler and (ytr_long.sum() > 0):
        w = np.where(ytr_long > 0.5, args.oversample_long, 1.0).astype(np.float32)
        sampler = WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)
        train_loader = DataLoader(tr_ds, batch_size=64, sampler=sampler, drop_last=False)
        print(f"Using WeightedRandomSampler (oversample_long={args.oversample_long})")
    else:
        train_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(va_ds, batch_size=256, shuffle=False, drop_last=False)
    test_loader= DataLoader(te_ds, batch_size=256, shuffle=False, drop_last=False)

    # Model & opt
    model = LSTMTwoHead(n_feat=n_feat, hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[MODEL] Input features (F) = {n_feat}, seq_len = {seq_len}, hidden = {args.hidden}")

    # Regression loss
    if args.loss == "huber":
        def reg_loss(pred, target):
            return huber_loss(pred, target, beta=args.huber_beta, reduction="none").squeeze(1)
    elif args.loss == "l1":
        def reg_loss(pred, target):
            return torch.abs(pred - target).squeeze(1)
    else: # l2
        def reg_loss(pred, target):
            return ((pred - target) ** 2).squeeze(1)

    # Class imbalance handling (pos_weight)
    pos_weight = None
    p_long = max(1e-6, float(ytr_long.mean()))
    if 0 < p_long < 1:
        pos_weight = torch.tensor([(1.0 - p_long) / p_long], dtype=torch.float32, device=device)

    best_val_rmse = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        # classifier warm-up schedule: ramp 0 -> alpha_cls over warmup_epochs
        if epoch <= args.warmup_epochs:
            alpha_cls_now = args.alpha_cls * (epoch / max(1, args.warmup_epochs))
        else:
            alpha_cls_now = args.alpha_cls

        epoch_loss = 0.0
        n_seen = 0

        for xb, yb_sec, yb_long in train_loader:
            xb = xb.to(device)
            yb_sec = yb_sec.to(device)
            yb_long= yb_long.to(device)

            optim.zero_grad()
            logit_long, pred_sec = model(xb)

            # classification loss (with optional focal)
            loss_cls = bce_with_logits_focal(logit_long, yb_long, gamma=args.focal_gamma, pos_weight=pos_weight)

            # regression loss with different weights for long/short examples
            per_sample_reg = reg_loss(pred_sec, yb_sec)  # [B]
            w_reg = torch.where(yb_long > 0.5,
                                torch.tensor(args.reg_w_long, device=device),
                                torch.tensor(args.reg_w_short, device=device))
            loss_reg = (per_sample_reg * w_reg).mean()

            loss = alpha_cls_now * loss_cls + args.alpha_reg * loss_reg
            loss.backward()
            optim.step()

            bs = xb.size(0)
            epoch_loss += loss.item() * bs
            n_seen += bs

        # ---- Eval on validation (in raw space) ----
        model.eval()
        with torch.no_grad():
            # Full-batch eval tensors
            Xva_t = torch.from_numpy(Xva_s).to(device)
            Xte_t = torch.from_numpy(Xte_s).to(device)
            logit_val, pred_val = model(Xva_t)
            logit_te,  pred_te  = model(Xte_t)

            pv_sec = inv(pred_val.squeeze(1).cpu().numpy())
            pt_sec = inv(pred_te.squeeze(1).cpu().numpy())
            yv_exp = yva_raw
            yt_exp = yte_raw

        v_mae = mean_absolute_error(yv_exp, pv_sec)
        v_rmse= math.sqrt(mean_squared_error(yv_exp, pv_sec))
        v_r2  = r2_score(yv_exp, pv_sec)

        t_mae = mean_absolute_error(yt_exp, pt_sec)
        t_rmse= math.sqrt(mean_squared_error(yt_exp, pt_sec))
        t_r2  = r2_score(yt_exp, pt_sec)

        print(f"Epoch {epoch:03d} | train loss {epoch_loss/max(1,n_seen):.4f} | val MAE {v_mae:.4f} RMSE {v_rmse:.4f} R2 {v_r2:.4f}")

        improved = v_rmse + 1e-9 < best_val_rmse
        if improved:
            best_val_rmse = v_rmse
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val RMSE={best_val_rmse:.4f})")
                break

    # load best state
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    # ---- Final Eval + Export ----
    model.eval()
    with torch.no_grad():
        Xva_t = torch.from_numpy(Xva_s).to(device)
        Xte_t = torch.from_numpy(Xte_s).to(device)

        logit_val, pred_val = model(Xva_t)
        logit_te,  pred_te  = model(Xte_t)

        pv_sec = inv(pred_val.squeeze(1).cpu().numpy())
        pt_sec = inv(pred_te.squeeze(1).cpu().numpy())
        yv_exp = yva_raw
        yt_exp = yte_raw

        pv_prob = torch.sigmoid(logit_val).squeeze(1).cpu().numpy()
        pt_prob = torch.sigmoid(logit_te).squeeze(1).cpu().numpy()

    # Metrics tables
    val_tbl = pd.DataFrame({"MAE":[mean_absolute_error(yv_exp, pv_sec)],
                            "RMSE":[math.sqrt(mean_squared_error(yv_exp, pv_sec))],
                            "R2":[r2_score(yv_exp, pv_sec)]})
    test_tbl= pd.DataFrame({"MAE":[mean_absolute_error(yt_exp, pt_sec)],
                            "RMSE":[math.sqrt(mean_squared_error(yt_exp, pt_sec))],
                            "R2":[r2_score(yt_exp, pt_sec)]})
    print("\nValidation:\n", val_tbl)
    print("\nTest:\n", test_tbl)

    # Build row-wise export with identifiers
    def last_rows(df_split):
        # return the last row (target row) for each rolling window
        # build matching index to (A,B,C,split) of the target rows
        rows = []
        for i in range(len(df_split) - seq_len + 1):
            rows.append(df_split.iloc[i+seq_len-1][["A","B","C","split"]])
        return pd.DataFrame(rows)

    val_rows  = last_rows(val_df)
    test_rows = last_rows(test_df)

    val_out = val_rows.copy()
    val_out["true_sec"] = yv_exp
    val_out["pred_sec"] = pv_sec
    val_out["pred_long_prob"] = pv_prob

    test_out = test_rows.copy()
    test_out["true_sec"] = yt_exp
    test_out["pred_sec"] = pt_sec
    test_out["pred_long_prob"] = pt_prob

    export = pd.concat([val_out, test_out], ignore_index=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    export.to_csv("outputs/lstm_predictions.csv", index=False)
    print("Wrote per-row predictions -> outputs/lstm_predictions.csv")

    # Save model + small report
    torch.save(model.state_dict(), "outputs/lstm_cycle_time.pt")
    with open("outputs/metrics_report_lstm.md","w") as f:
        f.write("# LSTM Two-Head Report\n")
        f.write(f"- Timestamp: {dt.datetime.utcnow().isoformat()}Z\n")
        f.write(f"- Val: MAE={val_tbl.MAE.iloc[0]:.4f}, RMSE={val_tbl.RMSE.iloc[0]:.4f}, R2={val_tbl.R2.iloc[0]:.4f}\n")
        f.write(f"- Test: MAE={test_tbl.MAE.iloc[0]:.4f}, RMSE={test_tbl.RMSE.iloc[0]:.4f}, R2={test_tbl.R2.iloc[0]:.4f}\n")

    print("\nSaved model -> outputs/lstm_cycle_time.pt")
    print("Wrote report -> outputs/metrics_report_lstm.md")


if __name__ == "__main__":
    main()
