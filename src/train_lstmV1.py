#!/usr/bin/env python3
import os
import math
import argparse
import datetime as dt
from typing import List, Tuple, Dict

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

PARQUET_PATH = "outputs/features_spark.parquet"
MODEL_OUT = "outputs/lstm_cycle_time.pt"
REPORT_OUT = "outputs/metrics_report_lstm.md"
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_split_counts(df):
    print("Rows total:", len(df))
    print("Has 'split' + target:", {"split" in df.columns and "CycleTime_sec" in df.columns})
    print(df["split"].value_counts(dropna=False))
    print("\nTarget non-null per split:")
    print(df[df["CycleTime_sec"].notna()].groupby("split")["CycleTime_sec"].count())

def pick_feature_columns(cols: List[str]) -> List[str]:
    # Keep engineered series stats, drop non-features
    drop = {"A","B","C","D","split","next_ts","CycleTime_sec"}
    feats = [c for c in cols if c not in drop]
    # Safety: keep only numeric-looking engineered stats
    # (your Spark output uses E_len_mean, E_max_mean, etc.)
    return feats

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_sec: np.ndarray, y_long: np.ndarray):
        # X_seq: [N, L, F]
        # y_sec: [N]   (seconds, possibly log1p-transformed)
        # y_long: [N]  (0/1 flag for "long cycle")
        self.X = torch.from_numpy(X_seq).float()
        self.y_sec = torch.from_numpy(y_sec).float()
        self.y_long = torch.from_numpy(y_long).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_sec[idx], self.y_long[idx]

'''
class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        # X_seq: [N, L, F]
        # y: [N]
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: [B, L, F]
        out, _ = self.lstm(x)      # out: [B, L, H]
        last = out[:, -1, :]       # take last step
        return self.head(last).squeeze(-1)
'''

class LSTMTwoHead(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden,
                            num_layers=layers, batch_first=True, dropout=dropout if layers>1 else 0.0)
        self.head_cls = nn.Linear(hidden, 1)   # logits for is_long
        self.head_reg = nn.Linear(hidden, 1)   # seconds
    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        h = out[:, -1, :]          # last hidden state
        logit_long = self.head_cls(h).squeeze(1)  # [B]
        pred_sec   = self.head_reg(h).squeeze(1)  # [B]
        return logit_long, pred_sec

'''
def build_sequences(
    df: pd.DataFrame,
    feat_cols: List[str],
    seq_len: int,
    include_prev_ct: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows across the *time-ordered* dataframe.
    Assumes single machine. If multiple machines later, groupby id and do per-group windows.
    """
    df = df.copy()
    # Add previous cycle time as a feature if requested
    if include_prev_ct:
        df["prev_ct"] = df["CycleTime_sec"].shift(1)
        # We won't impute across big gaps; rows without prev_ct will be dropped by windowing anyway
        feat_cols_ext = feat_cols + ["prev_ct"]
    else:
        feat_cols_ext = feat_cols

    # Only rows with target present are candidates for windows
    # But window needs L rows of *features* ending at t-1; target at t
    # We'll construct indexes so that for each t we take [t-L .. t-1] as input and t as target
    valid = df[feat_cols_ext + ["CycleTime_sec"]].dropna().index
    # We’ll do a straightforward global sliding window on the filtered df
    dff = df.loc[valid].reset_index(drop=True)

    X_list = []
    y_list = []
    for t in range(seq_len, len(dff)):
        # inputs use rows [t-seq_len .. t-1]
        window = dff.iloc[t - seq_len : t]
        # target is at time t (same contiguous block assumption)
        target_row = dff.iloc[t]
        X_list.append(window[feat_cols_ext].to_numpy(dtype=np.float32))
        y_list.append(float(target_row["CycleTime_sec"]))

    if not X_list:
        return np.empty((0, seq_len, len(feat_cols_ext)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y
'''

def build_sequences(
    df: pd.DataFrame,
    feat_cols: List[str],
    seq_len: int,
    include_prev_ct: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build sliding windows across the *time-ordered* dataframe.
    Inputs are rows [t-seq_len .. t-1]; target is at time t.
    Returns (X, y, meta) where meta holds A/B/C for the target row (t).
    """
    df = df.copy()

    # Optional previous cycle time as a feature
    if include_prev_ct:
        df["prev_ct"] = df["CycleTime_sec"].shift(1)
        feat_cols_ext = feat_cols + ["prev_ct"]
    else:
        feat_cols_ext = feat_cols

    # keep only rows where all required features AND target exist
    valid_idx = df[feat_cols_ext + ["CycleTime_sec"]].dropna().index
    dff = df.loc[valid_idx].reset_index(drop=True)

    X_list, y_list = [], []
    meta_A, meta_B, meta_C = [], [], []

    # inputs: [t-seq_len .. t-1], target/meta: row t
    for t in range(seq_len, len(dff)):
        window = dff.iloc[t - seq_len : t]
        target_row = dff.iloc[t]

        X_list.append(window[feat_cols_ext].to_numpy(dtype=np.float32))
        y_list.append(float(target_row["CycleTime_sec"]))

        # metadata aligned with the target row
        meta_A.append(target_row["A"])
        meta_B.append(target_row["B"])
        meta_C.append(target_row["C"])

    if not X_list:
        X = np.empty((0, seq_len, len(feat_cols_ext)), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)
        meta = {"A": np.array([]), "B": np.array([]), "C": np.array([])}
        return X, y, meta

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    meta = {"A": np.array(meta_A), "B": np.array(meta_B), "C": np.array(meta_C)}
    return X, y, meta

def evaluate_split(model, loader, device):
    model.eval()
    yp = []
    yt = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            yp.append(pred)
            yt.append(yb.numpy())
    y_pred = np.concatenate(yp)
    y_true = np.concatenate(yt)
    return (
        mean_absolute_error(y_true, y_pred),
        rmse(y_true, y_pred),
        r2_score(y_true, y_pred),
        y_true,
        y_pred,
    )

def evaluate_split_twohead(model, loader, device, log_target: bool):
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for xb, yb_sec, _ in loader:
            xb = xb.to(device)
            _, pred_sec = model(xb)
            yp.append(pred_sec.cpu().numpy())
            yt.append(yb_sec.cpu().numpy())
    y_pred = np.concatenate(yp)
    y_true = np.concatenate(yt)

    # back-transform if trained on log1p
    if log_target:
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)

    return (
        mean_absolute_error(y_true, y_pred),
        rmse(y_true, y_pred),
        r2_score(y_true, y_pred),
        y_true,
        y_pred,
    )


def main():
    set_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--loss", choices=["mae","huber","mse"], default="mae",
               help="Training loss: mae (L1), huber (SmoothL1), or mse (L2).")
    ap.add_argument("--huber_beta", type=float, default=1.0,
               help="Beta parameter for SmoothL1Loss (transition point).")
    ap.add_argument("--log_target", action="store_true",
               help="Train on log1p(target); metrics reported in seconds (back-transformed).")
    ap.add_argument("--long_thresh", type=float, default=90.0,
               help="Seconds >= long_thresh are 'long' cycles.")
    ap.add_argument("--alpha_cls", type=float, default=0.3,
               help="Weight for long/short classification loss.")
    ap.add_argument("--alpha_reg", type=float, default=1.0,
               help="Weight for regression (seconds) loss.")
    ap.add_argument("--reg_w_long", type=float, default=1.0,
               help="Per-sample weight for regression on long cycles.")
    ap.add_argument("--reg_w_short", type=float, default=0.2,
               help="Per-sample weight for regression on short cycles.")

    args = ap.parse_args()

    # after parsing args
    if args.loss == "mae":
        loss_fn = torch.nn.L1Loss(reduction="none")
    elif args.loss == "huber":
        reg_loss_fn = nn.SmoothL1Loss(beta=args.huber_beta, reduction="none")
    elif args.loss == "l1":
        reg_loss_fn = nn.L1Loss(reduction="none")
    else:
        reg_loss_fn = nn.MSELoss(reduction="none")


    # ----------------- Load & basic checks -----------------
    df = pd.read_parquet(PARQUET_PATH)
    # Keep only rows with a target; we’ll still need earlier rows for prev_ct via shift,
    # but the sliding-window builder takes care of that internally.
    print("Columns:", list(df.columns))
    print_split_counts(df)

    # Sort globally by timestamp C (already parsed to timestamp in Spark and written back as string)
    # If C has type object/string, sort by it; if timestamp-like, pandas will handle it.
    df = df.sort_values("C").reset_index(drop=True)

    # Feature columns
    feat_cols = pick_feature_columns(list(df.columns))
    print(f"\n# Feature columns: {len(feat_cols)}")

    # ----------------- Split frames -----------------
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    te = df[df["split"] == "test"].copy()

    # Standardize features on *train only*
    scaler = StandardScaler()
    # Fit scaler on *row-wise* features (not sequences yet)
    tr_feat = tr[feat_cols].copy()
    tr_feat = tr_feat.fillna(tr_feat.mean(numeric_only=True))
    scaler.fit(tr_feat.to_numpy(dtype=np.float32))

    def transform_feats(d: pd.DataFrame) -> pd.DataFrame:
        dd = d.copy()
        # simple mean impute for missing engineered stats
        dd[feat_cols] = dd[feat_cols].fillna(tr_feat.mean(numeric_only=True))
        dd[feat_cols] = scaler.transform(dd[feat_cols].to_numpy(dtype=np.float32))
        return dd

    tr = transform_feats(tr)
    va = transform_feats(va)
    te = transform_feats(te)

    # ----------------- Build sequences -----------------
    Xtr, ytr, meta_tr = build_sequences(tr, feat_cols, seq_len=args.seq_len, include_prev_ct=True)
    Xva, yva, meta_va = build_sequences(va, feat_cols, seq_len=args.seq_len, include_prev_ct=True)
    Xte, yte, meta_te = build_sequences(te, feat_cols, seq_len=args.seq_len, include_prev_ct=True)

    Xva_t = torch.from_numpy(Xva).float()
    Xte_t = torch.from_numpy(Xte).float()

    ytr_raw, yva_raw, yte_raw = ytr.copy(), yva.copy(), yte.copy()

    ytr_long = (ytr_raw >= args.long_thresh).astype(np.float32)
    yva_long = (yva_raw >= args.long_thresh).astype(np.float32)
    yte_long = (yte_raw >= args.long_thresh).astype(np.float32)

    print(f"Train long-rate (>= {args.long_thresh}s): {ytr_long.mean():.4f}")

    if args.log_target:
        ytr = np.log1p(ytr)
        yva = np.log1p(yva)
        yte = np.log1p(yte)


    print(f"\nShapes -> Xtr:{Xtr.shape}, ytr:{ytr.shape}; Xva:{Xva.shape}, yva:{yva.shape}; Xte:{Xte.shape}, yte:{yte.shape}")
    if Xtr.shape[0] == 0 or Xva.shape[0] == 0 or Xte.shape[0] == 0:
        raise RuntimeError("Not enough sequential data to train/evaluate. Try lowering seq_len.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SeqDataset(Xtr, ytr, ytr_long)
    val_ds   = SeqDataset(Xva, yva, yva_long)
    test_ds  = SeqDataset(Xte, yte, yte_long)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ----------------- Model & training -----------------
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = LSTMRegressor(in_dim=Xtr.shape[-1], hidden=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    model = LSTMTwoHead(in_dim=Xtr.shape[-1],
                    hidden=args.hidden,
                    layers=args.layers,
                    dropout=args.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    #loss_fn = nn.L1Loss()  # MAE tends to be robust for cycle times

    # Regression loss (seconds)
    if args.loss == "mae":
        reg_loss_fn = nn.L1Loss(reduction="none")
    elif args.loss == "mse":
        reg_loss_fn = nn.MSELoss(reduction="none")
    else:
        reg_loss_fn = nn.SmoothL1Loss(beta=args.huber_beta, reduction="none")

    # Classification loss (long vs short) — use logits
    # Use pos_weight to counter class imbalance if needed:
    pos_weight = None
    p_long = max(1e-6, float(ytr_long.mean()))
    if 0 < p_long < 1:
        # crude inverse prior: more weight to minority class
        pos_weight = torch.tensor([(1.0 - p_long) / p_long], dtype=torch.float32, device=device)
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for xb, yb_sec, yb_long in train_loader:
            xb = xb.to(device)
            yb_sec = yb_sec.to(device)
            yb_long = yb_long.to(device)

            optim.zero_grad()
            logit_long, pred_sec = model(xb)
            loss_cls = cls_loss_fn(logit_long, yb_long)
            loss_reg_each = reg_loss_fn(pred_sec, yb_sec)  # [B]
            reg_w = torch.where(yb_long > 0.5,
                                torch.tensor(args.reg_w_long, device=device),
                                torch.tensor(args.reg_w_short, device=device))
            loss_reg = (loss_reg_each * reg_w).mean()

            # total loss
            loss = args.alpha_cls * loss_cls + args.alpha_reg * loss_reg
            loss.backward()
            optim.step()

            #epoch_loss += loss.item() * xb.size(0)
            bs = xb.size(0)
            epoch_loss += loss.item() * bs
            n_samples += bs

        #epoch_loss /= len(train_ds)
        epoch_loss /= max(1, n_samples)

        # quick val MAE for early stop
        model.eval()
        with torch.no_grad():
            #v_mae, v_rmse, v_r2, _, _ = evaluate_split(model, val_loader, device)
            v_mae, v_rmse, v_r2, _, _ = evaluate_split_twohead(model, val_loader, device, args.log_target)

        print(f"Epoch {epoch:03d} | train loss {epoch_loss:.4f} | val MAE {v_mae:.4f} RMSE {v_rmse:.4f} R2 {v_r2:.4f}")

        if v_rmse < best_val - 1e-6:
            best_val = v_rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val RMSE={best_val:.4f})")
                break

    # ----------------- Load best & final eval -----------------
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    #v_mae, v_rmse, v_r2, yv, pv = evaluate_split(model, val_loader, device)
    #t_mae, t_rmse, t_r2, yt, pt = evaluate_split(model, test_loader, device)
    v_mae, v_rmse, v_r2, yv, pv = evaluate_split_twohead(model, val_loader, device, args.log_target)
    t_mae, t_rmse, t_r2, yt, pt = evaluate_split_twohead(model, test_loader, device, args.log_target)

    print("\nValidation:")
    print(pd.DataFrame([{"MAE": v_mae, "RMSE": v_rmse, "R2": v_r2}]))
    print("\nTest:")
    print(pd.DataFrame([{"MAE": t_mae, "RMSE": t_rmse, "R2": t_r2}]))

    thr = float(args.long_thresh)

    model.eval()
    with torch.no_grad():
        # stack full tensors (you already created Xva_t/Xte_t earlier)
        logit_val, pred_val = model(Xva_t.to(device))
        logit_te,  pred_te  = model(Xte_t.to(device))

        # regression outputs to numpy
        pv_sec = pred_val.squeeze(1).cpu().numpy()
        pt_sec = pred_te.squeeze(1).cpu().numpy()

        # classifier probs if the head exists (shape [N,1])
        pv_prob = torch.sigmoid(logit_val).squeeze(1).cpu().numpy()
        pt_prob = torch.sigmoid(logit_te).squeeze(1).cpu().numpy()

    if args.log_target:
        #yv_eval = np.expm1(yv)
        #yv_exp = yv_eval
        yv_exp = yva_raw
        #pv_eval = np.expm1(pv)
        #pv_exp = pv_eval
        pv_sec = np.expm1(pv_sec)
        #yt_eval = np.expm1(yt)
        #yt_exp = yt_eval
        yt_exp = yte_raw
        #pt_eval = np.expm1(pt)
        #pt_exp = pt_eval
        pt_sec = np.expm1(pt_sec)

        # recompute metrics in raw target space
        #from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        #v_mae = mean_absolute_error(yv_eval, pv_eval)
        #v_rmse = np.sqrt(mean_squared_error(yv_eval, pv_eval))
        #v_r2 = r2_score(yv_eval, pv_eval)

        #t_mae = mean_absolute_error(yt_eval, pt_eval)
        #t_rmse = np.sqrt(mean_squared_error(yt_eval, pt_eval))
        #t_r2 = r2_score(yt_eval, pt_eval)

    else:
        #yv_exp, pv_exp = yv, pv
        #yv_exp = yva
        yv_exp = yv
        yt_exp = yt
        #yt_exp, pt_exp = yt, pt
        #yt_exp = yte
        pv_exp = pv_sec
        pt_exp = pt_sec

    v_mae = mean_absolute_error(yv_exp, pv_exp)
    v_rmse = np.sqrt(mean_squared_error(yv_exp, pv_exp))
    v_r2 = r2_score(yv_exp, pv_exp)

    t_mae = mean_absolute_error(yt_exp, pt_exp)
    t_rmse = np.sqrt(mean_squared_error(yt_exp, pt_exp))
    t_r2 = r2_score(yt_exp, pt_exp)

    print("\nValidation:")
    print(pd.DataFrame([{"MAE": v_mae, "RMSE": v_rmse, "R2": v_r2}]))
    print("\nTest:")
    print(pd.DataFrame([{"MAE": t_mae, "RMSE": t_rmse, "R2": t_r2}]))

    def predict_heads(model, loader, device, log_target: bool):
        model.eval()
        secs, probs = [], []
        with torch.no_grad():
            for xb, yb_sec, _ in loader:
                xb = xb.to(device)
                logit_long, pred_sec = model(xb)
                prob_long = torch.sigmoid(logit_long)
                s = pred_sec
                if log_target:
                    s = torch.expm1(s)
                secs.append(s.cpu().numpy())
                probs.append(prob_long.cpu().numpy())
        return np.concatenate(secs), np.concatenate(probs)

    pv_sec, pv_prob = predict_heads(model, val_loader, device, args.log_target)
    pt_sec, pt_prob = predict_heads(model, test_loader, device, args.log_target)

    # Build export DataFrames (aligning with meta_va/meta_te captured during sequence build)
    val_pred_df = pd.DataFrame({
        "A": meta_va["A"],
        "B": meta_va["B"],
        "C": meta_va["C"],
        "true_sec": yv_exp,
        #"true_sec": yv,
        #"pred_sec": pv_exp,
        "pred_sec": pv_sec,
        "pred_long_prob": pv_prob,
        "is_long_true": (yv_exp >= thr).astype(int),
        "is_long_pred": (pv_prob >= 0.5).astype(int),
        "split": "val",
    })
    test_pred_df = pd.DataFrame({
        "A": meta_te["A"],
        "B": meta_te["B"],
        "C": meta_te["C"],
        "true_sec": yt_exp,
        #"true_sec": yt,
        #"pred_sec": pt_exp,
        "pred_sec": pt_sec,
        "pred_long_prob": pt_prob,
        "is_long_true": (yt_exp >= thr).astype(int),
        "is_long_pred": (pt_prob >= 0.5).astype(int),
        "split": "test",
    })

    preds_all = pd.concat([val_pred_df, test_pred_df], ignore_index=True)
    out_csv = Path("outputs/lstm_predictions.csv")
    preds_all.to_csv(out_csv, index=False)
    print(f"\nWrote per-row predictions -> {out_csv}")

    print("\nValidation:")
    print(pd.DataFrame([{"MAE": v_mae, "RMSE": v_rmse, "R2": v_r2}]))
    print("\nTest:")
    print(pd.DataFrame([{"MAE": t_mae, "RMSE": t_rmse, "R2": t_r2}]))

    # ----------------- Write report -----------------
    with open(REPORT_OUT, "w") as f:
        f.write("# LSTM Cycle Time metrics\n\n")
        f.write(f"- Timestamp: {dt.datetime.utcnow().isoformat()}Z\n")
        f.write(f"- Seq len: {args.seq_len}, Hidden: {args.hidden}, Layers: {args.layers}, Dropout: {args.dropout}\n")
        f.write(f"- Train windows: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n\n")
        f.write("## Validation\n\n")
        f.write(f"- MAE: {v_mae:.4f}\n- RMSE: {v_rmse:.4f}\n- R2: {v_r2:.6f}\n\n")
        f.write("## Test\n\n")
        f.write(f"- MAE: {t_mae:.4f}\n- RMSE: {t_rmse:.4f}\n- R2: {t_r2:.6f}\n\n")
    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Wrote report -> {REPORT_OUT}")

if __name__ == "__main__":
    main()
