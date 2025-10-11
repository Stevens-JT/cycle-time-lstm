#!/usr/bin/env python3
import os
import math
import argparse
import datetime as dt
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --------------------
# Paths / constants
# --------------------
PARQUET_PATH = "outputs/features_spark.parquet"
MODEL_OUT = "outputs/lstm_cycle_time.pt"
REPORT_OUT = "outputs/metrics_report_lstm.md"
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_split_counts(df: pd.DataFrame):
    print("Rows total:", len(df))
    print("Has 'split' + target:", {"split" in df.columns and "CycleTime_sec" in df.columns})
    print(df["split"].value_counts(dropna=False))
    print("\nTarget non-null per split:")
    print(df[df["CycleTime_sec"].notna()].groupby("split")["CycleTime_sec"].count())


def pick_feature_columns(cols: List[str]) -> List[str]:
    # Drop meta/target columns; keep engineered numeric features
    drop = {"A", "B", "C", "D", "split", "next_ts", "CycleTime_sec"}
    feats = [c for c in cols if c not in drop]
    return feats


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


# --------------------
# Dataset & Model
# --------------------
class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_sec: np.ndarray, y_long: np.ndarray):
        # X_seq: [N, L, F]; y_sec: [N]; y_long: [N]
        self.X = torch.from_numpy(X_seq).float()
        self.y_sec = torch.from_numpy(y_sec).float()
        self.y_long = torch.from_numpy(y_long).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_sec[idx], self.y_long[idx]


class LSTMTwoHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head_cls = nn.Linear(hidden, 1)  # logits for long/short
        self.head_reg = nn.Linear(hidden, 1)  # seconds (or log-seconds)

    def forward(self, x):  # x: [B, T, F]
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # last step
        logit_long = self.head_cls(h).squeeze(1)  # [B]
        pred_sec = self.head_reg(h).squeeze(1)    # [B]
        return logit_long, pred_sec


# --------------------
# Sequence builder
# --------------------

def build_sequences(
    df: pd.DataFrame,
    feat_cols: List[str],
    seq_len: int,
    include_prev_ct: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Sliding windows on time-ordered rows: inputs are [t-L..t-1], target/meta are row t.
    Returns (X, y, meta) where meta holds A/B/C for row t.
    """
    df = df.copy()

    if include_prev_ct:
        df["prev_ct"] = df["CycleTime_sec"].shift(1)
        feat_cols_ext = feat_cols + ["prev_ct"]
    else:
        feat_cols_ext = feat_cols

    valid_idx = df[feat_cols_ext + ["CycleTime_sec"]].dropna().index
    dff = df.loc[valid_idx].reset_index(drop=True)

    X_list, y_list = [], []
    meta_A, meta_B, meta_C = [], [], []

    for t in range(seq_len, len(dff)):
        window = dff.iloc[t - seq_len : t]
        target_row = dff.iloc[t]
        X_list.append(window[feat_cols_ext].to_numpy(dtype=np.float32))
        y_list.append(float(target_row["CycleTime_sec"]))
        meta_A.append(target_row["A"])  # align meta with target row
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


# --------------------
# Evaluation helpers
# --------------------

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


def predict_heads(model, loader, device, log_target: bool):
    model.eval()
    secs, probs = [], []
    with torch.no_grad():
        for xb, yb_sec, _ in loader:
            xb = xb.to(device)
            logit_long, pred_sec = model(xb)
            if log_target:
                pred_sec = torch.expm1(pred_sec)
            secs.append(pred_sec.cpu().numpy())
            probs.append(torch.sigmoid(logit_long).cpu().numpy())
    return np.concatenate(secs), np.concatenate(probs)


# --------------------
# Main
# --------------------

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
    ap.add_argument("--loss", choices=["mae", "huber", "mse"], default="mae")
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--log_target", action="store_true",
                    help="Train on log1p(target); metrics reported back-transformed.")
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

    # Regression loss (seconds)
    if args.loss == "mae":
        reg_loss_fn = nn.L1Loss(reduction="none")
    elif args.loss == "mse":
        reg_loss_fn = nn.MSELoss(reduction="none")
    else:
        reg_loss_fn = nn.SmoothL1Loss(beta=args.huber_beta, reduction="none")

    # ----------------- Load & basic checks -----------------
    df = pd.read_parquet(PARQUET_PATH)
    print("Columns:", list(df.columns))
    print_split_counts(df)

    # Global time order
    df = df.sort_values("C").reset_index(drop=True)

    # Feature columns
    feat_cols = pick_feature_columns(list(df.columns))
    print(f"\n# Feature columns: {len(feat_cols)}")

    # ----------------- Split frames -----------------
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    te = df[df["split"] == "test"].copy()

    # ----------------- Scale numeric features (train-only fit) -----------------
    scaler = StandardScaler()
    tr_feat = tr[feat_cols].copy()
    tr_feat = tr_feat.fillna(tr_feat.mean(numeric_only=True))
    scaler.fit(tr_feat.to_numpy(dtype=np.float32))

    def transform_feats(d: pd.DataFrame) -> pd.DataFrame:
        dd = d.copy()
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

    # Keep copies of raw seconds for reporting/export
    ytr_raw, yva_raw, yte_raw = ytr.copy(), yva.copy(), yte.copy()

    # Long/short flags
    ytr_long = (ytr_raw >= args.long_thresh).astype(np.float32)
    yva_long = (yva_raw >= args.long_thresh).astype(np.float32)
    yte_long = (yte_raw >= args.long_thresh).astype(np.float32)
    print(f"Train long-rate (>= {args.long_thresh}s): {ytr_long.mean():.4f}")

    # Optional log1p transform for regression head
    if args.log_target:
        ytr = np.log1p(ytr)
        yva = np.log1p(yva)
        yte = np.log1p(yte)

    print(f"\nShapes -> Xtr:{Xtr.shape}, ytr:{ytr.shape}; Xva:{Xva.shape}, yva:{yva.shape}; Xte:{Xte.shape}, yte:{yte.shape}")
    if Xtr.shape[0] == 0 or Xva.shape[0] == 0 or Xte.shape[0] == 0:
        raise RuntimeError("Not enough sequential data to train/evaluate. Try lowering seq_len.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SeqDataset(Xtr, ytr, ytr_long)
    val_ds = SeqDataset(Xva, yva, yva_long)
    test_ds = SeqDataset(Xte, yte, yte_long)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ----------------- Model -----------------
    model = LSTMTwoHead(
        in_dim=Xtr.shape[-1],
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Class loss w/ imbalance handling
    p_long = max(1e-6, float(ytr_long.mean()))
    pos_weight = torch.tensor([(1.0 - p_long) / p_long], dtype=torch.float32, device=device) if 0 < p_long < 1 else None
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ----------------- Train -----------------
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

            # losses
            loss_cls = cls_loss_fn(logit_long, yb_long)
            loss_reg_each = reg_loss_fn(pred_sec, yb_sec)  # [B]
            reg_w = torch.where(
                yb_long > 0.5,
                torch.tensor(args.reg_w_long, device=device),
                torch.tensor(args.reg_w_short, device=device),
            )
            loss_reg = (loss_reg_each * reg_w).mean()
            loss = args.alpha_cls * loss_cls + args.alpha_reg * loss_reg

            loss.backward()
            optim.step()

            bs = xb.size(0)
            epoch_loss += loss.item() * bs
            n_samples += bs

        epoch_loss /= max(1, n_samples)

        # quick val for early stop (already back-transformed if log_target)
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
    v_mae, v_rmse, v_r2, yv_sec, pv_sec_eval = evaluate_split_twohead(model, val_loader, device, args.log_target)
    t_mae, t_rmse, t_r2, yt_sec, pt_sec_eval = evaluate_split_twohead(model, test_loader, device, args.log_target)

    print("\nValidation:")
    print(pd.DataFrame([{"MAE": v_mae, "RMSE": v_rmse, "R2": v_r2}]))
    print("\nTest:")
    print(pd.DataFrame([{"MAE": t_mae, "RMSE": t_rmse, "R2": t_r2}]))

    # ----------------- Export per-row predictions -----------------
    thr = float(args.long_thresh)
    pv_sec, pv_prob = predict_heads(model, val_loader, device, args.log_target)
    pt_sec, pt_prob = predict_heads(model, test_loader, device, args.log_target)

    val_pred_df = pd.DataFrame({
        "A": meta_va["A"],
        "B": meta_va["B"],
        "C": meta_va["C"],
        "true_sec": yva_raw,
        "pred_sec": pv_sec,
        "pred_long_prob": pv_prob,
        "is_long_true": (yva_raw >= thr).astype(int),
        "is_long_pred": (pv_prob >= 0.5).astype(int),
        "split": "val",
    })
    test_pred_df = pd.DataFrame({
        "A": meta_te["A"],
        "B": meta_te["B"],
        "C": meta_te["C"],
        "true_sec": yte_raw,
        "pred_sec": pt_sec,
        "pred_long_prob": pt_prob,
        "is_long_true": (yte_raw >= thr).astype(int),
        "is_long_pred": (pt_prob >= 0.5).astype(int),
        "split": "test",
    })

    preds_all = pd.concat([val_pred_df, test_pred_df], ignore_index=True)
    out_csv = Path("outputs/lstm_predictions.csv")
    preds_all.to_csv(out_csv, index=False)
    print(f"\nWrote per-row predictions -> {out_csv}")

    # ----------------- Write summary report -----------------
    with open(REPORT_OUT, "w") as f:
        f.write("# LSTM Cycle Time metrics\n\n")
        f.write(f"- Timestamp: {dt.datetime.utcnow().isoformat()}Z\n")
        f.write(
            f"- Seq len: {args.seq_len}, Hidden: {args.hidden}, Layers: {args.layers}, Dropout: {args.dropout}\n"
        )
        f.write(
            f"- Train windows: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n\n"
        )
        f.write("## Validation\n\n")
        f.write(f"- MAE: {v_mae:.4f}\n- RMSE: {v_rmse:.4f}\n- R2: {v_r2:.6f}\n\n")
        f.write("## Test\n\n")
        f.write(f"- MAE: {t_mae:.4f}\n- RMSE: {t_rmse:.4f}\n- R2: {t_r2:.6f}\n\n")

    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Wrote report -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
