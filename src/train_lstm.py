import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import math, time

PARQUET = "outputs/features_spark.parquet"
TARGET = "CycleTime_sec"
ID = "serial_id"
ORDER = "cycle_number"

WINDOW = 10
EPOCHS = 100
PATIENCE = 10          # early stopping patience (epochs)
BATCH = 64
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMReg(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def make_windows(df, feat_cols, window=10):
    Xw, yw = [], []
    for _, g in df.groupby(ID):
        arr = g[feat_cols].values
        y = g[TARGET].values
        if len(arr) <= window:
            continue
        for i in range(window, len(arr)):
            Xw.append(arr[i-window:i])
            yw.append(y[i])
    return np.asarray(Xw, dtype=np.float32), np.asarray(yw, dtype=np.float32).reshape(-1,1)

def rmse(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))

def main():
    df = pd.read_parquet(PARQUET).dropna(subset=[TARGET])
    # Drop unused columns
    drop = [c for c in df.columns if c.lower() in ["cycle_timestamp", "good_label"]]
    for c in drop:
        if c in df.columns: df = df.drop(columns=[c])
    # Sort for deterministic windowing
    if ORDER in df.columns:
        df = df.sort_values([ID, ORDER])
    else:
        df = df.sort_values([ID])

    # Split by time-based labels produced by Spark ETL
    if "split" in df.columns:
        tr = df[df["split"] == "train"].copy()
        va = df[df["split"] == "val"].copy()
        te = df[df["split"] == "test"].copy()
    else:
        # Fallback to simple 80/20 split on index (rare)
        n = len(df); ntr = int(n*0.8)
        tr, va, te = df.iloc[:ntr], df.iloc[ntr:], df.iloc[ntr:]

    # Feature columns (numeric only, excluding id/order/target/split)
    exclude = {TARGET, ID, ORDER, "split"}
    feat_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    # Scale features with fit on TRAIN only; transform VA/TEST
    scaler = StandardScaler()
    tr.loc[:, feat_cols] = scaler.fit_transform(tr[feat_cols])
    va.loc[:, feat_cols] = scaler.transform(va[feat_cols])
    te.loc[:, feat_cols] = scaler.transform(te[feat_cols])

    # Build windows per split
    Xtr, ytr = make_windows(tr, feat_cols, WINDOW)
    Xva, yva = make_windows(va, feat_cols, WINDOW)
    Xte, yte = make_windows(te, feat_cols, WINDOW)

    # Convert to tensors
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE); ytr_t = torch.from_numpy(ytr).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE); yva_t = torch.from_numpy(yva).to(DEVICE)
    Xte_t = torch.from_numpy(Xte).to(DEVICE); yte_t = torch.from_numpy(yte).to(DEVICE)

    model = LSTMReg(in_dim=Xtr.shape[-1] if Xtr.size else len(feat_cols)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    best_state = None
    bad_epochs = 0

    def batch_iter(X, y, bs):
        n = len(X)
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            sel = idx[i:i+bs]
            yield X[sel], y[sel]

    for ep in range(1, EPOCHS+1):
        model.train()
        if len(Xtr_t) > 0:
            for xb, yb in batch_iter(Xtr_t, ytr_t, BATCH):
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            if len(Xva_t) > 0:
                pred_v = model(Xva_t).cpu().numpy()
                yv = yva_t.cpu().numpy()
                val_rmse = rmse(pred_v, yv)
            else:
                val_rmse = float("nan")
        print(f"Epoch {ep}/{EPOCHS} - val RMSE: {val_rmse:.4f}" if not math.isnan(val_rmse) else f"Epoch {ep}/{EPOCHS} - val RMSE: nan")

        # early stopping
        improved = not math.isnan(val_rmse) and val_rmse < best_val - 1e-6
        if improved:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {ep} (best val RMSE: {best_val:.4f})")
            break

    # Load best state if we have it
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        if len(Xte_t) > 0:
            pred_t = model(Xte_t).cpu().numpy()
            yt = yte_t.cpu().numpy()
            test_rmse = rmse(pred_t, yt)
        else:
            test_rmse = float("nan")

    # Save model + metrics
    Path("outputs").mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), "outputs/model_lstm.pt")
joblib.dump(scaler, "outputs/lstm_scaler.joblib")
with open("outputs/lstm_features.txt","w") as f:
    f.write("\n".join(feat_cols))
    lines = [
        "# LSTM Metrics",
        "",
        f"- Best Validation RMSE: {best_val:.6f}",
        f"- Test RMSE: {test_rmse:.6f}",
        f"- Window: {WINDOW}, Hidden: 128, Batch: {BATCH}, LR: {LR}, Patience: {PATIENCE}",
        f"- Device: {DEVICE}",
        f"- Train windows: {len(Xtr)}, Val windows: {len(Xva)}, Test windows: {len(Xte)}"
    ]
    Path("outputs/lstm_metrics.md").write_text("\n".join(lines))
    print("Saved outputs/model_lstm.pt and outputs/lstm_metrics.md")

if __name__ == "__main__":
    main()
