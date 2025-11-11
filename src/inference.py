import argparse, pandas as pd, numpy as np, torch, torch.nn as nn, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Defaults
FEATURES_PARQUET = "outputs/features_spark.parquet"
BASELINE_MODEL = "outputs/best_baseline.joblib"
#LSTM_MODEL = "outputs/model_lstm.pt"

LSTM_SCALER = "outputs/lstm_scaler.joblib"
LSTM_FEATS = "outputs/lstm_features.txt"

LSTM_MODEL = "outputs/lstm_cycle_time.pt"  # saved by train_lstm.py
ID = "A"         # id column in your parquet
ORDER = "B"      # cycle number column in your parquet
TS = "C"         # timestamp column (if needed)

TARGET = "CycleTime_sec"
#ID = "serial_id"
#ORDER = "cycle_number"

WINDOW = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

'''
class LSTMReg(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
'''

class LSTMTwoHead(nn.Module):
    def __init__(self, n_feat, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=0.0 if num_layers==1 else dropout)
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
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        logit_long = self.head_cls(h)
        pred_sec   = self.head_reg(h)
        return logit_long, pred_sec


def load_features(path):
    df = pd.read_parquet(path)
    return df

def infer_baseline(args):
    model_path = Path(BASELINE_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {model_path}")
    pipe = joblib.load(model_path)

    if args.input is not None:
        # Expect a single-row CSV/JSON with columns matching features parquet (post-ETL)
        if args.input.endswith(".csv"):
            df = pd.read_csv(args.input)
        else:
            df = pd.read_json(args.input, lines=False)
    else:
        # Use the most recent 'test' rows from features parquet
        df_all = load_features(args.features)
        if "split" in df_all.columns:
            df = df_all[df_all["split"]=="test"].copy()
            # take last 5 for display
            df = df.sort_values("cycle_timestamp" if "cycle_timestamp" in df.columns else ORDER).tail(5)
        else:
            df = df_all.tail(5)

    # Drop identifiers and target if present
    drop_cols = [c for c in df.columns if c.lower() in ["serial_id","cycle_number","cycle_timestamp","good_label","split", TARGET.lower()]]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    preds = pipe.predict(X)
    out = df[[ID, ORDER]].copy() if ID in df.columns and ORDER in df.columns else pd.DataFrame(index=df.index)
    out["pred_CycleTime_sec"] = preds
    print(out.to_string(index=False))

'''
def infer_lstm(args):
    model_path = Path(LSTM_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model not found at {model_path}")
    # Load features parquet
    df = load_features(args.features).dropna(subset=[TARGET])
    # Sort
    if ORDER in df.columns:
        df = df.sort_values([ID, ORDER])
    else:
        df = df.sort_values([ID])

    # Determine feature columns
    if Path(LSTM_FEATS).exists():
        feat_cols = Path(LSTM_FEATS).read_text().strip().splitlines()
    else:
        exclude = {TARGET, ID, ORDER, "split"}
        feat_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    # Get scaler
    if Path(LSTM_SCALER).exists():
        scaler = joblib.load(LSTM_SCALER)
    else:
        # Fit scaler on train split
        if "split" in df.columns:
            tr = df[df["split"]=="train"]
        else:
            tr = df
        scaler = StandardScaler().fit(tr[feat_cols])

    # Choose serial id
    sid = args.serial_id
    if sid is None:
        # use the largest group by default (most data)
        sid = df.groupby(ID).size().sort_values(ascending=False).index[0]
    g = df[df[ID]==sid]
    if len(g) < WINDOW:
        raise ValueError(f"Not enough rows for serial_id={sid} to form a window of {WINDOW}.")

    # Build last window
    g = g.tail(WINDOW)
    Xw = g[feat_cols].values.astype(np.float32)
    Xw = scaler.transform(Xw).astype(np.float32)
    Xw = np.expand_dims(Xw, 0)  # [1, T, F]

    # Load model
    model = LSTMReg(in_dim=Xw.shape[-1]).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        #pred = model(torch.from_numpy(Xw).to(DEVICE)).cpu().numpy().ravel()[0]
        logit_long, pred = model(x)     # pred is seconds
        y_hat = pred.squeeze(1).cpu().numpy()
    print(f"serial_id={sid} -> predicted next CycleTime_sec: {pred:.4f}")
'''

def infer_lstm(args):
    model_path = Path(LSTM_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model not found at {model_path}")

    # -------- Load features parquet --------
    df_all = load_features(args.features).dropna(subset=[TARGET])

    # Sort deterministically
    sort_cols = []
    if ID in df_all.columns: sort_cols.append(ID)
    if ORDER in df_all.columns: sort_cols.append(ORDER)
    if TS in df_all.columns: sort_cols.append(TS)
    if sort_cols:
        df_all = df_all.sort_values(sort_cols)
    else:
        df_all = df_all.sort_values(df_all.columns.tolist())

    # -------- Load scaler artifacts (mean/std/feature_cols/seq_len) --------
    art_path = Path(LSTM_SCALER)
    if not art_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found at {art_path}. "
                                f"Train LSTM first so outputs/lstm_scaler.joblib exists.")
    art = joblib.load(art_path)  # dict: mean, std, feature_cols, seq_len
    feat_cols = art.get("feature_cols")
    mu  = np.asarray(art.get("mean"), dtype=np.float32)    # [F]
    std = np.asarray(art.get("std"),  dtype=np.float32)    # [F]
    SEQ = int(art.get("seq_len", 5))
    if feat_cols is None:
        # fallback to text file if needed
        if Path(LSTM_FEATS).exists():
            feat_cols = [ln.strip() for ln in Path(LSTM_FEATS).read_text().splitlines() if ln.strip()]
        else:
            raise RuntimeError("No feature list found in scaler artifact or outputs/lstm_features.txt")

    F = len(feat_cols)

    # We’ll use the test split if present; otherwise use all rows.
    if "split" in df_all.columns:
        df_use = df_all[df_all["split"] == "test"].copy()
        if len(df_use) < SEQ:
            # fallback: use all rows if test is too short
            df_use = df_all.copy()
    else:
        df_use = df_all.copy()

    # Deterministic sort by time (and optionally A/B)
    sort_cols = []
    if "C" in df_use.columns: sort_cols.append("C")
    if "B" in df_use.columns: sort_cols.append("B")
    if "A" in df_use.columns: sort_cols.append("A")
    if sort_cols:
        df_use = df_use.sort_values(sort_cols)
    else:
        df_use = df_use.sort_values(df_use.columns.tolist())

    if len(df_use) < SEQ:
        raise ValueError(f"Not enough rows ({len(df_use)}) to form a window of {SEQ} globally.")

    window_df = df_use.tail(SEQ)

    # Guard: ensure all required features are present
    missing = [c for c in feat_cols if c not in window_df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing[:8]}{'...' if len(missing)>8 else ''}")

    Xw = window_df[feat_cols].values.astype(np.float32)  # [SEQ, F]

    '''
    # -------- Choose serial id --------
    sid = args.serial_id
    if sid is None:
        # pick the id with most rows
        sid = df_all.groupby(ID).size().sort_values(ascending=False).index[0]

    g = df_all[df_all[ID] == sid]
    if len(g) < SEQ:
        raise ValueError(f"Not enough rows for serial_id={sid} to form a window of {SEQ}.")

    # -------- Build last window with the exact training features --------
    g = g.tail(SEQ)
    # guard against missing columns
    missing = [c for c in feat_cols if c not in g.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in features parquet: {missing[:8]}{'...' if len(missing)>8 else ''}")

    Xw = g[feat_cols].values.astype(np.float32)      # [SEQ, F]
    '''

    # impute with training means (mu) for any nan/inf
    Xw[~np.isfinite(Xw)] = np.nan
    mask = np.isnan(Xw)
    if mask.any():
        Xw[mask] = np.broadcast_to(mu, (SEQ, F))[mask]
    # standardize with saved mean/std
    Xw = (Xw - mu) / np.maximum(std, 1e-8)
    Xw = np.expand_dims(Xw, 0)                       # [1, SEQ, F]

    # -------- Load model (two-head) and predict --------
    model = LSTMTwoHead(n_feat=F, hidden=64, num_layers=1, dropout=0.0).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(Xw).to(DEVICE)
        logit_long, pred = model(xb)   # pred: [1,1]
        sec = float(pred.squeeze(1).cpu().numpy().ravel()[0])

    #print(f"serial_id={sid} (last {SEQ} cycles) -> predicted next CycleTime_sec: {sec:.4f}")
    print(f"(global last {SEQ} rows) -> predicted next CycleTime_sec: {sec:.4f}")

def main():
    p = argparse.ArgumentParser(description="Inference for Cycle Time Prediction")
    p.add_argument("--model", choices=["auto","baseline","lstm"], default="auto", help="Which model to use")
    p.add_argument("--features", default=FEATURES_PARQUET, help="Path to features parquet")
    p.add_argument("--input", default=None, help="Path to single-row features file (CSV/JSON) for baseline prediction")
    p.add_argument("--serial-id", default=None, help="Serial ID for LSTM next-cycle prediction (uses last WINDOW rows)")
    args = p.parse_args()

    if args.model in ["auto","baseline"] and Path(BASELINE_MODEL).exists():
        infer_baseline(args)
        return
    if args.model in ["auto","lstm"] and Path(LSTM_MODEL).exists():
        infer_lstm(args)
        return
    raise SystemExit("No suitable model found. Train baselines or LSTM first.")

if __name__ == "__main__":
    main()
