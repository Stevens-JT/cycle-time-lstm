import argparse, pandas as pd, numpy as np, torch, torch.nn as nn, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Defaults
FEATURES_PARQUET = "outputs/features_spark.parquet"
BASELINE_MODEL = "outputs/best_baseline.joblib"
LSTM_MODEL = "outputs/model_lstm.pt"
LSTM_SCALER = "outputs/lstm_scaler.joblib"
LSTM_FEATS = "outputs/lstm_features.txt"

TARGET = "CycleTime_sec"
ID = "serial_id"
ORDER = "cycle_number"

WINDOW = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMReg(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

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
        pred = model(torch.from_numpy(Xw).to(DEVICE)).cpu().numpy().ravel()[0]
    print(f"serial_id={sid} -> predicted next CycleTime_sec: {pred:.4f}")

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
