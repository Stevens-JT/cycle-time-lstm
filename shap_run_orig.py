# shap_run.py — checkpoint-compatible + feature auto-alignment
import argparse, warnings, re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# ---- Force CPU to avoid CUDA warning
DEVICE = "cpu"

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

WINDOW_DEFAULT = 10
TARGET = "CycleTime_sec"
ID = "serial_id"
ORDER = "cycle_number"

OUTPUTS = Path("outputs")
FEATURES_PARQUET = OUTPUTS / "features_spark.parquet"
CYCLES_CSV = Path("cycles.csv")  # fallback if needed
MODEL_PATH = OUTPUTS / "lstm_cycle_time.pt"  # your filename
SCALER_PATH = OUTPUTS / "lstm_scaler.joblib"
FEATS_PATH = OUTPUTS / "lstm_features.txt"   # if training-time list exists, we’ll prefer it

# ---- Model that matches checkpoint: hidden=64, two heads (reg + cls)
class LSTMRegMTL(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.head_reg = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_cls = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, task="reg"):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y_reg = self.head_reg(h)
        y_cls = self.head_cls(h)
        if task == "both":
            return y_reg, y_cls
        return y_reg if task == "reg" else y_cls

def load_features_df():
    if FEATURES_PARQUET.exists():
        return pd.read_parquet(FEATURES_PARQUET)
    if CYCLES_CSV.exists():
        return pd.read_csv(CYCLES_CSV)
    raise FileNotFoundError("No features file found. Expected outputs/features_spark.parquet or cycles.csv")

# Conservative non-feature patterns (ids, meta, splits, timestamps, labels, etc.)
_EXCLUDE_PATTERNS = re.compile(
    r"(?:^|_)(id|serial|job|idx|index|number|timestamp|ts|date|time|label|split|target|cycle[_]?time)\b",
    re.IGNORECASE
)

def training_feature_list(df: pd.DataFrame, input_size_ckpt: int) -> list[str]:
    # 1) If training-time list exists, use it
    if FEATS_PATH.exists():
        feats = [ln.strip() for ln in FEATS_PATH.read_text().splitlines() if ln.strip()]
        feats = [c for c in feats if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
        if len(feats) != input_size_ckpt:
            raise ValueError(
                f"lstm_features.txt has {len(feats)} numeric cols, but checkpoint expects {input_size_ckpt}."
            )
        return feats

    # 2) Derive from parquet: numeric, exclude known meta/ids/targets
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    # hard excludes
    hard_exclude = {TARGET, ID, ORDER, "cycle_timestamp", "good_label", "split"}
    candidates = [c for c in numeric_cols if c not in hard_exclude and not _EXCLUDE_PATTERNS.search(c)]
    # If still too many, deterministically trim by name (stable, reproducible)
    if len(candidates) > input_size_ckpt:
        candidates = sorted(candidates)[:input_size_ckpt]
    if len(candidates) < input_size_ckpt:
        raise ValueError(
            f"Only {len(candidates)} numeric feature candidates after exclusion, "
            f"but checkpoint expects {input_size_ckpt}. Add back columns or provide lstm_features.txt."
        )
    return candidates

def build_windows(df, feat_cols, window):
    if ID in df.columns and ORDER in df.columns:
        df = df.sort_values([ID, ORDER])
        groups = [g for _, g in df.groupby(ID)]
    else:
        df = df.sort_values(ORDER if ORDER in df.columns else df.columns[0])
        groups = [df]
    Xseq, y = [], []
    for g in groups:
        g = g.dropna(subset=[TARGET]) if TARGET in g.columns else g.copy()
        vals = g[feat_cols].values.astype(np.float32)
        tgt = g[TARGET].values.astype(np.float32) if TARGET in g.columns else None
        for i in range(len(g) - window):
            Xseq.append(vals[i:i+window])
            if tgt is not None:
                y.append(tgt[i+window])
    Xseq = np.array(Xseq, dtype=np.float32)
    y = np.array(y, dtype=np.float32) if len(y) else None
    return Xseq, y

def aggregate_shap_over_time(shap_values, feat_cols, window):
    N, dim = shap_values.shape
    F = len(feat_cols)
    assert dim == window * F, f"Expected {window*F} attributions per sample, got {dim}"
    sv = shap_values.reshape(N, window, F)
    return np.abs(sv).mean(axis=1).mean(axis=0)  # [F]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    ap.add_argument("--max_bg", type=int, default=30)
    ap.add_argument("--max_eval", type=int, default=120)
    ap.add_argument("--nsamples", type=int, default=100)
    args = ap.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # 1) Load df
    df = load_features_df()
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])

    # 2) Inspect checkpoint to get expected input size + hidden size
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    Wih = state["lstm.weight_ih_l0"]
    hidden_ckpt = Wih.shape[0] // 4
    input_size_ckpt = Wih.shape[1]

    # 3) Build/align feature list to checkpoint input size
    feat_cols = training_feature_list(df, input_size_ckpt)
    # Choose training slice to compute statistics
    tr_df = df[df["split"] == "train"] if "split" in df.columns else df

    # Per-feature means for imputation
    col_means = tr_df[feat_cols].mean().astype("float32").values

    # Persist the final feature list for reproducibility
    (OUTPUTS / "lstm_features_autosel.txt").write_text("\n".join(feat_cols))

    # 4) Scaler
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
    else:
        from sklearn.preprocessing import StandardScaler
        #scaler = StandardScaler().fit(df[feat_cols].values)
        tr_vals = tr_df[feat_cols].values.astype("float32")

        # Impute NaNs in training matrix with per-feature means
        nan_mask = np.isnan(tr_vals)
        if nan_mask.any():
            rows, cols = np.where(nan_mask)
            tr_vals[rows, cols] = col_means[cols]

        scaler = StandardScaler().fit(tr_vals)
        joblib.dump(scaler, SCALER_PATH)

    # 5) Windows + scale
    Xseq, y = build_windows(df, feat_cols, args.window)
    if Xseq.shape[0] < max(args.max_bg + 10, 30):
        raise ValueError(f"Not enough samples for SHAP: have {Xseq.shape[0]}")
    for t in range(Xseq.shape[1]):
        #Xseq[:, t, :] = scaler.transform(Xseq[:, t, :])
        Xt = Xseq[:, t, :]  # [N, F]
        nan_mask = np.isnan(Xt)
        if nan_mask.any():
            r, c = np.where(nan_mask)
            Xt[r, c] = col_means[c]  # impute with training means
        Xseq[:, t, :] = scaler.transform(Xt)

    # 6) Reconstruct model EXACTLY like the checkpoint (hidden=hidden_ckpt)
    model = LSTMRegMTL(in_dim=len(feat_cols), hidden=hidden_ckpt).to(DEVICE)
    #model.load_state_dict(state, strict=True)
    state_pruned = {k: v for k, v in state.items() if not k.startswith("head_cls.")}
    missing, unexpected = model.load_state_dict(state_pruned, strict=False)
    model.eval()

    # 7) SHAP wrapper
    def predict_fn(flat_2d: np.ndarray) -> np.ndarray:
        W, F = args.window, len(feat_cols)
        Xr = flat_2d.reshape(-1, W, F).astype(np.float32)
        with torch.no_grad():
            tens = torch.from_numpy(Xr).to(DEVICE)
            y_reg = model(tens, task="reg")
            if isinstance(y_reg, tuple):
                y_reg = y_reg[0]
            out = y_reg.cpu().numpy().reshape(-1)
        return out

    # Subsample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(Xseq.shape[0]), size=min(Xseq.shape[0], args.max_eval), replace=False)
    evalX = Xseq[idx]
    bidx = rng.choice(idx, size=min(len(idx), args.max_bg), replace=False)
    bgX = Xseq[bidx]
    evalX_flat = evalX.reshape(evalX.shape[0], -1)
    bgX_flat = bgX.reshape(bgX.shape[0], -1)

    top_rows = None
    if SHAP_AVAILABLE:
        warnings.filterwarnings("ignore")
        explainer = shap.KernelExplainer(predict_fn, bgX_flat, link="identity")
        shap_values = explainer.shap_values(evalX_flat, nsamples=args.nsamples)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs = aggregate_shap_over_time(shap_values, feat_cols, args.window)
        order = np.argsort(-mean_abs)
        top_rows = [(feat_cols[i], float(mean_abs[i])) for i in order[:20]]

        # Plots (last timestep only for visualization clarity)
        import matplotlib.pyplot as plt
        resh = shap_values.reshape(evalX.shape[0], args.window, len(feat_cols))[:, -1, :]
        lastX = pd.DataFrame(evalX[:, -1, :], columns=feat_cols)

        shap.summary_plot(resh, features=lastX, feature_names=feat_cols, plot_type="bar", show=False)
        plt.tight_layout(); plt.savefig(OUTPUTS / "shap_summary_bar.png", dpi=200); plt.close()

        shap.summary_plot(resh, features=lastX, feature_names=feat_cols, show=False)
        plt.tight_layout(); plt.savefig(OUTPUTS / "shap_beeswarm.png", dpi=200); plt.close()

        top1 = feat_cols[order[0]]
        shap.dependence_plot(top1, resh, lastX, interaction_index=None, show=False)
        plt.tight_layout(); plt.savefig(OUTPUTS / f"shap_dependence_{top1}.png", dpi=200); plt.close()

        if len(order) > 1:
            top2 = feat_cols[order[1]]
            shap.dependence_plot(top2, resh, lastX, interaction_index=None, show=False)
            plt.tight_layout(); plt.savefig(OUTPUTS / f"shap_dependence_{top2}.png", dpi=200); plt.close()
    else:
        # Fallback: permutation importance with a linear probe
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error
        X_last = Xseq[:, -1, :]
        n = len(X_last); ntr = int(0.7 * n)
        Xtr, Xte = X_last[:ntr], X_last[ntr:]
        ytr, yte = (y[:ntr], y[ntr:])
        probe = Ridge(alpha=1.0).fit(Xtr, ytr)
        base = mean_absolute_error(yte, probe.predict(Xte))
        scores = []
        rng = np.random.default_rng(0)
        for j in range(Xte.shape[1]):
            Xp = Xte.copy(); rng.shuffle(Xp[:, j])
            mae = mean_absolute_error(yte, probe.predict(Xp))
            scores.append(mae - base)
        order = np.argsort(-np.array(scores))
        top_rows = [(feat_cols[i], float(scores[i])) for i in order[:20]]
        import matplotlib.pyplot as plt
        plt.figure()
        plt.bar([f for f,_ in top_rows[:15]], [v for _,v in top_rows[:15]])
        plt.xticks(rotation=60, ha="right"); plt.tight_layout()
        plt.savefig(OUTPUTS / "perm_importance.png", dpi=200); plt.close()

    if top_rows is not None:
        pd.DataFrame(top_rows, columns=["feature","importance"]).to_csv(OUTPUTS / "top_features.csv", index=False)

    # Stats for LaTeX
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    desc = df[num_cols].describe().T.reset_index().rename(columns={"index":"column"})
    desc.to_csv(OUTPUTS / "dataset_stats.csv", index=False)

    print("SHAP run complete. Artifacts written to outputs/")

if __name__ == "__main__":
    main()
