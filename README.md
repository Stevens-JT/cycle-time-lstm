# Cycle Time Prediction (PySpark + PyTorch)

End-to-end project for **cycle time prediction** on manufacturing data using **local PySpark** for ETL and **PyTorch/scikit-learn** for modeling.
Designed to complete Weeks 1–8 deliverables rapidly while satisfying a **Big Data** requirement via Spark.

## ✨ Features
- **Local PySpark ETL**: parses 12 time-series columns (arrays/arrays-of-arrays) and computes `CycleTime_sec` from timestamps.
- **Time-based data splits**: `train` / `val` / `test` derived from timestamp percentiles → avoids look-ahead leakage.
- **Baseline regressors**: Ridge / RandomForest / XGBoost + Markdown **metrics report**.
- **LSTM** (optional): sequence model with **early stopping** on validation RMSE and its own metrics report.
- **Inference**: predict using the best baseline (single-row CSV) or the LSTM (last WINDOW cycles for a serial).
- **Makefile**: one-command tasks (`make etl`, `make baselines`, `make lstm`, `make template`, `make clean`).

---

## 🗂 Project Structure
```
cycle-time/
├─ config.yaml
├─ requirements.txt
├─ Makefile
├─ data/
│  └─ cycles.csv                # ← place your CSV here
├─ outputs/                     # artifacts (parquet, metrics, models)
├─ notebooks/
│  ├─ 01_audit.ipynb
│  └─ 02_feature_checks.ipynb
└─ src/
   ├─ utils.py                  # robust array parser (JSON-like strings -> Python lists)
   ├─ etl_spark.py              # Spark ETL, feature summarization, time-based split
   ├─ train_baselines.py        # Ridge/RF/XGB; writes metrics_report.md
   ├─ train_lstm.py             # LSTM with early stopping; writes lstm_metrics.md
   ├─ inference.py              # baseline/LSTM inference
   └─ make_template_row.py      # creates one-row CSV template for baseline inference
```

---

## 🧰 Prerequisites
- **Python 3.9+** recommended
- **Java 8+** (for Spark) – on Windows, install a JDK and set `JAVA_HOME`
- **pip** (or conda/mamba)

> This project runs Spark in **local mode**, no cluster required.

---

## 🚀 Setup
1) **Clone or create the git repo**
   ```bash
   # if starting from scratch locally
   git init cycle-time
   cd cycle-time
   # copy the contents of this folder into your new repo (or unzip the provided archive here)
   git add .
   git commit -m "Initial commit: cycle-time project"
   ```

2) **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```

3) **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4) **Add your data & configure columns**
   - Put your CSV as `data/cycles.csv`
   - Open `config.yaml` and set the exact names for:
     - `time_series_columns:` 12 array columns
     - `id_column:` (e.g., `serial_id`)
     - `cycle_number_column:` (e.g., `cycle_number`)
     - `timestamp_column:` (e.g., `cycle_timestamp`)
     - `good_bad_label_column:` (kept but **not used** for training)

  **SHAP Feature Selection and Explainability **
  - Pre-training: set OUT_DIR = Path("outputs/shap_pre")
  - Post-training: set OUT_DIR = Path("outputs/shap_post")
    - Inputs required:
      - outputs/features_spark.parquet (from make etl)
      - For post-training: outputs/lstm_cycle_time.pt, outputs/lstm_scaler.joblib, outputs/lstm_features.txt (from make lstm)
  Pre-training feature selection
  1) Open shap_run.py and set:
    - OUT_DIR = Path("outputs/shap_pre")
  2) Run (window is only used to keep interface consistent; pre-selection operates on tabular train features):
    - python shap_run.py --window 10

  Artifacts (in outputs/shap_pre/):

    - top_features.csv — ranked features by mean |SHAP| (use top-K for training)

    - dataset_stats.csv — row/feature counts per split + target stats

    - shap_summary_bar.png — bar chart of mean |SHAP|

    - shap_beeswarm.png — distribution of SHAP values per feature

    - shap_dependence_1.png, shap_dependence_2.png, … — dependence plots for top features

    How to use it

      1) Open outputs/top_features.csv and select top-K features (e.g., 50).

      2) Save/update outputs/lstm_features.txt with one feature name per line.

      3) Retrain the LSTM:
        - make lstm
  Post-training explainability
    1) Train the LSTM (writes model + scaler + feature list):
      - make lstm
    2) Open shap_run.py and set:
      - OUT_DIR = Path("outputs/shap_post")
    3) Run with the same sequence length used in training (the script will read it from the scaler when available, but pass it to be explicit):
      - python shap_run.py --window 10
    The script will:
      - Load outputs/lstm_cycle_time.pt
      - Load outputs/lstm_scaler.joblib (contains mean, std, feature_cols, seq_len)
      - Build windows per the training feature list
      - Produce timestep-aware SHAP plots

---

## 🏃‍♀️ Run the Pipeline
### 1) ETL (Spark, local mode)
```bash
make etl
# writes:
#   outputs/raw_spark.parquet
#   outputs/features_spark.parquet
#   outputs/data_dictionary.md
```
- Computes `CycleTime_sec` as `lead(timestamp) - timestamp` within each `serial_id` ordered by `cycle_number` (fallback to timestamp).
- Adds a **time-based split** column `split ∈ {train,val,test}` via 70/85 percentiles.

### 2) Baseline Models (Ridge, RF, XGB)
```bash
make baselines
# writes:
#   outputs/results_val_baselines.csv   (validation metrics, used for model selection)
#   outputs/results_baselines.csv       (final test metrics)
#   outputs/best_baseline.joblib
#   outputs/metrics_report.md
```

### 3) LSTM (optional)
```bash
make lstm
# writes:
#   outputs/model_lstm.pt
#   outputs/lstm_scaler.joblib
#   outputs/lstm_features.txt
#   outputs/lstm_metrics.md
```
- Trains on `train`, early-stops by **validation RMSE**, reports on `test`.

---

## 🔮 Inference
### Baseline (single-row CSV)
1) Create a correctly formatted input row:
```bash
make template
# -> outputs/inference_template.csv  (one row, only feature columns)
```
2) Predict:
```bash
python src/inference.py --model baseline --input outputs/inference_template.csv
```

### LSTM (next-cycle for a serial)
```bash
python src/inference.py --model lstm --serial-id YOUR_SERIAL
# uses the last WINDOW cycles from that serial in outputs/features_spark.parquet
```

### Auto mode
```bash
python src/inference.py           # uses baseline if found, else LSTM
```

---

## ⚙️ Configuration Notes
- `config.yaml` controls column names & paths.
- If your time-series columns are **arrays of arrays** (multi-channel), the ETL summarizes each channel then aggregates.
- If arrays differ in length, summarizations (min/max/mean/std/first/last/sum/slope/len) remain robust.

---

## 🧪 Reproducibility
Artifacts and reports are written to `outputs/`:
- `metrics_report.md` (baselines) captures versions, rows, features, split strategy, and results.
- `lstm_metrics.md` records best validation RMSE, test RMSE, and settings.
- Optional: commit `outputs/*.md` and `config.yaml` to version results; ignore large Parquet/model files.

---

## 🧹 Cleaning
```bash
make clean
# removes outputs/* (keep your raw CSV in data/)
```

---

## 🛠 Troubleshooting (Windows)
- **Java/Spark not found**: Install a JDK (e.g., Azul Zulu 8/11), set `JAVA_HOME`, and ensure `java -version` works in the same terminal.
- **Long paths**: Enable long paths in Git/Windows if needed.
- **C++ build tools**: If XGBoost wheels fail, install MS Build Tools or use `pip install xgboost` prebuilt wheel for your Python version.

---

## 📄 License
Choose a license for your repo (e.g., MIT). Example `LICENSE` can be added if you want.

---

## 🙋 FAQ
**Q: Do I need Postgres or any DB?**  
A: No. Data volume is small. Spark runs in **local** mode and reads the CSV directly.

**Q: Can I use only the time-series columns?**  
A: Yes. This repo is built to summarize those 12 time-series columns and ignore pre-aggregated “trend” stats.

**Q: Where do I set the 12 columns?**  
A: `config.yaml → time_series_columns:`

**Q: How is the split determined?**  
A: By timestamp percentiles (70% train, next 15% val, last 15% test). This simulates deployment-time performance.
