# Jane Street RMF – Leakage‑Safe Rolling‑CV Pipeline (LightGBM + Polars)

This repo gives you a clean, interview‑ready pipeline for the **Jane Street Real-Time Market Data Forecasting** dataset.
It uses **Polars** for fast IO/transforms and **LightGBM** with a **time‑aware rolling CV** + **purge gap** to avoid leakage.

## Quickstart

1) Install deps
```bash
pip install -r requirements.txt
```

2) Set your dataset folder in `cfg/config.yaml`:
```yaml
data_dir: /kaggle/input/jane-street-real-time-market-data-forecasting
artifacts_dir: artifacts
target: responder_6
purge_gap_time_ids: 5
n_folds: 5
use_partitions: [7,8,9]   # which train.parquet partitions to scan (adjust for RAM)
lgbm_params:
  n_estimators: 400
  learning_rate: 0.05
  max_depth: 4
  num_leaves: 31
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.0
  reg_lambda: 0.0
```

3) Train (saves models + CV metrics to `artifacts/`):
```bash
python -m src.train
```

4) Local inference over `test.parquet` (+ optional `lags.parquet`), writes `submission.csv`:
```bash
python -m src.infer
```

### Notes
- Feature recipe is **cross‑sectional (per timestamp)** (means/std + z‑scores) + simple time encodings. This is **legal at inference** because it only uses rows from the **same `(date_id,time_id)` batch**.
- Rolling CV uses **expanding window** with a **purge gap in `time_id`** to avoid label leakage.
- Start with `use_partitions: [7,8,9]` for RAM; add more when available.
- The pipeline is intentionally simple and honest; extend with symbol‑rolling features once comfortable (but beware leakage).

