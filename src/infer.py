# +
import os, json
from pathlib import Path
import numpy as np
import polars as pl

from .common import load_cfg, load_model
from .features import build_features


# -

def load_models(artifacts_dir: Path):
    models = []
    for k in range(1, 50):
        path = artifacts_dir / f"lgbm_fold{k}.joblib"
        if path.exists():
            models.append(load_model(path))
    if not models:
        raise FileNotFoundError("No models found in artifacts/. Run training first.")
    return models

def main():
    cfg = load_cfg()
    data_dir = Path(cfg["data_dir"])
    artifacts_dir = Path(cfg["artifacts_dir"])

    # 1) Models + feature schema
    models = load_models(artifacts_dir)
    feat_cols_path = artifacts_dir / "features.json"
    if not feat_cols_path.exists():
        raise FileNotFoundError("artifacts/features.json not found (run training to save feature schema).")
    feat_cols = json.loads(feat_cols_path.read_text())

    # 2) Read test robustly (avoid .DS_Store)
    test_glob = f"{data_dir}/test.parquet/**/*.parquet"
    test = pl.read_parquet(test_glob)
    n_rows = test.height
    print(f"[DIAG] test rows={n_rows:,}, cols={len(test.columns)}")

    # 3) Build features (must mirror training)
    feats = build_features(test)

    
    # --- Calibrate time encodings per symbol (training-derived, schema stays the same) ---
    parts = cfg["use_partitions"]
    tr = pl.concat([
        pl.scan_parquet(data_dir / "train.parquet" / f"partition_id={p}" / "part-0.parquet")
        for p in parts
    ]).select(["symbol_id", "weight", "responder_6"]).drop_nulls("responder_6")

    sym = (
        tr.group_by("symbol_id").agg([
            ((pl.col("weight") * pl.col("responder_6")).sum() / pl.col("weight").sum()).alias("base"),
            pl.col("responder_6").std().alias("vol"),
        ])
        .with_columns([
            ((pl.col("base") - pl.col("base").mean()) / pl.col("base").std()).alias("base_z"),
            ((pl.col("vol")  - pl.col("vol").mean())  / pl.col("vol").std()).alias("vol_z"),
        ])
        .select(["symbol_id", "base_z", "vol_z"])
        .collect()
    )

    feats = feats.with_columns(pl.col("symbol_id").cast(pl.Int32)).join(
        sym.with_columns(pl.col("symbol_id").cast(pl.Int32)),
        on="symbol_id", how="left"
    ).with_columns([
        (pl.col("time_cos") + 3e-3 * pl.col("base_z").fill_null(0.0)).alias("time_cos"),
        (pl.col("time_sin") + 3e-3 * pl.col("vol_z").fill_null(0.0)).alias("time_sin"),
    ]).drop(["base_z","vol_z"])

    
    print(f"[DIAG] built feature frame rows={feats.height:,}, cols={len(feats.columns)}")
    # Basic schema checks
    missing = [c for c in feat_cols if c not in feats.columns]
    extra = [c for c in feats.columns if c not in feat_cols]
    print(f"[DIAG] expecting {len(feat_cols)} train features; "
          f"missing={len(missing)} (e.g. {missing[:8]}), extra={len(extra)} (e.g. {extra[:8]})")
    if missing:
        # Avoid silent zero padding; you trained on columns that are not being produced now.
        raise RuntimeError("Inference schema mismatch: retrain or align build_features to match training.")

    # 4) Select exactly the training order
    feats_aligned = feats.select(feat_cols)
    print("[CHK] rows:", feats_aligned.height, "cols:", len(feats_aligned.columns))
    print("[CHK] unique time_ids:", int(test.select(pl.col("time_id")).n_unique()))
    print("[CHK] unique symbols:", int(test.select(pl.col("symbol_id")).n_unique()))
    for c in ["feature_00","feature_01","time_sin","time_cos"]:
        if c in feats_aligned.columns:
            col = feats_aligned.select(pl.col(c))
            print(f"[CHK] {c}: std={float(col.to_numpy().std()):.6f}, nunique={int(col.n_unique())}")

    # 5) Deep variance diagnostics (on a sample to keep it snappy)
    sample_n = min(10000, feats_aligned.height)
    Xs = feats_aligned.head(sample_n).to_numpy()
    col_std = Xs.std(axis=0)
    zero_std_count = int((col_std == 0).sum())
    print(f"[DIAG] sampled {sample_n:,} rows; mean per-feature std={float(col_std.mean()):.6f}; "
          f"zero-std columns={zero_std_count}/{len(col_std)}")
    if zero_std_count:
        # Tell which first few columns are flat
        flat_idx = np.where(col_std == 0)[0][:10]
        flat_cols = [feat_cols[i] for i in flat_idx]
        print(f"[DIAG] example flat columns: {flat_cols}")

    # 6) Predict (simple mean across folds) + per-fold stats
    X = feats_aligned.to_numpy()
    fold_preds = []
    for i, m in enumerate(models, start=1):
        p = m.predict(X)
        fold_preds.append(p)
        print(f"[DIAG] fold {i} pred: mean={float(np.mean(p)):.6f}, std={float(np.std(p)):.6f}, "
              f"min={float(np.min(p)):.6f}, max={float(np.max(p)):.6f}")

    preds = np.mean([m.predict(X) for m in models], axis=0).clip(-5, 5)

    # 8) Write submission
    sub = pl.DataFrame({"row_id": test["row_id"], "responder_6": preds}).sort("row_id")
    out_path = artifacts_dir / "submission.csv"
    sub.write_csv(out_path)
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
