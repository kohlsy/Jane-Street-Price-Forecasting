import os, json, math
from pathlib import Path
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from .common import load_cfg, ID_COLS, TARGET_COL, WEIGHT_COL, feature_cols, save_model
from .features import build_features
from .metric import weighted_r2, weighted_rmse, weighted_mae, zero_mean_weighted_r2

def load_train_frame(cfg):
    data_dir = Path(cfg["data_dir"])
    parts = cfg["use_partitions"]
    scans = [pl.scan_parquet(data_dir / "train.parquet" / f"partition_id={p}" / "part-0.parquet") for p in parts]
    df = pl.concat(scans).select(ID_COLS + [WEIGHT_COL] + [f"feature_{i:02d}" for i in range(79)] + [TARGET_COL])
    # Only supervised rows (target present)
    df = df.filter(pl.col(TARGET_COL).is_not_null())
    return df

def make_folds(df_dates, n_folds):
    # df_dates is a sorted array of unique date_id values
    # Expanding window: split by contiguous date blocks
    dates = np.array(df_dates, dtype=int)
    fold_bounds = np.array_split(dates, n_folds)
    folds = []
    for i in range(n_folds):
        val_dates = fold_bounds[i]
        train_dates = np.concatenate(fold_bounds[:i]) if i > 0 else dates[:0]
        if train_dates.size == 0:
            # for first fold, train on early chunk (to allow model fit)
            train_dates = val_dates[: max(1, len(val_dates)//2) ]
            val_dates = val_dates[max(1, len(val_dates)//2) :]
        folds.append((train_dates, val_dates))
    return folds

def apply_purge(df, train_dates, val_dates, purge_gap):
    # Remove any rows from train that are too close in time_id to val edge within the same date
    train = df.filter(pl.col("date_id").is_in(train_dates))
    val = df.filter(pl.col("date_id").is_in(val_dates))
    # Find, per date, min time_id in val; drop train rows within [min_val_tid - purge_gap, min_val_tid + purge_gap]
    if val.height == 0: 
        return train, val
    val_min = val.group_by("date_id").agg(pl.col("time_id").min().alias("val_min_tid"))
    train = train.join(val_min, on="date_id", how="left").with_columns(
        pl.when(pl.col("val_min_tid").is_not_null())
          .then((pl.col("time_id") < (pl.col("val_min_tid") - purge_gap)) | (pl.col("time_id") > (pl.col("val_min_tid") + purge_gap)))
          .otherwise(True)
          .alias("keep_row")
    ).filter(pl.col("keep_row")).drop("keep_row", "val_min_tid")
    return train, val

# +
# def fit_fold(train_df, val_df, cfg, fold_id):
#     # Build features per fold (safe: cross-sectional; no future info)
#     train_f = build_features(train_df)
#     val_f = build_features(val_df)

#     feat_cols = (
#     [c for c in train_f.columns if c.startswith("feature_")] +
#     [c for c in ("time_sin", "time_cos") if c in train_f.columns]
#     )
    
#     X_tr = train_f.select(feat_cols).to_numpy()
#     y_tr = train_f.select(TARGET_COL).to_numpy().ravel()
#     w_tr = train_f.select(WEIGHT_COL).to_numpy().ravel()

#     X_va = val_f.select(feat_cols).to_numpy()
#     y_va = val_f.select(TARGET_COL).to_numpy().ravel()
#     w_va = val_f.select(WEIGHT_COL).to_numpy().ravel()

#     params = cfg["lgbm_params"].copy()
#     params.setdefault("random_state", cfg.get("random_state", 42))
#     model = LGBMRegressor(**params)
#     model.fit(X_tr, y_tr, sample_weight=w_tr, feature_name=feat_cols)

#     pred_va = model.predict(X_va).clip(-5, 5)

#     metrics = {
#         "weighted_r2": float(weighted_r2(y_va, pred_va, w_va)),
#         "zero_mean_weighted_r2": float(zero_mean_weighted_r2(y_va, pred_va, w_va)),
#         "wrmse": float(weighted_rmse(y_va, pred_va, w_va)),
#         "wmae": float(weighted_mae(y_va, pred_va, w_va)),
#         "n_train": int(len(y_tr)),
#         "n_val": int(len(y_va)),
#     }

#     return model, feat_cols, metrics
# -

def fit_fold(train_df, val_df, cfg, fold_id):
    # --- Build features (exactly as before) ---
    train_f = build_features(train_df)
    val_f   = build_features(val_df)

    feat_cols = [c for c in train_f.columns
                 if c.startswith("feature_") or c.endswith("_cs_z") or c in ["time_sin", "time_cos"]]

    X_tr = train_f.select(feat_cols).to_numpy()
    y_tr = train_f.select(TARGET_COL).to_numpy().ravel()
    w_tr = train_f.select(WEIGHT_COL).to_numpy().ravel()

    X_va = val_f.select(feat_cols).to_numpy()
    y_va = val_f.select(TARGET_COL).to_numpy().ravel()
    w_va = val_f.select(WEIGHT_COL).to_numpy().ravel()

    # --- LightGBM model (as before) ---
    params = cfg["lgbm_params"].copy()
    params.setdefault("random_state", cfg.get("random_state", 42))
    model = LGBMRegressor(**params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, feature_name=feat_cols)

    pred_va = model.predict(X_va).clip(-5, 5)

    # ============================================================
    #                  BASELINES (no leakage)
    # ============================================================

    # 1) Zero baseline
    pred0 = np.zeros_like(y_va)

    # 2) Global weighted-mean (train-only)
    import numpy as _np
    _y_tr = train_df.select(TARGET_COL).to_numpy().ravel()
    _w_tr = train_df.select(WEIGHT_COL).to_numpy().ravel()
    gmean = float(_np.average(_y_tr, weights=_w_tr)) if _w_tr.sum() > 0 else float(_y_tr.mean())
    pred_g = _np.full_like(y_va, gmean, dtype=float)

    # 3) Per-symbol weighted-mean (train-only)
    #    Compute for symbols in train, then map to val; if a val symbol is unseen, fall back to global mean
    sym_mean = (
        train_df
        .select(["symbol_id", TARGET_COL, WEIGHT_COL])
        .group_by("symbol_id")
        .agg([
            (pl.col(WEIGHT_COL) * pl.col(TARGET_COL)).sum().alias("w_sum"),
            pl.col(WEIGHT_COL).sum().alias("w_tot"),
        ])
        .with_columns((pl.col("w_sum") / pl.col("w_tot")).fill_null(gmean).alias("sym_mean"))
        .select(["symbol_id", "sym_mean"])
    )
    val_sym = (
        val_df.select(["symbol_id"])
              .join(sym_mean, on="symbol_id", how="left")
              .with_columns(pl.col("sym_mean").fill_null(gmean))
    )
    pred_s = val_sym.select("sym_mean").to_numpy().ravel()

    # --- Metrics: model + baselines ---
    model_metrics = {
        "weighted_r2": float(weighted_r2(y_va, pred_va, w_va)),
        "zero_mean_weighted_r2": float(zero_mean_weighted_r2(y_va, pred_va, w_va)),
        "wrmse": float(weighted_rmse(y_va, pred_va, w_va)),
        "wmae": float(weighted_mae(y_va, pred_va, w_va)),
    }

    zero_metrics = {
        "wrmse_zero": float(weighted_rmse(y_va, pred0, w_va)),
        "wmae_zero": float(weighted_mae(y_va, pred0, w_va)),
    }

    gmean_metrics = {
        "wrmse_gmean": float(weighted_rmse(y_va, pred_g, w_va)),
        "wmae_gmean": float(weighted_mae(y_va, pred_g, w_va)),
    }

    psym_metrics = {
        "wrmse_psym": float(weighted_rmse(y_va, pred_s, w_va)),
        "wmae_psym": float(weighted_mae(y_va, pred_s, w_va)),
    }

    # Pretty print absolute and % improvement vs per-symbol baseline (strongest of the three)
    base_wrmses = {
        "zero": zero_metrics["wrmse_zero"],
        "gmean": gmean_metrics["wrmse_gmean"],
        "psym": psym_metrics["wrmse_psym"],
    }
    best_base_name = min(base_wrmses, key=base_wrmses.get)
    best_base_wrms = base_wrmses[best_base_name]
    imp_abs = best_base_wrms - model_metrics["wrmse"]
    imp_pct = 100.0 * imp_abs / best_base_wrms if best_base_wrms > 0 else 0.0
    print(f"[FOLD {fold_id}] WRMSE: model={model_metrics['wrmse']:.6f} | "
          f"best_baseline({best_base_name})={best_base_wrms:.6f} | "
          f"Δ={imp_abs:.6f} ({imp_pct:.2f}%)")

    metrics = {
        **model_metrics,
        **zero_metrics,
        **gmean_metrics,
        **psym_metrics,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
    }

    return model, feat_cols, metrics


def main():
    cfg = load_cfg()
    out_dir = Path(cfg["artifacts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_frame(cfg).collect()
    unique_dates = sorted(df["date_id"].unique())

    folds = make_folds(unique_dates, cfg["n_folds"])
    all_metrics = []
    feat_cols_final = None

    for k, (train_dates, val_dates) in enumerate(folds, start=1):
        train_df = df.filter(pl.col("date_id").is_in(train_dates))
        val_df = df.filter(pl.col("date_id").is_in(val_dates))
        train_df, val_df = apply_purge(df, train_dates, val_dates, cfg["purge_gap_time_ids"])

        if val_df.height == 0 or train_df.height == 0:
            continue

        model, feat_cols_fold, metrics = fit_fold(train_df, val_df, cfg, fold_id=k)
        all_metrics.append({"fold": k, **metrics})

        # Save model
        save_model(model, out_dir / f"lgbm_fold{k}.joblib")
        # Keep the most complete feature list
        if (feat_cols_final is None) or (len(feat_cols_fold) > len(feat_cols_final)):
            feat_cols_final = feat_cols_fold

    # Save CV metrics & feature list
    (out_dir / "cv_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (out_dir / "features.json").write_text(json.dumps(feat_cols_final, indent=2))

    print("CV metrics:", json.dumps(all_metrics, indent=2))
    print("Saved models to", out_dir)

if __name__ == "__main__":
    main()
