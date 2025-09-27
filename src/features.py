import polars as pl
import numpy as np

from .common import ID_COLS

def _safe_denom_expr() -> pl.Expr:
    """
    Returns an expression equal to max(1.0, max(time_id)), as a Float64.
    Implemented using when/otherwise so it works on Polars-LTS.
    """
    max_tid = pl.col("time_id").max().cast(pl.Float64)
    return pl.when(max_tid < 1.0).then(pl.lit(1.0)).otherwise(max_tid)


from typing import Union
def add_time_encodings(df: Union[pl.DataFrame, pl.LazyFrame]):
    """
    Cyclic encode time_id into [sin, cos] of (time_id / max(time_id)) * 2π.
    Logic identical to your original; implemented with expression methods
    that exist on Polars-LTS (angle.sin(), angle.cos()).
    """
    denom = _safe_denom_expr()  # max(1.0, max(time_id)) as Float64
    angle = (pl.col("time_id").cast(pl.Float64) / denom) * (2.0 * np.pi)

    return df.with_columns(
        angle.sin().alias("time_sin"),
        angle.cos().alias("time_cos"),
    )

def cross_sectional_stats(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """
    Compute per-timestamp (date_id,time_id) mean/std and per-row z-scores.
    This is safe at inference since it only uses the current batch.
    """
    group_keys = ["date_id", "time_id"]
    means = df.select(
        pl.concat_list([pl.col(c) for c in feature_cols])
    )  # dummy to ensure not empty

    # mean & std per timestamp
    df_stats = df.with_columns(
        *[pl.col(c).mean().over(group_keys).alias(f"{c}_cs_mean") for c in feature_cols],
        *[pl.col(c).std().over(group_keys).alias(f"{c}_cs_std") for c in feature_cols],
    )
    # z = (x - mean) / std
    for c in feature_cols:
        df_stats = df_stats.with_columns(
            (pl.col(c) - pl.col(f"{c}_cs_mean")) / pl.when(pl.col(f"{c}_cs_std") == 0.0).then(1.0).otherwise(pl.col(f"{c}_cs_std"))
            .alias(f"{c}_cs_z")
        )
    # drop raw means/stds to keep width modest; keep z + raw features
    drop_cols = [f"{c}_cs_mean" for c in feature_cols] + [f"{c}_cs_std" for c in feature_cols]
    df_stats = df_stats.drop(drop_cols)
    return df_stats

def build_features(df: pl.DataFrame) -> pl.DataFrame:
    # Identify features
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    # Fill missing raw features with 0
    df = df.with_columns([pl.col(feat_cols).cast(pl.Float32)]).fill_null(0.0)
    # Add time encodings
    df = add_time_encodings(df)
    # Add cross-sectional z-scores
    df = cross_sectional_stats(df, feat_cols)
    # Cast everything to float32 (except ID columns)
    float_cols = [c for c in df.columns if c not in ["date_id","time_id","symbol_id","row_id","is_scored"]]
    return df.with_columns([pl.col(float_cols).cast(pl.Float32)])
