import os, json, yaml, joblib
from pathlib import Path

ID_COLS = ["date_id", "time_id", "symbol_id"]
TARGET_COL = "responder_6"
WEIGHT_COL = "weight"

def load_cfg():
    with open(Path(__file__).resolve().parent.parent / "cfg" / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def feature_cols(df_cols):
    # all feature_XX columns (79)
    return [c for c in df_cols if c.startswith("feature_")]

def save_model(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)
