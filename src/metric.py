import numpy as np

def weighted_r2(y_true, y_pred, w):
    """
    Weighted R^2 using weights w.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); w = np.asarray(w)
    y_bar = np.average(y_true, weights=w)
    ss_res = np.sum(w * (y_true - y_pred)**2)
    ss_tot = np.sum(w * (y_true - y_bar)**2)
    if ss_tot == 0: 
        return 0.0
    return 1.0 - ss_res/ss_tot

def weighted_rmse(y_true, y_pred, w):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); w = np.asarray(w)
    return np.sqrt(np.sum(w * (y_true - y_pred)**2) / np.sum(w))

def weighted_mae(y_true, y_pred, w):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); w = np.asarray(w)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

def zero_mean_weighted_r2(y_true, y_pred, w):
    """
    Often responder is near-zero mean. This variant centers both series.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); w = np.asarray(w)
    y_true = y_true - np.average(y_true, weights=w)
    y_pred = y_pred - np.average(y_pred, weights=w)
    num = np.sum(w * y_true * y_pred)
    den = np.sqrt(np.sum(w * y_true**2) * np.sum(w * y_pred**2))
    if den == 0:
        return 0.0
    return (num/den)**2  # squared weighted correlation
