"""Readout training and goodness-of-fit metrics.

All training functions here fit only the external readout W_ext, mapping DUT
outputs H -> task targets Z. The DUT's own internal readout (e.g. LIESN's
W_out) is fixed at construction and is never modified by these functions.
"""

import numpy as np


def train_readout_ols(H, Z) -> np.ndarray:
    """
    Ordinary least squares with intercept.

    Appends a bias (ones) column to H before fitting so the readout can absorb
    any DC offset between the reservoir outputs and the targets — necessary
    for any DUT whose electrode outputs or targets are not zero-mean (e.g.
    Ag2S-NWN driven by bias_positive()'d inputs).

    The bias weight is returned as the LAST ROW of W_ext, not discarded: an
    intercept fitted during training only corrects predictions if it survives
    to prediction time. Use `predict_readout(H, W_ext)` to predict — do not
    use bare `H @ W_ext`.

    H: vector matrix, shape (T, N_h). Z: targets, shape (T, K).
    Returns W_ext: shape (N_h + 1, K) — rows 0..N_h-1 are the H weights, the
    last row is the intercept.
    """
    T = H.shape[0]
    H_bias = np.hstack([H, np.ones((T, 1))])
    W_ext, *_ = np.linalg.lstsq(H_bias, Z, rcond=None)
    return W_ext


def train_readout_ridge(H, Z, alpha=1e-4) -> np.ndarray:
    """
    Ridge regression with intercept. Adds alpha * I to the Gram matrix before
    inversion for numerical stability. Preferred over OLS when H may be
    near-singular (e.g. memory capacity and IPC computations).

    Appends a bias (ones) column to H before fitting. The bias column is not
    regularized (its diagonal entry in the Gram matrix is left at its natural
    value rather than inflated by alpha), consistent with standard practice.

    The bias weight is returned as the LAST ROW of W_ext, not discarded (see
    train_readout_ols for why). Use `predict_readout(H, W_ext)` to predict —
    do not use bare `H @ W_ext`.

    H: vector matrix, shape (T, N_h). Z: targets, shape (T, K).
    Returns W_ext: shape (N_h + 1, K) — rows 0..N_h-1 are the H weights, the
    last row is the intercept.
    """
    T, N_h = H.shape
    H_bias = np.hstack([H, np.ones((T, 1))])
    reg = np.eye(N_h + 1) * alpha
    reg[-1, -1] = 0.0           # do not regularize the bias term
    G = H_bias.T @ H_bias + reg
    W_ext = np.linalg.solve(G, H_bias.T @ Z)
    return W_ext


def train_readout_classify(H, Z_onehot) -> np.ndarray:
    """
    Linear classifier for classification benchmarks, trained via OLS (with
    intercept — see train_readout_ols) on one-hot targets. At prediction
    time, use `predict_readout(H_test, W_ext)` then argmax over the columns
    to get the predicted class.

    H: vector matrix, shape (T, N_h). Z_onehot: shape (T, n_classes).
    Returns W_ext: shape (N_h + 1, n_classes) — last row is the intercept.
    """
    W_ext = train_readout_ols(H, Z_onehot)
    return W_ext


def predict_readout(H, W_ext) -> np.ndarray:
    """
    Predict using a readout trained by train_readout_ols/ridge/classify.

    Those functions fit with an intercept and return W_ext of shape
    (N_h + 1, K), with the fitted intercept as the last row — a bare
    `H @ W_ext` would ignore it and silently ignore any DC offset the
    readout learned to correct for. This appends the matching bias column
    to H before predicting.

    H: vector matrix, shape (T, N_h). W_ext: shape (N_h + 1, K), as returned
    by train_readout_ols/ridge/classify.
    Returns predictions: shape (T, K).
    """
    H_bias = np.hstack([H, np.ones((H.shape[0], 1))])
    return H_bias @ W_ext


def r_squared(y_true, y_pred) -> float:
    """
    R^2 = 1 - SS_res / SS_tot.

    Accepts 1D arrays (single target, scalar output) or 2D arrays (per-column
    R^2, then averaged). Guards against near-zero variance targets (returns
    0.0) to avoid division by near-zero, e.g. constant delayed targets at
    boundary samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 1:
        var = np.var(y_true)
        if var < 1e-12:
            return 0.0
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot

    r2_cols = []
    for k in range(y_true.shape[1]):
        r2_cols.append(r_squared(y_true[:, k], y_pred[:, k]))
    return float(np.mean(r2_cols))


def nrmse(y_true, y_pred) -> float:
    """
    Normalized RMSE = RMSE / std(y_true). Used for regression benchmarks
    (Section 4). Guards against zero std as in r_squared.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    std = np.std(y_true)
    if std < 1e-12:
        return 0.0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / std
