"""Smoke tests for prc_toolkit.analysis.readout and orthogonality."""

import numpy as np

from prc_toolkit.analysis.orthogonality import observational_orthogonality
from prc_toolkit.analysis.readout import (
    nrmse,
    predict_readout,
    r_squared,
    train_readout_classify,
    train_readout_ols,
    train_readout_ridge,
)


def test_train_readout_ols_recovers_linear_map():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(200, 4))
    W_true = rng.normal(size=(4, 2))
    Z = H @ W_true
    W_ext = train_readout_ols(H, Z)
    assert W_ext.shape == (5, 2)  # N_h + 1 rows: 4 feature weights + intercept
    np.testing.assert_allclose(predict_readout(H, W_ext), Z, atol=1e-8)
    np.testing.assert_allclose(W_ext[-1], 0.0, atol=1e-8)  # zero-mean data -> ~0 intercept


def test_predict_readout_shape_and_bare_matmul_mismatch():
    """predict_readout appends the bias column train_readout_ridge expects;
    bare H @ W_ext would raise (wrong shape) since W_ext carries the extra
    intercept row."""
    rng = np.random.default_rng(3)
    H = rng.normal(size=(20, 4))
    Z = rng.normal(size=(20, 2))
    W_ext = train_readout_ridge(H, Z)
    pred = predict_readout(H, W_ext)
    assert pred.shape == (20, 2)
    try:
        H @ W_ext
        assert False, "expected a shape mismatch from bare H @ W_ext"
    except ValueError:
        pass


def test_train_readout_ridge_shapes_and_stability():
    rng = np.random.default_rng(1)
    H = rng.normal(size=(50, 3))
    Z = rng.normal(size=(50, 1))
    W_ext = train_readout_ridge(H, Z, alpha=1e-2)
    assert W_ext.shape == (4, 1)  # N_h + 1 rows: 3 feature weights + intercept
    assert np.all(np.isfinite(W_ext))


def test_train_readout_intercept_handles_nonzero_mean():
    """Regression test for prompts/00d_readout_bias.md: without an intercept
    that survives to prediction time, a nonzero-mean H/Z relationship (as
    produced by e.g. bias_positive()'d DUT drives) gives strongly negative
    R^2 even on training data."""
    rng = np.random.default_rng(0)
    T, N_h = 200, 5
    H = rng.standard_normal((T, N_h)) + 3.0   # nonzero mean outputs
    true_w = rng.standard_normal(N_h)
    Z = H @ true_w + 7.0 + rng.standard_normal(T) * 0.01  # nonzero mean target

    W_ols = train_readout_ols(H, Z.reshape(-1, 1))
    W_ridge = train_readout_ridge(H, Z.reshape(-1, 1))

    pred_ols = predict_readout(H, W_ols)
    pred_ridge = predict_readout(H, W_ridge)

    assert r_squared(Z, pred_ols.ravel()) > 0.99
    assert r_squared(Z, pred_ridge.ravel()) > 0.99


def test_train_readout_classify_argmax():
    rng = np.random.default_rng(2)
    H = rng.normal(size=(300, 4))
    labels = rng.integers(0, 3, size=300)
    Z_onehot = np.eye(3)[labels]
    W_ext = train_readout_classify(H, Z_onehot)
    preds = np.argmax(predict_readout(H, W_ext), axis=1)
    assert preds.shape == (300,)


def test_r_squared_perfect_and_constant_guard():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == 1.0
    y_const = np.full(10, 5.0)
    assert r_squared(y_const, y_const + 1.0) == 0.0


def test_r_squared_2d_averages_columns():
    y_true = np.stack([np.arange(10.0), np.arange(10.0)], axis=1)
    y_pred = y_true.copy()
    assert r_squared(y_true, y_pred) == 1.0


def test_nrmse_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nrmse(y, y) == 0.0


def test_observational_orthogonality_orthogonal_columns():
    H = np.eye(3)
    val = observational_orthogonality(H)
    np.testing.assert_allclose(val, 1.0)


def test_observational_orthogonality_identical_columns():
    H = np.column_stack([np.arange(10.0)] * 2)
    val = observational_orthogonality(H)
    np.testing.assert_allclose(val, 0.0, atol=1e-8)


def test_observational_orthogonality_nan_single_electrode():
    H = np.arange(10.0).reshape(-1, 1)
    val = observational_orthogonality(H)
    assert np.isnan(val)
