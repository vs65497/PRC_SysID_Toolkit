# Update Prompt: Add Intercept Column to Readout Training

**File to update:** `prc_toolkit/analysis/readout.py`

## Background

`train_readout_ols` and `train_readout_ridge` currently fit `Y = H @ W_ext` with no
intercept. This works accidentally for LI-ESN, whose reservoir outputs and i.i.d.
uniform targets are both zero-mean. It fails for any DUT whose electrode outputs have
a nonzero mean — including Ag2S-NWN driven by `bias_positive()`-shifted inputs, and
any hardware DUT with a DC offset — producing catastrophically negative R² values even
on training data.

The fix is to append a column of ones to `H` inside each training function before
fitting, and strip the bias weight back out of `W_ext` before returning, so the
caller's interface (`W_ext` maps `H` of shape `(T, N_h)` to predictions) is unchanged.

`train_readout_classify` calls `train_readout_ols` internally and will receive the fix
automatically — do not modify it separately.

`r_squared` and `nrmse` are not affected — do not modify them.

---

## Change 1: `train_readout_ols`

Replace the current body with:

```python
def train_readout_ols(H, Z) -> np.ndarray:
    """
    Ordinary least squares with intercept.

    Appends a bias (ones) column to H before fitting so the readout can absorb
    any DC offset between the reservoir outputs and the targets. The returned
    W_ext maps the original H (without the ones column) to predictions; the
    bias weight is discarded.

    H: array, shape (T, N_h). Z: targets, shape (T, K).
    Returns W_ext: shape (N_h, K).
    """
    T = H.shape[0]
    H_bias = np.hstack([H, np.ones((T, 1))])
    W_bias, *_ = np.linalg.lstsq(H_bias, Z, rcond=None)
    return W_bias[:-1]          # strip bias row; shape (N_h, K)
```

## Change 2: `train_readout_ridge`

Replace the current body with:

```python
def train_readout_ridge(H, Z, alpha=1e-4) -> np.ndarray:
    """
    Ridge regression with intercept.

    Appends a bias (ones) column to H before fitting. The bias column is not
    regularized (its diagonal entry in the Gram matrix is left at its natural
    value rather than inflated by alpha), consistent with standard practice.
    The returned W_ext maps the original H (without the ones column) to
    predictions; the bias weight is discarded.

    H: array, shape (T, N_h). Z: targets, shape (T, K).
    Returns W_ext: shape (N_h, K).
    """
    T, N_h = H.shape
    H_bias = np.hstack([H, np.ones((T, 1))])
    reg = np.eye(N_h + 1) * alpha
    reg[-1, -1] = 0.0           # do not regularize the bias term
    G = H_bias.T @ H_bias + reg
    W_bias = np.linalg.solve(G, H_bias.T @ Z)
    return W_bias[:-1]          # strip bias row; shape (N_h, K)
```

---

## What not to change

- Do not modify `train_readout_classify`, `r_squared`, or `nrmse`.
- The return shape of both functions must remain `(N_h, K)` — callers pass `H`
  of shape `(T, N_h)` and expect predictions via `H @ W_ext`. The ones column
  is internal to the fit only.

---

## Verification

After applying this change, verify with a simple sanity check:

```python
import numpy as np
from prc_toolkit.analysis.readout import train_readout_ols, train_readout_ridge

rng = np.random.default_rng(0)
T, N_h = 200, 5
H = rng.standard_normal((T, N_h)) + 3.0   # nonzero mean outputs
true_w = rng.standard_normal(N_h)
Z = H @ true_w + 7.0 + rng.standard_normal(T) * 0.01  # nonzero mean target

W_ols   = train_readout_ols(H, Z.reshape(-1, 1))
W_ridge = train_readout_ridge(H, Z.reshape(-1, 1))

pred_ols   = H @ W_ols
pred_ridge = H @ W_ridge

from prc_toolkit.analysis.readout import r_squared
assert r_squared(Z, pred_ols.ravel())   > 0.99, "OLS R² should be near 1.0"
assert r_squared(Z, pred_ridge.ravel()) > 0.99, "Ridge R² should be near 1.0"
print("OLS R²:  ", r_squared(Z, pred_ols.ravel()))
print("Ridge R²:", r_squared(Z, pred_ridge.ravel()))
```

Both R² values should be > 0.99. Before this fix, both would be strongly negative
due to the nonzero mean offset.

Also confirm that existing LI-ESN notebook results are numerically stable after this
change — since LI-ESN outputs are zero-mean, the fitted bias weight should be near
zero and R² values should be unchanged to within numerical noise.
