# Prompt 00 — Shared Infrastructure (`prc_toolkit/`)

## Context

We are building a software toolkit for identifying and characterizing Physical Reservoir Computers
(PRCs) from input-output measurements alone. The device under test (DUT) is treated as a black box.
The toolkit is organized as three Jupyter notebooks (one per testing phase) plus a shared library of
modules they all import. This prompt defines that shared library.

All code is Python 3. Dependencies: numpy, scipy, matplotlib, jupyter. No ML frameworks needed.

---

## Repository layout to create

```
prc_toolkit/
├── dut/
│   ├── __init__.py
│   ├── base.py          # Abstract DUT interface
│   ├── liesn.py         # Leaky-integrator Echo State Network simulation
│   └── ag2s_nwn.py      # Ag2S Nanowire Network simulation (see Prompt 00b)
├── signals/
│   ├── __init__.py
│   └── generators.py    # All input signal generators
├── analysis/
│   ├── __init__.py
│   ├── lissajous.py     # Lissajous plot rendering
│   ├── readout.py       # Readout training (OLS, ridge, classifier) and R²
│   └── orthogonality.py # Observational orthogonality metric
├── utils/
│   ├── __init__.py
│   └── settling.py      # Warmup / cooldown / settling detection
├── results/
│   └── .gitkeep         # Folder where JSON result files are written
└── config.py            # Global simulation parameters
```

---

## `config.py`

Define the following global constants. Every notebook and module imports from here.

```python
FS = 100          # Sampling rate in Hz. Satisfies Nyquist for multisine max freq of 11 Hz (2*11=22 Hz).
DT = 1.0 / FS    # Timestep in seconds.
V_MAX = 1.0      # Hard ceiling on input amplitude (hardware safety limit). Normalized units.
SETTLE_EPS = 1e-4  # Threshold for settling detection: |Δh̄/Δt| < SETTLE_EPS.
SETTLE_WINDOW = 50 # Number of samples over which to compute the settling criterion.
N_TRIALS = 2      # Number of trials for tests requiring repeated runs (FMP/ESP, consistency, CLE).
RESULTS_DIR = "results/"
```

---

## `dut/base.py` — Abstract DUT Interface

Define an abstract base class `BaseDUT` with the following interface. Every DUT (simulated or
hardware) must implement this.

```python
class BaseDUT(ABC):
    def reset(self, x0=None):
        """
        Reset internal state. x0 sets initial condition; if None, use zeros.
        Note for hardware implementations: physical reset may not be instantaneous.
        Some substrates (e.g. a cup of coffee, mechanical tensegrity) require entrainment
        on a forcing periodic sequence to reach a reproducible rest state. For this
        iteration, all DUTs are software simulations and reset is exact.
        """

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Apply a single input sample.
        u: np.ndarray of shape (N_u,) — input electrode voltages.
           For single-input DUTs, shape is (1,). Use np.atleast_1d(u_scalar) to convert.
        Returns h: np.ndarray of shape (N_h,) — output electrode vector.
        Callers who need a scalar output take np.linalg.norm(h).
        """

    def run(self, u_seq: np.ndarray, x0=None) -> np.ndarray:
        """
        Apply a sequence of inputs.
        u_seq: np.ndarray of shape (T, N_u) — input sequence.
               For single-input DUTs, shape is (T, 1). Callers generating scalar
               signals of shape (T,) should reshape: u_seq = u_scalar[:, np.newaxis].
        Returns H: np.ndarray of shape (T, N_h) — output at each timestep.
        Resets state to x0 before running.
        Default implementation calls step() in a loop; subclasses may override.
        """
```

**I/O convention (critical — enforce throughout):**
- Input `u` per timestep is always `np.ndarray` of shape `(N_u,)`. For single-input DUTs,
  N_u=1. Scalar signals from generators (shape `(T,)`) are reshaped to `(T, 1)` before
  passing to `run()`. Use the helper `to_input_seq(u_scalar)` defined below.
- Output `h` from `step()` is always a **vector** `np.ndarray` of shape `(N_h,)`, one value
  per output electrode.
- Tests that need a scalar output compute `np.linalg.norm(h, axis=-1)` themselves.
- Tests that need the full vector use `H` directly.
- The docstring of every test function must explicitly state which it uses.

**Helper function — add to `dut/base.py`:**
```python
def to_input_seq(u):
    """
    Ensure input sequence has shape (T, N_u).
    Accepts shape (T,) and returns shape (T, 1).
    Accepts shape (T, N_u) and returns unchanged.
    """
    u = np.asarray(u)
    if u.ndim == 1:
        return u[:, np.newaxis]
    return u
```

Notebooks call `to_input_seq()` on every scalar signal before passing to `dut.run()`.

---

## `dut/liesn.py` — LI-ESN Simulated DUT

Implement a Leaky-Integrator Echo State Network as a `BaseDUT` subclass. This is the primary
simulated DUT used in Sections 1–3.

### Parameters (all set at construction, stored as instance attributes)

| Parameter | Type | Description |
|-----------|------|-------------|
| `N_x` | int | Number of reservoir neurons (default 50) |
| `N_u` | int | Number of input electrodes (default 1) |
| `N_h` | int | Number of output electrodes (default 5) |
| `alpha` | float | Leaking rate ∈ (0,1] (default 0.3) |
| `spectral_radius` | float | Target spectral radius ρ(W) (default 1.1) |
| `sigma_process` | float | Process noise gain — std of Gaussian noise added inside tanh, before activation (default 0.0) |
| `sigma_measure` | float | Measurement noise gain — std of Gaussian noise added to h(t) after W_out projection (default 0.0) |
| `seed` | int | Random seed for reproducibility (default 42) |

### State update equations

```
x(n) = (1 - alpha) * x(n) + alpha * f(W_in @ [1; u(n)] + W @ x(n-1) + noise_process)
h(n) = W_out @ x(n) + noise_measure
```

Where:
- `f` is `np.tanh`
- `W_in` has shape `(N_x, 1 + N_u)` — one column for bias, N_u columns for input — drawn from U[-1,1]
- `u(n)` has shape `(N_u,)`. For single-input (N_u=1), this is a length-1 vector.
- `W` has shape `(N_x, N_x)` — drawn from N(0,1), then scaled so ρ(W) = spectral_radius
- `W_out` has shape `(N_h, N_x)` — drawn from U[-1,1], fixed (not trained). This is the
  DUT's internal readout. Do NOT train W_out. Only the external readout W_ext is trained.
- `noise_process ~ N(0, sigma_process²)` shape `(N_x,)`, new draw each step
- `noise_measure ~ N(0, sigma_measure²)` shape `(N_h,)`, new draw each step
- Set noise to zero if the corresponding sigma is 0.0 (skip the draw for speed)

### Spectral radius scaling

After drawing W from N(0,1): compute the largest absolute eigenvalue `rho = max(|eig(W)|)`,
then scale `W = W * (spectral_radius / rho)`.

### Notes

- `W_in`, `W`, `W_out` are generated once in `__init__` using `np.random.default_rng(seed)`.
- `reset(x0)` sets `self.x = x0` if provided, else `self.x = np.zeros(N_x)`.
- `step(u)` updates `self.x` in place and returns `h` of shape `(N_h,)`.
- The `run()` method from base is sufficient; no need to override.
- Store `self.rng` as the seeded generator and use it for all noise draws.

---

## `signals/generators.py` — Input Signal Generators

All functions return `np.ndarray` of shape `(T,)` — scalar time series.
All functions accept `fs=FS` and `duration` in seconds, computing `T = int(duration * fs)`.

### `multisine(duration, amplitude=1.0, fs=FS) -> np.ndarray`

```
u(t) = amplitude * (sin(2π·1·t) + sin(2π·3·t) + sin(2π·7·t) + sin(2π·11·t))
```

Frequencies are 1, 3, 7, 11 Hz — integer multiples of 1 Hz, so they land on exact FFT bins
when duration is an integer number of seconds.

### `iid_uniform(duration, amplitude=1.0, fs=FS, seed=None) -> np.ndarray`

i.i.d. samples from U[-amplitude, amplitude]. Used for Section 3 tests (consistency, MC, IPC, PSD).

### `dc_near_zero(duration, amplitude=None, fs=FS) -> np.ndarray`

Constant signal at `amplitude`. Default amplitude is `0.01 * V_MAX` (1% of safe limit).
Used as the probe signal in FMP/ESP test after initial settling.

### `sine_sweep(duration, amplitude, n_steps, fs=FS) -> list[tuple[float, np.ndarray]]`

Returns a list of `(amplitude_linear, signal)` tuples. Amplitudes are spaced evenly in dB
from -20 dB to 0 dB relative to the provided `amplitude` ceiling (which should be V_MAX).

```python
dB_steps = np.linspace(-20, 0, n_steps)
amplitudes_linear = amplitude * 10 ** (dB_steps / 20)
```

Each signal is a pure sine at 1 Hz: `A * sin(2π·1·t)`.
Used for the Section 1.2 safe region sweep.

### `poisson_spike_train(duration, rate_hz, amplitude=1.0, pulse_width_samples=2, fs=FS, seed=None) -> np.ndarray`

Generate a discrete-time approximation of a Poisson spike train.
- Draw inter-spike intervals from Exponential(1/rate_hz), convert to sample indices.
- At each spike time, set `pulse_width_samples` consecutive samples to `amplitude`.
- Clamp so no spike extends past the end of the array.
- Return signal of shape `(T,)`.

### `delayed_spike_train(u_template, spike_idx, delay_samples) -> np.ndarray`

Given a template spike train `u_template`, shift the spike at sample index `spike_idx` by
`delay_samples` (positive = later). Return the modified copy. Used to create u⁽¹⁾ from u⁽⁰⁾
in the Separation Property test.

---

## `analysis/lissajous.py` — Lissajous Plot Renderer

### `lissajous_response(h_seq, label="", color=None, ax=None)`

Plot type A: `dh/dt` vs `h(t)`.
- **Input:** `h_seq` — scalar time series, shape `(T,)`. Compute derivative via `np.gradient`.
- **Scalar input required.** Caller passes `np.linalg.norm(H, axis=1)` if H is a matrix.

### `lissajous_io(h_seq, u_seq, label="", color=None, ax=None)`

Plot type B: `h(t)` vs `u(t)`.
- **Inputs:** both scalar, shape `(T,)`.

### `lissajous_residual(h_seq, u_seq, label="", color=None, ax=None)`

Plot type C: `(h(t) - u(t))` vs `u(t)`.
- **Inputs:** both scalar, shape `(T,)`.

### `lissajous_state(Wx_seq, label="", color=None, ax=None)`

Plot type D: `d(Wx)/dt` vs `Wx(t)`. Simulation only — not available for hardware DUTs.
- **Input:** `Wx_seq` — scalar time series, shape `(T,)`. Caller computes `W @ x` and passes
  `np.linalg.norm` of that.

### `fingerprint_grid(sweep_results, titles=("Response LP", "I/O LP", "Residual LP", "State LP"))`

Compose the 4-column visual fingerprint grid (Figure 7 style).
- `sweep_results`: list of dicts, one per amplitude step, each with keys:
  `{'amplitude_dB': float, 'u': array, 'h_scalar': array, 'Wx_scalar': array or None}`
- Color each trajectory by amplitude in dB using a yellow-to-purple colormap (matching paper).
- Plot all trajectories overlaid on each subplot.
- Add a colorbar labeled "Input Amplitude (dB)".
- Return the figure.

---

## `analysis/readout.py` — Readout Training and R²

All training functions train only the **external readout W_ext**. The DUT's internal W_out
is fixed and never modified. W_ext maps DUT outputs H → task targets Z.

### `train_readout_ols(H, Z) -> np.ndarray`

Ordinary least squares. H shape `(T, N_h)`. Z shape `(T, K)`.
Returns W_ext shape `(N_h, K)`.
Use `np.linalg.lstsq(H, Z, rcond=None)[0]`.

### `train_readout_ridge(H, Z, alpha=1e-4) -> np.ndarray`

Ridge regression. Adds `alpha * I` to the Gram matrix before inversion for numerical
stability. Preferred over OLS for MC and IPC where H may be near-singular.

```python
G = H.T @ H + alpha * np.eye(H.shape[1])
W_ext = np.linalg.solve(G, H.T @ Z)
return W_ext
```

### `train_readout_classify(H, Z_onehot) -> np.ndarray`

Linear classifier for classification benchmarks. Z_onehot shape `(T, n_classes)` — one-hot
encoded class labels. Trains via OLS on the one-hot targets. At prediction time, argmax
over the output gives the predicted class.

```python
W_ext = train_readout_ols(H, Z_onehot)
# Prediction: y_pred_class = np.argmax(H_test @ W_ext, axis=1)
return W_ext
```

### `r_squared(y_true, y_pred) -> float`

Standard R² = 1 - SS_res / SS_tot.
- Handles 1D arrays (single target) and 2D arrays (per-column, then averaged).
- Guard: if `np.var(y_true) < 1e-12`, return 0.0 to avoid division by near-zero.
  This handles the case of constant delayed targets at boundary samples.

### `nrmse(y_true, y_pred) -> float`

Normalized Root Mean Square Error = RMSE / std(y_true).
Used for regression benchmarks (Section 4). Guard against zero std as above.

---

## `analysis/orthogonality.py` — Observational Orthogonality

### `observational_orthogonality(H) -> float`

**Input:** `H` — vector output matrix, shape `(T, N_h)`. Uses the full electrode vector, not scalar.

Guard: if `N_h < 2`, there are no pairs to compare. Return `float('nan')` and print a warning:
"observational_orthogonality: N_h < 2, cannot compute pairwise metric. Returning NaN."

Compute the mean pairwise (1 - |cosine similarity|) across all pairs i ≠ j of electrode columns.

```python
# For each pair (i, j), i != j:
cos_ij = (h_i @ h_j) / (norm(h_i) * norm(h_j) + 1e-12)
score_ij = 1 - abs(cos_ij)
# Return mean over all pairs
```

Returns a scalar in [0, 1]. High value → electrodes observe independent directions.
Report alongside every Section 2 and Section 3 test result.

---

## `utils/settling.py` — Settling Detection

### `is_settled(h_history, eps=SETTLE_EPS, window=SETTLE_WINDOW) -> bool`

Given recent output history `h_history` of shape `(window, N_h)` (vector outputs),
compute `|Δh̄/Δt|` as the change in mean output magnitude over the window.
Return True if below `eps`.

### `run_until_settled(dut, u_seq, max_samples=10000, eps=SETTLE_EPS, window=SETTLE_WINDOW) -> np.ndarray`

Drive the DUT with `u_seq` repeated as needed until settled or `max_samples` reached.
Returns the full output history H of shape `(T_actual, N_h)`.

If `max_samples` is reached before settling, print a warning:
"run_until_settled: max_samples reached before settling criterion met (eps={eps}).
Proceeding with unsettled state. Consider increasing max_samples or adjusting input."
The caller receives whatever history was accumulated; subsequent tests will run from
this unsettled state. The warning ensures the operator is aware.

---

## General implementation notes

- All random number generation uses `np.random.default_rng(seed)` — no `np.random.seed()`.
- Every function that produces a plot should accept an optional `ax` argument. If `ax=None`,
  create a new figure. Return the axes object.
- All modules must have docstrings on every public function specifying whether inputs/outputs
  are **scalar** or **vector**.
- Do not import from notebook files. The library is one-directional: notebooks import from library.
- Write a `tests/` folder with at minimum one smoke test per module that instantiates a LI-ESN,
  runs 1 second of input through it, and checks output shapes.
