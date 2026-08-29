# Generation Prompt: Notebook 4 — Benchmarks (Approximation Property)

**File to generate:** `04_benchmarks.ipynb`
**Supporting data to download and commit:** `data/sunspot_monthly.csv`

---

## Overview

Generate a new Jupyter notebook `04_benchmarks.ipynb` implementing Section 4 of the
toolkit: benchmarking the DUT's approximation property via a standard suite of RC
tasks. The notebook follows the same structure and style conventions as notebooks
01–03: markdown prose cells explaining each test, configuration cells at the top,
analysis cells below, results saved to `results/section4_results.json`.

All benchmarks use a single scalar input (N_u=1) and are compatible with hardware
DUTs having as few as 1 input and 1–3 output electrodes. MNIST and Double Pole
Balancing are explicitly out of scope for this release — do not implement them.

Performance across all regression benchmarks uses `nrmse()` from
`prc_toolkit.analysis.readout`. Do not implement a separate NRMSE function.
The XOR task reports classification accuracy (defined below), not NRMSE.

All readout training uses `train_readout_ridge()` from `prc_toolkit.analysis.readout`.
The bias row in `W_ext` must be handled via `predict_readout()` — do not use bare
`H @ W_ext` for predictions anywhere in this notebook. Locate the correct import
path for `predict_readout()` from existing usage in notebooks 02–03.

---

## Supporting data: Sunspot numbers

Before generating the notebook, download the WDC-SILSO Royal Observatory of Belgium
monthly mean sunspot number series and save it as `data/sunspot_monthly.csv`.

The file is available at:
```
https://www.sidc.be/products/sn/SN_m_tot_V2.0.txt
```

This is a fixed-width text file. Parse it and save only the two columns needed:
- `year_month`: a float of the form `YYYY.YYY` (the fractional year column,
  column index 2 in the file)
- `sunspot_number`: the monthly mean sunspot number (column index 3)

Replace any missing-data sentinel values (-1 in this dataset) with NaN, then
drop those rows. Save as CSV with a header row. Commit `data/sunspot_monthly.csv`
alongside the notebook.

---

## Top configuration cell

```python
# ── Notebook 4: Benchmarks ───────────────────────────────────────────────────
# Loads DUT configuration from results/section1_results.json (for V_SAFE and
# DUT_PARAMS) and runs a standard suite of RC benchmark tasks.

import numpy as np
import matplotlib.pyplot as plt
import json, urllib.request
from pathlib import Path

from prc_toolkit.config import DT, SEED
# Locate and import train_readout_ridge, predict_readout, nrmse from existing
# usage in notebooks 02–03.

# --- DUT selection (match your choice in notebooks 02 and 03) ---
# "liesn"    : Leaky Integrator Echo State Network
# "ag2s_nwn" : Ag2S Nanowire Network
DUT_MODEL = "liesn"

# --- Hardware mode ---
# False: simulation mode (default).
# True : hardware mode — set when running against a physical DUT.
HARDWARE_MODE = False

# --- Training / test split (synthetic tasks) ---
# Fixed step counts are used for generated sequences, consistent with RC
# literature convention. Percentage splits are not used for synthetic data
# since series length is unlimited.
N_WASHOUT = 200    # steps discarded at start (reservoir settles)
N_TRAIN   = 5000   # steps used to fit readout weights
N_TEST    = 1000   # steps used to evaluate NRMSE

# --- Training / test split (real data: sunspot) ---
# Percentage split used since dataset length is fixed.
TRAIN_FRAC = 0.70  # 70% train, 30% test (after washout)
SUNSPOT_WASHOUT = 50  # steps discarded before split

# --- XOR lag ---
# d=2 is the default. d=1 or d=3 are also reasonable choices.
XOR_LAG = 2

# --- Mackey-Glass parameters ---
MG_BETA  = 0.2
MG_GAMMA = 0.1
MG_N     = 10
MG_TAU   = 17
MG_SUBSAMPLE = 10   # integrate at dt=0.1, subsample every 10 steps
MG_X0    = 0.5
MG_H     = 1        # prediction horizon (steps ahead)

# --- Lorenz parameters ---
LZ_SIGMA = 10.0
LZ_RHO   = 28.0
LZ_BETA  = 8.0 / 3.0
LZ_DT    = 0.01     # Euler integration step
LZ_X0    = (1.0, 1.0, 1.0)
# No subsampling for Lorenz — every Euler step is used directly.

# --- NARMA-10 parameters ---
NARMA_ORDER  = 10
NARMA_ALPHA  = 0.3
NARMA_BETA   = 0.05
NARMA_GAMMA  = 1.5
NARMA_DELTA  = 0.1

# Load DUT params from Section 1 results
with open("results/section1_results.json") as f:
    s1 = json.load(f)
V_SAFE   = s1["V_SAFE"]
DUT_PARAMS = s1["DUT_PARAMS"]
```

Follow with a DUT instantiation block identical in structure to notebooks 02 and 03:
the `assert DUT_MODEL in (...)` guard, the `if/elif` conditional instantiation, and
the `prepare_input()` helper for `bias_positive()` on Ag2S-NWN. Locate
`bias_positive()` import from existing usage in the codebase — do not guess the path.

---

## Washout helper

Add a shared washout helper used by all benchmarks:

```python
def washout(dut, u_seq, n_washout):
    """
    Drive dut with the first n_washout steps of u_seq (without recording).
    Leaves dut in a settled state ready for the benchmark run.
    u_seq: shape (T, 1).
    """
    for t in range(n_washout):
        dut.step(u_seq[t])
```

---

## Section 4.1 — NARMA-10

### Prose cell

Explain: NARMA-10 (Atiya & Parlos) is a tenth-order nonlinear autoregressive moving
average system. It tests both memory (requires 10 past states) and nonlinear
processing (product terms). The task is imitation of known dynamics: given input
u(t), reproduce x_narma(t+1). Performance is evaluated with NRMSE.

State the recurrence clearly using NARMA-specific variable names to avoid collision
with reservoir state notation:

```
x_narma(t+1) = α_narma · x_narma(t)
             + β_narma · x_narma(t) · Σᵢ₌₀^{N-1} x_narma(t-i)
             + γ_narma · u(t-N+1) · u(t)
             + δ_narma
```

with α_narma=0.3, β_narma=0.05, γ_narma=1.5, δ_narma=0.1, N=10.
Input u(t) is i.i.d. uniform on [0, 0.5].

Note that the first N=10 steps are discarded as initial transient before training.

### Code cell

```python
rng = np.random.default_rng(SEED)
T_total = N_WASHOUT + N_TRAIN + N_TEST + NARMA_ORDER

# Generate NARMA-10 input and target
u_narma = rng.uniform(0, 0.5, T_total)
x_narma = np.zeros(T_total)
for t in range(NARMA_ORDER, T_total - 1):
    x_narma[t+1] = (
        NARMA_ALPHA * x_narma[t]
        + NARMA_BETA * x_narma[t] * np.sum(x_narma[t-NARMA_ORDER+1:t+1])
        + NARMA_GAMMA * u_narma[t-NARMA_ORDER+1] * u_narma[t]
        + NARMA_DELTA
    )

u_seq = prepare_input(u_narma.reshape(-1, 1))

dut.reset()
washout(dut, u_seq, N_WASHOUT + NARMA_ORDER)

H_narma = np.array([dut.step(u_seq[t]) for t in
                    range(N_WASHOUT + NARMA_ORDER, T_total - 1)])
Z_narma = x_narma[N_WASHOUT + NARMA_ORDER + 1:].reshape(-1, 1)

# Train / test split
H_train, H_test = H_narma[:N_TRAIN], H_narma[N_TRAIN:]
Z_train, Z_test = Z_narma[:N_TRAIN], Z_narma[N_TRAIN:]

W_narma = train_readout_ridge(H_train, Z_train)
pred_narma = predict_readout(H_test, W_narma)
nrmse_narma = nrmse(Z_test.ravel(), pred_narma.ravel())

print(f"NARMA-10 NRMSE: {nrmse_narma:.4f}")
```

Plot predicted vs target over the test period.

---

## Section 4.2 — Mackey-Glass System

### Prose cell

Explain: the Mackey-Glass system models circulating blood cell concentration and
produces a chaotic scalar time series for τ=17. The task is one-step-ahead prediction
of known chaotic dynamics. A low NRMSE here indicates strong memory and nonlinear
processing capability.

State the discretization:
```
x(t + Δt) = x(t) + Δt · ( β · x(t-τ) / (1 + x^n(t-τ)) − γ · x(t) )
```
Integrated at Δt=0.1, subsampled every 10 steps to produce the integer-indexed
series. Standard parameters: β=0.2, γ=0.1, n=10, τ=17, x(t)=0.5 for t<0.

### Code cell

```python
# Generate Mackey-Glass series
dt_mg   = 0.1
tau_steps = int(MG_TAU / dt_mg)   # delay in fine-grid steps
T_fine  = (N_WASHOUT + N_TRAIN + N_TEST + MG_H) * MG_SUBSAMPLE + tau_steps

mg = np.full(T_fine, MG_X0)
for t in range(tau_steps, T_fine - 1):
    mg[t+1] = mg[t] + dt_mg * (
        MG_BETA * mg[t - tau_steps] / (1 + mg[t - tau_steps] ** MG_N)
        - MG_GAMMA * mg[t]
    )
mg_series = mg[tau_steps::MG_SUBSAMPLE]   # subsample to integer-indexed series

u_mg = mg_series[:-MG_H].reshape(-1, 1)
z_mg = mg_series[MG_H:].reshape(-1, 1)

u_seq_mg = prepare_input(u_mg)

dut.reset()
washout(dut, u_seq_mg, N_WASHOUT)

H_mg = np.array([dut.step(u_seq_mg[t])
                 for t in range(N_WASHOUT, len(u_seq_mg))])
Z_mg = z_mg[N_WASHOUT:]

H_train, H_test = H_mg[:N_TRAIN], H_mg[N_TRAIN:]
Z_train, Z_test = Z_mg[:N_TRAIN], Z_mg[N_TRAIN:]

W_mg = train_readout_ridge(H_train, Z_train)
pred_mg = predict_readout(H_test, W_mg)
nrmse_mg = nrmse(Z_test.ravel(), pred_mg.ravel())

print(f"Mackey-Glass NRMSE (h={MG_H}): {nrmse_mg:.4f}")
```

Plot predicted vs target over the test period.

---

## Section 4.3 — Lorenz'63 Attractor

### Prose cell

Explain: the Lorenz'63 system is a set of three coupled ODEs producing chaotic
dynamics for standard parameters. The task is simultaneous one-step-ahead prediction
of all three state variables, given only x(t) as the scalar input. The reservoir must
reconstruct the full attractor geometry from a single observable — a demanding test
of both memory and nonlinear capacity. NRMSE is computed independently for each
output dimension and reported as the mean.

State the Euler discretization for each variable. Note Δt=0.01, no subsampling.

Note in the prose that N_y=3 here — the readout produces a 3-dimensional output,
one per Lorenz variable. This is within the hardware constraint of ≤4 output
electrodes.

### Code cell

```python
# Generate Lorenz series via forward Euler
T_lz = N_WASHOUT + N_TRAIN + N_TEST + 1
xyz  = np.zeros((T_lz, 3))
xyz[0] = LZ_X0
for t in range(T_lz - 1):
    x, y, z = xyz[t]
    xyz[t+1, 0] = x + LZ_DT * LZ_SIGMA * (y - x)
    xyz[t+1, 1] = y + LZ_DT * (x * (LZ_RHO - z) - y)
    xyz[t+1, 2] = z + LZ_DT * (x * y - LZ_BETA * z)

u_lz  = xyz[:-1, 0:1]          # x(t) as scalar input, shape (T-1, 1)
z_lz  = xyz[1:,  :]            # [x,y,z](t+1) as targets, shape (T-1, 3)

u_seq_lz = prepare_input(u_lz)

dut.reset()
washout(dut, u_seq_lz, N_WASHOUT)

H_lz = np.array([dut.step(u_seq_lz[t])
                 for t in range(N_WASHOUT, len(u_seq_lz))])
Z_lz = z_lz[N_WASHOUT:]

H_train, H_test = H_lz[:N_TRAIN], H_lz[N_TRAIN:]
Z_train, Z_test = Z_lz[:N_TRAIN], Z_lz[N_TRAIN:]

W_lz   = train_readout_ridge(H_train, Z_train)
pred_lz = predict_readout(H_test, W_lz)
nrmse_lz = float(np.mean([nrmse(Z_test[:, k], pred_lz[:, k]) for k in range(3)]))

print(f"Lorenz NRMSE (mean over x,y,z): {nrmse_lz:.4f}")
```

Plot each predicted vs target dimension over the test period (3 subplots).

---

## Section 4.4 — Sunspot Numbers

### Prose cell

Explain: sunspot numbers are a real-world dataset of unknown underlying dynamics,
recorded consistently since 1749. The task is one-step-ahead prediction of the monthly
mean sunspot number — a standard RC benchmark for prediction of unknown dynamics.
Unlike the synthetic tasks, the dataset is finite; a 70/30 train/test split is used.

Note that comparison between implementations is sensitive to dataset version and
preprocessing. This notebook uses the WDC-SILSO Royal Observatory of Belgium monthly
mean total sunspot number series (Version 2.0), downloaded from sidc.be. A commented-
out cell provides a runtime download option to fetch the latest available data.

### Code cell — data loading

```python
SUNSPOT_PATH = Path("data/sunspot_monthly.csv")

# --- Bundled dataset (default) ---
# data/sunspot_monthly.csv is committed alongside this notebook and contains
# the WDC-SILSO monthly mean sunspot number series downloaded at build time.
import pandas as pd
df_ss = pd.read_csv(SUNSPOT_PATH).dropna()
ss_values = df_ss["sunspot_number"].values.astype(float)

# --- Runtime download (commented out) ---
# Uncomment to fetch the latest WDC-SILSO data instead of the bundled snapshot.
# Requires internet access. Output format matches the bundled CSV.
#
# url = "https://www.sidc.be/products/sn/SN_m_tot_V2.0.txt"
# raw = urllib.request.urlopen(url).read().decode()
# rows = []
# for line in raw.strip().splitlines():
#     cols = line.split()
#     val = float(cols[3])
#     if val >= 0:   # -1 indicates missing data in this dataset
#         rows.append({"year_month": float(cols[2]), "sunspot_number": val})
# ss_values = np.array([r["sunspot_number"] for r in rows])
```

### Code cell — benchmark

```python
# Normalize to [0, 1] for stable reservoir driving
ss_min, ss_max = ss_values.min(), ss_values.max()
ss_norm = (ss_values - ss_min) / (ss_max - ss_min)

u_ss = ss_norm[:-1].reshape(-1, 1)
z_ss = ss_norm[1:].reshape(-1, 1)

u_seq_ss = prepare_input(u_ss)

# 70/30 split after washout
n_total  = len(u_ss) - SUNSPOT_WASHOUT
n_train  = int(n_total * TRAIN_FRAC)
n_test   = n_total - n_train

dut.reset()
washout(dut, u_seq_ss, SUNSPOT_WASHOUT)

H_ss = np.array([dut.step(u_seq_ss[t])
                 for t in range(SUNSPOT_WASHOUT, len(u_seq_ss))])
Z_ss = z_ss[SUNSPOT_WASHOUT:]

H_train, H_test = H_ss[:n_train], H_ss[n_train:]
Z_train, Z_test = Z_ss[:n_train], Z_ss[n_train:]

W_ss   = train_readout_ridge(H_train, Z_train)
pred_ss = predict_readout(H_test, W_ss)
nrmse_ss = nrmse(Z_test.ravel(), pred_ss.ravel())

print(f"Sunspot NRMSE: {nrmse_ss:.4f}")
```

Plot predicted vs target over the test period. Denormalize for the plot
(`× (ss_max - ss_min) + ss_min`) so the y-axis shows actual sunspot numbers.

---

## Section 4.5 — XOR Task

### Prose cell

Explain: XOR is a binary computation benchmark. Given a binary input stream u(t) ∈
{0,1}, the target is the XOR of two values separated by lag d:

```
y_target(t) = u(t) ⊕ u(t − d)
```

This simultaneously tests nonlinear processing (XOR is not linearly separable) and
memory (lag d requires retention of past inputs). The readout output is thresholded
at 0.5 to produce a binary prediction; performance is reported as classification
accuracy. Note: d=2 is used here. d=1 or d=3 are also reasonable choices (set
XOR_LAG in the configuration cell).

Note that targets are {0,1} and the 0.5 threshold follows from this convention.

### Code cell

```python
rng_xor = np.random.default_rng(SEED + 1)   # separate seed from NARMA
T_xor   = N_WASHOUT + N_TRAIN + N_TEST + XOR_LAG

u_xor_raw = rng_xor.integers(0, 2, T_xor).astype(float)
y_xor_raw = np.zeros(T_xor)
for t in range(XOR_LAG, T_xor):
    y_xor_raw[t] = float(int(u_xor_raw[t]) ^ int(u_xor_raw[t - XOR_LAG]))

u_seq_xor = prepare_input(u_xor_raw.reshape(-1, 1))

dut.reset()
washout(dut, u_seq_xor, N_WASHOUT + XOR_LAG)

H_xor = np.array([dut.step(u_seq_xor[t])
                  for t in range(N_WASHOUT + XOR_LAG, T_xor)])
Z_xor = y_xor_raw[N_WASHOUT + XOR_LAG:].reshape(-1, 1)

H_train, H_test = H_xor[:N_TRAIN], H_xor[N_TRAIN:]
Z_train, Z_test = Z_xor[:N_TRAIN], Z_xor[N_TRAIN:]

W_xor   = train_readout_ridge(H_train, Z_train)
pred_xor_raw = predict_readout(H_test, W_xor).ravel()
pred_xor = (pred_xor_raw >= 0.5).astype(float)
accuracy_xor = float(np.mean(pred_xor == Z_test.ravel()))

print(f"XOR Accuracy (d={XOR_LAG}): {accuracy_xor:.4f}")
```

Plot a short segment of the test period showing input, target, and prediction.

---

## Results cell

```python
section4_results = {
    "nrmse_narma10":      float(nrmse_narma),
    "nrmse_mackey_glass": float(nrmse_mg),
    "nrmse_lorenz":       float(nrmse_lz),
    "nrmse_sunspot":      float(nrmse_ss),
    "xor_accuracy":       float(accuracy_xor),
    "xor_lag":            XOR_LAG,
    "mg_horizon":         MG_H,
    "test_params": {
        "DUT_MODEL":  DUT_MODEL,
        "N_WASHOUT":  N_WASHOUT,
        "N_TRAIN":    N_TRAIN,
        "N_TEST":     N_TEST,
        "TRAIN_FRAC": TRAIN_FRAC,
    }
}

Path("results").mkdir(exist_ok=True)
with open("results/section4_results.json", "w") as f:
    json.dump(section4_results, f, indent=2)

print("Results saved to results/section4_results.json")
print(f"  NARMA-10 NRMSE:      {nrmse_narma:.4f}")
print(f"  Mackey-Glass NRMSE:  {nrmse_mg:.4f}")
print(f"  Lorenz NRMSE:        {nrmse_lz:.4f}")
print(f"  Sunspot NRMSE:       {nrmse_ss:.4f}")
print(f"  XOR Accuracy:        {accuracy_xor:.4f}  (d={XOR_LAG})")
```

---

## README and paper update notes

Add a markdown cell at the very end of the notebook (after the results cell) with
the following note for the developer:

```
# Developer note — update required
# 1. README.md: add notebook 04 to the "Running the toolkit" section and the
#    "Test summary" table. Add section4_results.json to the results schema.
#    Remove MNIST and Double Pole Balancing from any benchmark lists.
# 2. Paper: Section 4 benchmark list should reflect the implemented set only:
#    NARMA-10, Mackey-Glass, Lorenz'63, Sunspot Numbers, XOR.
#    MNIST and Double Pole Balancing are deferred (hardware I/O constraints).
#    Note the train/test split convention (200 washout / 5000 train / 1000 test
#    for synthetic tasks; 70/30 for sunspot) in the methods or supplement.
```

---

## What not to change

- Do not modify any existing notebook (01–03) or any shared library file.
- Do not implement MNIST or Double Pole Balancing.
- Do not use bare `H @ W_ext` for predictions — always use `predict_readout()`.
- Do not implement a new NRMSE function — use `nrmse()` from the shared library.
- `DUT_MODEL = "liesn"` is the default in the committed notebook.

---

## Verification

After generating the notebook and downloading `data/sunspot_monthly.csv`:

1. Run the notebook end-to-end with `DUT_MODEL = "liesn"`. Confirm all five
   benchmarks complete without error and produce finite, non-NaN NRMSE/accuracy
   values.
2. Confirm `results/section4_results.json` is written with all five keys.
3. Confirm `data/sunspot_monthly.csv` exists and loads cleanly in the sunspot cell.
4. Briefly verify sanity of results for LI-ESN at default parameters:
   - NARMA-10 NRMSE should be well below 1.0 (typical good result: ~0.1–0.3)
   - Mackey-Glass NRMSE should be well below 1.0
   - Lorenz NRMSE should be finite (chaotic system; some error is expected)
   - Sunspot NRMSE should be below 1.0
   - XOR accuracy should be above 0.5 (chance) — ideally above 0.9 for a
     capable reservoir
