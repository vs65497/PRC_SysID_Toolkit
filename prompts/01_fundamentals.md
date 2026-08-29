# Prompt 01 — Notebook: Section 1 Fundamentals (`01_fundamentals.ipynb`)

## Context

This is the first of three Jupyter notebooks in the PRC toolkit. It implements the Fundamentals
testing phase. The operator runs this notebook first to confirm the DUT is not a simple electrical
component, find the safe input amplitude |V_safe|, and characterize distortion and noise.

Assumes the shared library from Prompt 00 is already implemented and importable.

The notebook is self-contained: it imports from `prc_toolkit`, runs all tests, and writes its
key results to `results/section1_results.json` for use by later notebooks.

---

## Notebook structure

Organize as the following cells in order. Use Markdown cells for headers and explanations.

---

### Cell 1 — Markdown header

```
# Section 1: Fundamentals
Toolkit for Physical Reservoir Computer System Identification.
Tests in this section confirm the DUT is nonlinear and establish the safe operating region.
```

---

### Cell 2 — Imports and configuration

```python
import numpy as np
import matplotlib.pyplot as plt
import json, os
from prc_toolkit.config import FS, DT, V_MAX, RESULTS_DIR
from prc_toolkit.dut.base import to_input_seq
from prc_toolkit.dut.liesn import LIESN
from prc_toolkit.dut.ag2s_nwn import Ag2SNWN          # available as alternative DUT
from prc_toolkit.signals.generators import multisine, sine_sweep, bias_positive
from prc_toolkit.analysis.lissajous import (
    lissajous_response, fingerprint_grid
)
from prc_toolkit.utils.settling import run_until_settled
```

---

### Cell 3 — DUT configuration (user-facing parameters)

```python
# ── DUT Parameters ──────────────────────────────────────────────────
N_X          = 50     # Reservoir size
N_H          = 5      # Number of output electrodes
ALPHA        = 0.3    # Leaking rate
SPEC_RADIUS  = 1.1    # Spectral radius ρ(W)
SIGMA_PROC   = 0.01   # Process noise gain (set 0.0 to disable)
SIGMA_MEAS   = 0.005  # Measurement noise gain (set 0.0 to disable)
SEED         = 42

# ── Safe Region Sweep Parameters ────────────────────────────────────
N_AMP_STEPS  = 10     # Number of amplitude steps from -20 dB to 0 dB
SWEEP_DURATION = 5.0  # Seconds of sine input per amplitude step

# ── Distortion / Noise Analysis Parameters ──────────────────────────
N_PERIODS    = 8      # Number of multisine periods to record
PERIOD_DURATION = 1.0 # Duration of one period in seconds (must be integer for FFT bin alignment)

# ── Instantiate DUT ─────────────────────────────────────────────────
dut = LIESN(
    N_x=N_X, N_h=N_H, alpha=ALPHA,
    spectral_radius=SPEC_RADIUS,
    sigma_process=SIGMA_PROC,
    sigma_measure=SIGMA_MEAS,
    seed=SEED
)

# ── Alternative DUT: Ag2S Nanowire Network ───────────────────────────
# Uncomment to use Ag2SNWN instead of LIESN.
# Note: inputs must be non-negative. Wrap all input signals with bias_positive().
# Note: N_wires <= 20 recommended for performance.
#
# dut = Ag2SNWN(
#     N_wires=20, N_in=1, N_out=3,
#     connectivity=0.3,
#     sigma_process=0.01, sigma_measure=0.005,
#     seed=42
# )
# # Example: u_ms = bias_positive(multisine(..., amplitude=V_SAFE), amplitude=V_SAFE)
```

---

### Cell 4 — Markdown: Section 1.1

```
## 1.1 Disqualifying Simple Devices

Visual check. The operator applies a slow DC voltage sweep and AC sweeps at multiple
frequencies to the DUT and plots the resulting I/V or V/V curves. Compare against
the signatures in Table 2 (wire, resistor, capacitor, inductor, diode).

For simulation, this test is bypassed — the LI-ESN is known to be nonlinear by construction.
In hardware, inspect the output of this cell manually before proceeding.
```

---

### Cell 5 — Section 1.1: Simple device check (simulation stub)

```python
# ── 1.1 Simple Device Disqualification ──────────────────────────────
# For simulation: generate a slow DC ramp and plot h vs u to confirm
# the DUT is not a wire or resistor.

ramp_duration = 2.0
T_ramp = int(ramp_duration * FS)
u_ramp = np.linspace(-V_MAX, V_MAX, T_ramp)

dut.reset()
H_ramp = dut.run(to_input_seq(u_ramp))   # reshape (T,) -> (T,1) for DUT interface
h_ramp_scalar = np.linalg.norm(H_ramp, axis=1)  # scalar: L2 norm of electrode vector

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(u_ramp, h_ramp_scalar, color='steelblue', lw=1.5)
ax.set_xlabel("Input u(t)  [normalized]")
ax.set_ylabel("Output ‖h(t)‖₂  [scalar]")
ax.set_title("1.1 — I/O curve (DC ramp): Visual check for simple devices")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# NOTE for hardware: replace u_ramp / H_ramp with measured data loaded from file.
# Diode check is VISUAL ONLY — inspect the curve shape against Table 2 in the paper.
# A diode shows exponential-like forward conduction and near-zero reverse current
# with sharp breakdown. If the curve resembles a diode, do not proceed.
print("OPERATOR: Inspect the I/O curve above.")
print("Confirm the DUT is NOT a wire, resistor, capacitor, inductor, or diode.")
print("(See Table 2 in the paper for expected signatures of each simple device.)")
```

---

### Cell 6 — Markdown: Section 1.2

```
## 1.2 Safe Region of Operation

A pure sine is applied at increasing amplitudes (in dB steps from -20 dB to 0 dB
relative to V_MAX). At each step, the Response Lissajous Plot (dh/dt vs h(t)) is
shown overlaid with all previous steps, colored by amplitude.

**Operator instruction:** Inspect the LP for amplitude crossings (trajectories
intersecting themselves across different amplitude steps). The onset of crossing
marks the boundary of safe operation. Record V_safe manually in the cell below.
```

---

### Cell 7 — Section 1.2: Safe region sweep

```python
# ── 1.2 Safe Region Sweep ────────────────────────────────────────────
# I/O convention: u is scalar (float), h is vector (N_h,).
# This test uses the scalar output: h_scalar = ‖h‖₂.

sweep = sine_sweep(
    duration=SWEEP_DURATION,
    amplitude=V_MAX,
    n_steps=N_AMP_STEPS,
    fs=FS
)

sweep_results = []
for amp_linear, u_step in sweep:
    amp_dB = 20 * np.log10(amp_linear / V_MAX)  # dB relative to V_MAX

    # Warm up DUT at this amplitude, then record
    dut.reset()
    run_until_settled(dut, to_input_seq(u_step))
    H_step = dut.run(to_input_seq(u_step))

    h_scalar = np.linalg.norm(H_step, axis=1)  # shape (T,) — scalar output
    Wx_scalar = None  # set to np.linalg.norm(dut.W @ dut.x ...) if you want column D

    sweep_results.append({
        'amplitude_dB': amp_dB,
        'amplitude_linear': amp_linear,
        'u': u_step,
        'h_scalar': h_scalar,
        'Wx_scalar': Wx_scalar,
    })

# ── Overlaid Response LP (type A) colored by amplitude in dB ─────────
fig_safe = fingerprint_grid(sweep_results, titles=("Response LP (dh/dt vs h)",))
fig_safe.suptitle("1.2 — Safe Region Sweep: Response LP", fontsize=13)
plt.show()
```

---

### Cell 8 — Section 1.2: Record V_safe

```python
# ── Set V_safe manually after inspecting the LP above ───────────────
# Increase this value toward 1.0 until crossings appear, then step back.

V_SAFE = 0.8   # ← EDIT THIS based on visual inspection of the LP above

print(f"V_safe set to: {V_SAFE:.3f}  ({20*np.log10(V_SAFE):.1f} dB re V_MAX)")
print("This value will be saved to results and used by later notebooks.")
```

---

### Cell 9 — Markdown: Section 1.3

```
## 1.3 Nonlinear System Analysis

A multisine input u(t) = sin(2π·1·t) + sin(2π·3·t) + sin(2π·7·t) + sin(2π·11·t)
is applied at amplitude V_safe. The FFT output is decomposed into:

- H_linear: energy at the input frequencies f = 1, 3, 7, 11 Hz
- H_odd:    energy at odd intermodulation products f = 5, 9 Hz
- H_even:   energy at even intermodulation products f = 0, 2, 4, 6, 8, 10 Hz

The noise floor is estimated by comparing FFT spectra across multiple periods.
A signal that varies from period to period is noise; one that remains coherent is distortion.
```

---

### Cell 10 — Section 1.3.1: Distortion analysis

```python
# ── 1.3.1 Distortion Analysis ────────────────────────────────────────
# I/O convention:
#   u is scalar (float), shape (T,)
#   h used as scalar: h_scalar = ‖h‖₂, shape (T,)
# Requires: period duration = integer seconds so multisine freqs hit exact FFT bins.

T_period = int(PERIOD_DURATION * FS)  # samples per period
total_duration = N_PERIODS * PERIOD_DURATION

u_ms = multisine(duration=total_duration, amplitude=V_SAFE, fs=FS)

dut.reset()
run_until_settled(dut, to_input_seq(u_ms))
H_ms = dut.run(to_input_seq(u_ms))  # shape (T_total, N_h)
h_ms = np.linalg.norm(H_ms, axis=1)  # scalar output, shape (T_total,)

# ── FFT over one period (use last period for steady state) ───────────
h_period = h_ms[-T_period:]
freqs = np.fft.rfftfreq(T_period, d=DT)   # Hz
H_fft = np.abs(np.fft.rfft(h_period)) / T_period

# ── Classify frequency bins ──────────────────────────────────────────
# Bin index k corresponds to frequency k * (FS / T_period) = k * 1 Hz (since PERIOD_DURATION=1s)
f_linear = [1, 3, 7, 11]    # input frequencies
f_odd    = [5, 9]            # odd intermodulation products
f_even   = [0, 2, 4, 6, 8, 10]  # even intermodulation products

def get_bins(freq_list, freqs):
    return [np.argmin(np.abs(freqs - f)) for f in freq_list]

bins_lin  = get_bins(f_linear, freqs)
bins_odd  = get_bins(f_odd, freqs)
bins_even = get_bins(f_even, freqs)

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(freqs, H_fft, color='gray', lw=0.8, label='Full spectrum')
ax.stem(freqs[bins_lin],  H_fft[bins_lin],  linefmt='b-', markerfmt='bo',
        basefmt=' ', label='H_linear (f=1,3,7,11)')
ax.stem(freqs[bins_odd],  H_fft[bins_odd],  linefmt='r-', markerfmt='rs',
        basefmt=' ', label='H_odd (f=5,9)')
ax.stem(freqs[bins_even], H_fft[bins_even], linefmt='g-', markerfmt='g^',
        basefmt=' ', label='H_even (f=0,2,4,6,8,10)')
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|H(f)|")
ax.set_title("1.3.1 — Distortion Analysis: Output Frequency Components")
ax.set_xlim([0, 15])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── Summary ──────────────────────────────────────────────────────────
H_linear_power = np.sum(H_fft[bins_lin]**2)
H_odd_power    = np.sum(H_fft[bins_odd]**2)
H_even_power   = np.sum(H_fft[bins_even]**2)
total_power    = np.sum(H_fft**2)

print(f"H_linear power: {H_linear_power:.4f}  ({100*H_linear_power/total_power:.1f}% of total)")
print(f"H_odd power:    {H_odd_power:.4f}  ({100*H_odd_power/total_power:.1f}% of total)")
print(f"H_even power:   {H_even_power:.4f}  ({100*H_even_power/total_power:.1f}% of total)")
```

---

### Cell 11 — Section 1.3.2: Noise floor analysis

```python
# ── 1.3.2 Noise Floor Analysis ───────────────────────────────────────
# I/O convention:
#   u is scalar, shape (T_period,) per period
#   h used as scalar: h_scalar = ‖h‖₂
# Compare FFT spectra across N_PERIODS to separate coherent distortion from noise.

T_total = len(h_ms)
H_periods = []
U_periods = []

for p in range(N_PERIODS):
    start = p * T_period
    end   = start + T_period
    H_periods.append(np.fft.rfft(h_ms[start:end]) / T_period)
    U_periods.append(np.fft.rfft(u_ms[start:end]) / T_period)

H_arr = np.array(H_periods)  # shape (N_PERIODS, T_period//2 + 1)
U_arr = np.array(U_periods)

H_mean = H_arr.mean(axis=0)
U_mean = U_arr.mean(axis=0)

# Per-frequency noise variance (variance across periods)
sigma2_H = (1 / (N_PERIODS - 1)) * np.sum(np.abs(H_arr - H_mean)**2, axis=0)
sigma2_U = (1 / (N_PERIODS - 1)) * np.sum(np.abs(U_arr - U_mean)**2, axis=0)

# ── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(freqs, np.abs(H_mean), color='steelblue', lw=1.5, label='|H̄(f)| (mean spectrum)')
axes[0].fill_between(freqs,
    np.abs(H_mean) - np.sqrt(sigma2_H),
    np.abs(H_mean) + np.sqrt(sigma2_H),
    alpha=0.3, color='steelblue', label='±1σ noise floor')
axes[0].set_ylabel("|H(f)|")
axes[0].set_title("1.3.2 — Noise Floor: Output Spectrum with Variance Band")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(freqs, sigma2_H, color='tomato', lw=1.2, label='σ²_H (output noise variance)')
axes[1].plot(freqs, sigma2_U, color='gray',   lw=1.0, label='σ²_U (input noise variance)', ls='--')
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Noise variance")
axes[1].set_title("Noise variance by frequency bin")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 15])

plt.tight_layout()
plt.show()

noise_floor_peak = float(np.max(sigma2_H))
print(f"Peak output noise floor (σ²_H): {noise_floor_peak:.6f}")
print("Coherent distortion appears as peaks in |H̄(f)| that do NOT appear in σ²_H.")
print("Noise appears as elevated σ²_H — it varies from period to period.")
```

---

### Cell 12 — Results summary and save

```python
# ── Section 1 Results Summary ────────────────────────────────────────

results_s1 = {
    "V_SAFE":             V_SAFE,
    "H_linear_power":     float(H_linear_power),
    "H_odd_power":        float(H_odd_power),
    "H_even_power":       float(H_even_power),
    "total_spectral_power": float(total_power),
    "noise_floor_peak_sigma2_H": noise_floor_peak,
    "DUT_params": {
        "N_x":           N_X,
        "N_h":           N_H,
        "alpha":         ALPHA,
        "spectral_radius": SPEC_RADIUS,
        "sigma_process": SIGMA_PROC,
        "sigma_measure": SIGMA_MEAS,
        "seed":          SEED,
    }
}

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "section1_results.json"), "w") as f:
    json.dump(results_s1, f, indent=2)

print("=" * 50)
print("SECTION 1 RESULTS SUMMARY")
print("=" * 50)
print(f"  V_safe:              {V_SAFE:.3f}")
print(f"  H_linear power:      {H_linear_power:.4f}  ({100*H_linear_power/total_power:.1f}%)")
print(f"  H_odd power:         {H_odd_power:.4f}  ({100*H_odd_power/total_power:.1f}%)")
print(f"  H_even power:        {H_even_power:.4f}  ({100*H_even_power/total_power:.1f}%)")
print(f"  Peak noise floor:    {noise_floor_peak:.6f}")
print(f"\nResults saved to {RESULTS_DIR}section1_results.json")
print("\nProceed to Section 2 with the V_safe value above.")
```

---

## Implementation notes for Claude Code

- Generate this as a `.ipynb` file using `nbformat`. Each cell above maps to one notebook cell.
  Markdown cells use `nbformat.v4.new_markdown_cell()`, code cells use `nbformat.v4.new_code_cell()`.
- Do not execute the notebook during generation — just write it.
- The LI-ESN `run()` method should be called with `dut.reset()` before each test to ensure
  reproducible initial conditions.
- The `fingerprint_grid` in Cell 7 should only render the Response LP column (type A) for this
  section. The full 4-column grid is rendered in Section 2.
- All plots use `plt.show()` (inline display in Jupyter). Do not save figures to disk in this notebook.
- If `RESULTS_DIR` does not exist, create it in Cell 12 (already shown above).
