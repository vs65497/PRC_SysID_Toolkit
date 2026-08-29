# Prompt 02 — Notebook: Section 2 System Identification (`02_system_identification.ipynb`)

## Context

Second of three notebooks. Implements qualitative system identification: the visual fingerprint,
fading memory / echo state property test, and separation property test. Also computes
observational orthogonality alongside every test.

Loads `results/section1_results.json` to retrieve V_safe. Writes key results to
`results/section2_results.json`.

Assumes shared library from Prompt 00 is implemented and importable.

---

## Notebook structure

---

### Cell 1 — Markdown header

```
# Section 2: System Identification
Qualitative tests for the Fading Memory Property (FMP), Echo State Property (ESP),
and Separation Property (SP). Visual fingerprinting via Lissajous Plot series.
Observational orthogonality is reported alongside every test.
```

---

### Cell 2 — Imports and configuration

```python
import numpy as np
import matplotlib.pyplot as plt
import json, os
from prc_toolkit.config import FS, DT, V_MAX, N_TRIALS, RESULTS_DIR
from prc_toolkit.dut.base import to_input_seq
from prc_toolkit.dut.liesn import LIESN
from prc_toolkit.dut.ag2s_nwn import Ag2SNWN          # available as alternative DUT
from prc_toolkit.signals.generators import (
    multisine, sine_sweep, dc_near_zero,
    poisson_spike_train, delayed_spike_train, bias_positive
)
from prc_toolkit.analysis.lissajous import (
    lissajous_response, lissajous_io, lissajous_residual,
    lissajous_state, fingerprint_grid
)
from prc_toolkit.analysis.orthogonality import observational_orthogonality
from prc_toolkit.utils.settling import run_until_settled
```

---

### Cell 3 — Load Section 1 results and configure DUT

```python
# ── Load V_safe from Section 1 ───────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "section1_results.json")) as f:
    s1 = json.load(f)

V_SAFE  = s1["V_SAFE"]
DUT_CFG = s1["DUT_params"]

print(f"Loaded V_safe = {V_SAFE:.3f} from Section 1 results.")

# ── DUT (same configuration as Section 1) ───────────────────────────
dut = LIESN(
    N_x=DUT_CFG["N_x"],
    N_h=DUT_CFG["N_h"],
    alpha=DUT_CFG["alpha"],
    spectral_radius=DUT_CFG["spectral_radius"],
    sigma_process=DUT_CFG["sigma_process"],
    sigma_measure=DUT_CFG["sigma_measure"],
    seed=DUT_CFG["seed"]
)

# ── Section 2 test parameters ────────────────────────────────────────
N_AMP_STEPS_FP    = 10     # Amplitude steps for the visual fingerprint sweep
FP_N_PERIODS      = 5      # Number of sine periods to record per amplitude step.
                            # Fixed period count rather than waiting for settling —
                            # chaotic reservoirs may never settle, and unsettled
                            # transient behavior is informative for the operator.

SETTLE_DURATION   = 10.0   # Seconds of settling input for FMP/ESP test
PROBE_DURATION    = 10.0   # Seconds of near-zero DC probe after settling (FMP test)
ESP_SWAP_DURATION = 10.0   # Seconds of u_C input for ESP observation

POISSON_RATE_HZ   = 5.0    # Average spike rate for separation property test
SPIKE_DURATION    = 15.0   # Total duration of spike train (seconds)
PULSE_WIDTH       = 2      # Spike pulse width in samples
SPIKE_IDX         = int(2.0 * FS)  # Index of the spike to delay (2 seconds in)
DELAY_SAMPLES     = int(0.1 * FS)  # Delay displacement in samples (100 ms)
SEED_SPIKES       = 7
```

---

### Cell 4 — Markdown: Section 2.1

```
## 2.1 Visual Fingerprint via Lissajous Plot Series

A pure sine is applied at N_AMP_STEPS amplitude levels, evenly spaced in dB from
-20 dB to 0 dB relative to V_safe. The four Lissajous Plot types are rendered overlaid,
colored by input amplitude (purple = quiet, yellow = loud).

**I/O convention for this test:**
- u(t): scalar input, shape (T,)
- h(t): scalar output = ‖h‖₂, shape (T,). Used in LP types A, B, C.
- Wx(t): scalar = ‖W·x‖₂, shape (T,). Used in LP type D (simulation only).
- Orthogonality: computed from full electrode vector H, shape (T, N_h).

Column D (State LP) is only available in simulation. For hardware DUTs, pass
`Wx_scalar=None` and that column will be omitted from the grid.
```

---

### Cell 5 — Section 2.1: Visual fingerprint sweep

```python
# ── 2.1 Visual Fingerprint ───────────────────────────────────────────
# Sine period = 1s (1 Hz), so FP_N_PERIODS seconds = FP_N_PERIODS periods.
FP_DURATION = float(FP_N_PERIODS)

sweep = sine_sweep(
    duration=FP_DURATION,
    amplitude=V_SAFE,
    n_steps=N_AMP_STEPS_FP,
    fs=FS
)

sweep_results = []
ortho_fp_list = []

for amp_linear, u_step in sweep:
    amp_dB = 20 * np.log10(amp_linear / V_SAFE)

    # Run for exactly FP_N_PERIODS — do NOT wait for settling.
    # Chaotic reservoirs may not settle, and the unsettled transient
    # behavior is informative for the operator to observe.
    dut.reset()
    H_step = dut.run(to_input_seq(u_step))   # shape (T, N_h) — full electrode vector

    h_scalar  = np.linalg.norm(H_step, axis=1)   # (T,) scalar for LP types A, B, C
    u_scalar  = u_step                            # (T,) scalar input

    # State LP (D): compute W @ x at each step — requires access to DUT internals
    # Run again capturing state to get Wx. In simulation: re-run collecting x(t).
    # For a clean implementation, add a `run_with_state()` method to LIESN that
    # returns both H and the full state trajectory X of shape (T, N_x).
    # Then Wx_scalar = np.linalg.norm(dut.W @ X.T, axis=0)  shape (T,)
    Wx_scalar = None  # Replace with above if run_with_state() is implemented

    ortho_fp_list.append(observational_orthogonality(H_step))

    sweep_results.append({
        'amplitude_dB':    amp_dB,
        'amplitude_linear': amp_linear,
        'u':               u_scalar,
        'h_scalar':        h_scalar,
        'Wx_scalar':       Wx_scalar,
    })

fig_fp = fingerprint_grid(sweep_results)
fig_fp.suptitle("2.1 — Visual Fingerprint: Lissajous Plot Series", fontsize=13)
plt.show()

ortho_fp_mean = float(np.mean(ortho_fp_list))
print(f"Observational orthogonality (mean across sweep): {ortho_fp_mean:.4f}")
print("(0 = redundant electrodes, 1 = maximally independent)")
```

---

### Cell 6 — Markdown: Section 2.3

```
## 2.3 Fading Memory Property (FMP) and Echo State Property (ESP)

**FMP test:** Each trial establishes a different initial condition using a distinct
multisine u_A^(i), then applies a near-zero DC probe u_B. If the outputs of all
trials converge to the same trajectory under u_B, the reservoir is forgetting
its initial conditions — demonstrating FMP.

**ESP test:** After settling with u_A^(i), the input is swapped to a new multisine
u_C (different frequencies and mean amplitude). If all trials converge to the same
output under u_C regardless of initial condition, ESP holds.

**I/O convention for this test:**
- u(t): scalar input, shape (T,)
- h(t): vector output H, shape (T, N_h). Plotted per-channel and as ‖h‖₂.
- Orthogonality: computed from full vector H.

N_TRIALS = 2 (set in config). Each trial uses a different random seed for u_A.
```

---

### Cell 7 — Section 2.3: FMP test

```python
# ── 2.3 FMP Test ─────────────────────────────────────────────────────
# Generate N_TRIALS distinct settling inputs u_A^(i) — different random phases

fmp_outputs = []   # list of h_scalar arrays, one per trial
fmp_H_vecs  = []   # list of H arrays (full vector), one per trial

u_B = dc_near_zero(duration=PROBE_DURATION, fs=FS)  # near-zero DC probe

for trial_i in range(N_TRIALS):
    # Distinct u_A: multisine with random phase offsets
    rng_trial = np.random.default_rng(trial_i + 100)
    phases = rng_trial.uniform(0, 2 * np.pi, 4)
    t = np.arange(int(SETTLE_DURATION * FS)) * DT
    u_A = V_SAFE * 0.5 * (
        np.sin(2*np.pi*1*t  + phases[0]) +
        np.sin(2*np.pi*3*t  + phases[1]) +
        np.sin(2*np.pi*7*t  + phases[2]) +
        np.sin(2*np.pi*11*t + phases[3])
    )

    # Settle to distinct initial condition
    dut.reset()
    dut.run(to_input_seq(u_A))   # drive to x_A^(i); discard output

    # Apply probe and record
    H_probe = dut.run(to_input_seq(u_B))   # shape (T_probe, N_h) — full electrode vector
    h_scalar_probe = np.linalg.norm(H_probe, axis=1)   # shape (T_probe,) scalar

    fmp_outputs.append(h_scalar_probe)
    fmp_H_vecs.append(H_probe)

# ── Plot FMP: all trials overlaid ────────────────────────────────────
t_probe = np.arange(len(u_B)) * DT
fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.tab10(np.linspace(0, 1, N_TRIALS))
for i, h_sc in enumerate(fmp_outputs):
    ax.plot(t_probe, h_sc, color=colors[i], lw=1.5, label=f"Trial {i}")

ax.set_xlabel("Time (s)")
ax.set_ylabel("‖h(t)‖₂  [scalar output]")
ax.set_title("2.3 — FMP Test: Output convergence under near-zero DC probe")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Orthogonality (use last trial's vector output as representative)
ortho_fmp = observational_orthogonality(fmp_H_vecs[-1])
print(f"Observational orthogonality (FMP trial): {ortho_fmp:.4f}")
print("FMP holds if all trial outputs converge to the same trajectory under u_B.")
```

---

### Cell 8 — Section 2.3: ESP test

```python
# ── 2.3 ESP Test ─────────────────────────────────────────────────────
# After settling with u_A^(i), swap to u_C and observe convergence.

# u_C: different from u_A — use different frequencies and mean amplitude
t_C = np.arange(int(ESP_SWAP_DURATION * FS)) * DT
u_C = V_SAFE * 0.4 * (
    np.sin(2*np.pi*2*t_C) +
    np.sin(2*np.pi*5*t_C) +
    np.sin(2*np.pi*9*t_C)
)

esp_outputs = []
esp_H_vecs  = []

for trial_i in range(N_TRIALS):
    rng_trial = np.random.default_rng(trial_i + 200)
    phases = rng_trial.uniform(0, 2 * np.pi, 4)
    t = np.arange(int(SETTLE_DURATION * FS)) * DT
    u_A = V_SAFE * 0.5 * (
        np.sin(2*np.pi*1*t  + phases[0]) +
        np.sin(2*np.pi*3*t  + phases[1]) +
        np.sin(2*np.pi*7*t  + phases[2]) +
        np.sin(2*np.pi*11*t + phases[3])
    )

    dut.reset()
    dut.run(to_input_seq(u_A))   # establish distinct initial condition

    H_esp = dut.run(to_input_seq(u_C))   # shape (T_esp, N_h) — full vector
    h_scalar_esp = np.linalg.norm(H_esp, axis=1)   # scalar output

    esp_outputs.append(h_scalar_esp)
    esp_H_vecs.append(H_esp)

# ── Plot ESP: all trials overlaid ────────────────────────────────────
t_esp = np.arange(len(u_C)) * DT
fig, ax = plt.subplots(figsize=(10, 4))
for i, h_sc in enumerate(esp_outputs):
    ax.plot(t_esp, h_sc, color=colors[i], lw=1.5, label=f"Trial {i}")

ax.set_xlabel("Time (s)")
ax.set_ylabel("‖h(t)‖₂  [scalar output]")
ax.set_title("2.3 — ESP Test: Output convergence under common input u_C")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

ortho_esp = observational_orthogonality(esp_H_vecs[-1])
print(f"Observational orthogonality (ESP trial): {ortho_esp:.4f}")
print("ESP holds if all trial outputs are visually indistinguishable under u_C.")
```

---

### Cell 9 — Markdown: Section 2.4

```
## 2.4 Separation Property (SP)

A Poisson spike train template u⁽⁰⁾ is generated. A variant u⁽¹⁾ is created by
delaying a single spike at t_spike by t_delay. Both are run through the DUT from
the same initial condition. The L2 norm of the difference between outputs is plotted
over time as the running mean d̄(t) and standard deviation σ(t).

SP holds when d̄(t) remains nonzero after the displaced spike with σ(t) small
relative to d̄(t) — indicating consistent, stable separation.

**I/O convention for this test:**
- u(t): scalar spike train input, shape (T,). Value is amplitude at spike times, 0 elsewhere.
- h(t): scalar output = ‖h‖₂ per trial, shape (T,).
- d(t) = |h⁽⁰⁾(t) − h⁽¹⁾(t)|  (scalar difference, shape (T,))
- Orthogonality: computed from full vector output H of one trial.

Both trials use identical initial conditions. The only difference is the delayed spike.
```

---

### Cell 10 — Section 2.4: Separation property test

```python
# ── 2.4 Separation Property Test ─────────────────────────────────────

# Generate template spike train u^(0)
u_template = poisson_spike_train(
    duration=SPIKE_DURATION,
    rate_hz=POISSON_RATE_HZ,
    amplitude=V_SAFE,
    pulse_width_samples=PULSE_WIDTH,
    fs=FS,
    seed=SEED_SPIKES
)

# Create variant u^(1) with one spike delayed
u_variant = delayed_spike_train(
    u_template=u_template,
    spike_idx=SPIKE_IDX,
    delay_samples=DELAY_SAMPLES
)

# ── Run both trials from IDENTICAL initial condition ──────────────────
# Warm up to a common state using a shared settling input, then run each variant.
u_settle_common = multisine(duration=SETTLE_DURATION, amplitude=V_SAFE * 0.5, fs=FS)

dut.reset()
dut.run(to_input_seq(u_settle_common))
# Save state after settling — both trials start from here
x_common = dut.x.copy()

# Trial 0: template
dut.reset(x0=x_common)
H0 = dut.run(to_input_seq(u_template))   # shape (T_spike, N_h) — full electrode vector
h0_scalar = np.linalg.norm(H0, axis=1)   # scalar, shape (T_spike,)

# Trial 1: variant (one spike delayed)
dut.reset(x0=x_common)
H1 = dut.run(to_input_seq(u_variant))   # shape (T_spike, N_h)
h1_scalar = np.linalg.norm(H1, axis=1)   # scalar

# ── Separation metric ─────────────────────────────────────────────────
# d(t) = scalar difference in output norms (not norm of difference vector — see note)
# Note: since h0_scalar and h1_scalar are already L2 norms, d(t) = |h0 - h1|
d = np.abs(h0_scalar - h1_scalar)   # shape (T_spike,) scalar

# Running mean and std (window = 1 second)
window = FS
d_mean = np.convolve(d, np.ones(window)/window, mode='same')
d_std  = np.array([
    np.std(d[max(0, i-window//2):min(len(d), i+window//2)])
    for i in range(len(d))
])

# ── Plot ──────────────────────────────────────────────────────────────
t_sp = np.arange(len(u_template)) * DT
t_spike_sec = SPIKE_IDX * DT

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Row 1: spike trains
axes[0].plot(t_sp, u_template, color='steelblue', lw=0.8, label='u⁽⁰⁾ template')
axes[0].plot(t_sp, u_variant + V_SAFE*0.05,  color='tomato',    lw=0.8,
             label='u⁽¹⁾ variant (offset for visibility)', alpha=0.8)
axes[0].axvline(t_spike_sec, color='k', ls='--', lw=1, label=f'Displaced spike (t={t_spike_sec:.1f}s)')
axes[0].set_ylabel("Input u(t)  [scalar]")
axes[0].set_title("2.4 — Separation Property Test")
axes[0].legend(fontsize=8)

# Row 2: outputs
axes[1].plot(t_sp, h0_scalar, color='steelblue', lw=1, label='‖h⁽⁰⁾‖₂')
axes[1].plot(t_sp, h1_scalar, color='tomato',    lw=1, label='‖h⁽¹⁾‖₂')
axes[1].axvline(t_spike_sec, color='k', ls='--', lw=1)
axes[1].set_ylabel("Output ‖h(t)‖₂  [scalar]")
axes[1].legend(fontsize=8)

# Row 3: separation d̄(t) and σ(t)
axes[2].plot(t_sp, d_mean, color='k',      lw=1.5, label='d̄(t) running mean')
axes[2].plot(t_sp, d_std,  color='tomato', lw=1.0, label='σ(t) running std', ls='--')
axes[2].axvline(t_spike_sec, color='k', ls='--', lw=1)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Separation d(t)")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Orthogonality (from trial 0 full vector output)
ortho_sp = observational_orthogonality(H0)
print(f"Observational orthogonality (SP test, trial 0): {ortho_sp:.4f}")

# Post-spike mean separation (mean d̄ after displaced spike time)
post_spike_mask = t_sp > t_spike_sec
d_post_mean = float(d_mean[post_spike_mask].mean())
d_post_std  = float(d_std[post_spike_mask].mean())
print(f"Post-spike mean separation d̄:  {d_post_mean:.4f}")
print(f"Post-spike mean σ(t):           {d_post_std:.4f}")
print(f"σ/d̄ ratio (lower is better):   {d_post_std/(d_post_mean+1e-12):.4f}")
print("SP holds when d̄ > 0 and σ/d̄ is small.")
```

---

### Cell 11 — Results summary and save

```python
# ── Section 2 Results Summary ────────────────────────────────────────

results_s2 = {
    "ortho_fingerprint_mean":   ortho_fp_mean,
    "ortho_fmp":                float(ortho_fmp),
    "ortho_esp":                float(ortho_esp),
    "ortho_sp":                 float(ortho_sp),
    "sp_post_spike_d_mean":     d_post_mean,
    "sp_post_spike_sigma_mean": d_post_std,
    "sp_sigma_over_d":          float(d_post_std / (d_post_mean + 1e-12)),
    "test_params": {
        "N_AMP_STEPS_FP":    N_AMP_STEPS_FP,
        "POISSON_RATE_HZ":   POISSON_RATE_HZ,
        "SPIKE_IDX":         SPIKE_IDX,
        "DELAY_SAMPLES":     DELAY_SAMPLES,
        "PULSE_WIDTH":       PULSE_WIDTH,
    }
}

with open(os.path.join(RESULTS_DIR, "section2_results.json"), "w") as f:
    json.dump(results_s2, f, indent=2)

print("=" * 50)
print("SECTION 2 RESULTS SUMMARY")
print("=" * 50)
print(f"  Ortho (fingerprint sweep): {ortho_fp_mean:.4f}")
print(f"  Ortho (FMP test):          {ortho_fmp:.4f}")
print(f"  Ortho (ESP test):          {ortho_esp:.4f}")
print(f"  Ortho (SP test):           {ortho_sp:.4f}")
print(f"  SP post-spike d̄:           {d_post_mean:.4f}")
print(f"  SP σ/d̄:                    {d_post_std/(d_post_mean+1e-12):.4f}")
print(f"\nResults saved to {RESULTS_DIR}section2_results.json")
print("Proceed to Section 3 for quantitative system characterization.")
```

---

## Implementation notes for Claude Code

- The `LIESN.reset(x0)` method must accept an `x0` parameter (ndarray of shape `(N_x,)`) so
  that both SP trials can start from the exact same saved state `x_common`.
- The `run_with_state()` method (mentioned in Cell 5 for column D of the fingerprint) should be
  added to `LIESN` if column D is desired. It returns `(H, X)` where X shape `(T, N_x)`.
  If not implemented, Wx_scalar remains None and column D is omitted from `fingerprint_grid`.
- The `fingerprint_grid` function must handle `Wx_scalar=None` gracefully — render only columns
  A, B, C in that case with a note "State LP: simulation only."
- For the SP test, note that d(t) = |‖h⁽⁰⁾‖₂ − ‖h⁽¹⁾‖₂| is used rather than ‖h⁽⁰⁾ − h⁽¹⁾‖₂.
  Both are valid scalar separation metrics; the former is simpler and matches the scalar I/O
  convention. Document this choice in a code comment.
- Generate as `.ipynb` using `nbformat`. Do not execute during generation.
