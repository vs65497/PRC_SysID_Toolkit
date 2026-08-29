# PRC Toolkit
### A Toolkit for System Identification of Black-box Physical Reservoir Computers

**Paper:** *A Toolkit for System Identification of Black-box Physical Reservoir Computers*<br>
**Available online** https://doi.org/10.5281/zenodo.22165842

**Author:** Von Simmons<br>
**Date:** August 2026<br>
**Contact:** [see paper]

---

## What this toolkit does

This toolkit provides a structured set of tests for determining whether a physical device —
found or manufactured — can function as a Physical Reservoir Computer (PRC). It works
exclusively from input-output measurements, making no assumptions about the device's
internal structure.

The toolkit proceeds in four phases:

1. **Fundamentals** — confirm the device is nonlinear and find a safe operating amplitude.
2. **System Identification** — qualitatively confirm fading memory, echo state, and separation properties.
3. **System Characterization** — quantify memory capacity, nonlinearity, and dynamical regime.
4. **Benchmarks** — measure task performance on a standard suite of reservoir computing benchmarks.

Results from each phase carry forward to the next via JSON files in the `results/` folder.

This software toolkit uses simulated DUT models (LI-ESN and Ag2S-NWN). Hardware adaptation
requires replacing DUT calls with measured I/O data; see implementation notes below.

---

## Requirements

```
python >= 3.9
numpy
scipy
matplotlib
jupyter
nbformat
```

Install:
```bash
pip install numpy scipy matplotlib jupyter nbformat
```

---

## Running the toolkit

Run the notebooks in order. Each notebook saves its key results to `results/` and the
next notebook loads them automatically.

```
01_fundamentals.ipynb          → results/section1_results.json
02_system_identification.ipynb → results/section2_results.json
03_system_characterization.ipynb → results/section3_results.json
04_benchmarks.ipynb            → results/section4_results.json
```

Start Jupyter:
```bash
jupyter notebook
```

Then open and run each notebook top-to-bottom.

---

## Hardware mode

Notebooks 02–04, plus Section 1.3 of notebook 01, support `DUT_MODEL = "hardware"`,
which loads pre-recorded output CSVs from `HARDWARE_DATA_PATH` instead of driving a
simulated DUT. Analysis code is completely unchanged — it consumes the same `H` array
regardless of source. Sections 1.1 and 1.2 are demonstration-only and raise a clear
error if run with `DUT_MODEL = "hardware"`.

**Demo dataset (default):** `HARDWARE_DATA_PATH` defaults to `data/demo/hardware_outputs/`,
a committed set of pre-generated outputs (captured from an LI-ESN simulation run,
standing in as "presumptive hardware") so `DUT_MODEL = "hardware"` works out of the
box with no physical DUT attached — useful for demoing the toolkit or verifying the
hardware-mode plumbing itself.

**Real hardware:**
1. Run `00_hardware_signals.ipynb` once. It generates every input sequence needed for
   notebooks 01–04 into `data/hardware/inputs/`, creates an empty `data/hardware/outputs/`,
   and prints step-by-step recording instructions for each file.
2. Play each input into your DUT and save the recorded output CSVs into
   `data/hardware/outputs/` using the exact filenames printed by the notebook.
3. In notebook 01 (Section 1.3 only) and each of notebooks 02–04, set
   `DUT_MODEL = "hardware"` and change `HARDWARE_DATA_PATH = "data/hardware/outputs"`
   (the real recording target — distinct from the demo path above).

**Regenerating the demo dataset:** if the toolkit's parameters change, refresh
`data/demo/hardware_outputs/` by setting `SAVE_DEMO_HARDWARE_DATA = True` in notebook 01
and each of notebooks 02–04, running them once with `DUT_MODEL = "liesn"`, then setting
the flag back to `False`.

**Two deviations from `prompts/00g_hardware_mode.md`,** found by cross-checking the
prompt against the actual notebook code before implementing:
- Notebook 01 is not wired for hardware mode (per the prompt's own explicit "no code
  changes to notebook 01" instruction). `00_hardware_signals.ipynb` still generates
  `section1_multisine.csv` for manual/off-line reference, but it is not auto-loaded.
- Notebook 03 has exactly one data-collection point (not three, as the prompt assumed) —
  a single cell produces both trials (`H0`, `H1`) reused across Sections 3.2–3.6. Hardware
  mode there uses two files, `section3_shared_run_trial00.csv` / `trial01.csv`.

---

## Test summary (Table of Contents)

| Phase | Test | Notebook | Section |
|-------|------|----------|---------|
| Fundamentals | Disqualifying Simple Devices | 01 | 1.1 |
| Fundamentals | Safe Region of Operation | 01 | 1.2 |
| Fundamentals | Distortion Analysis | 01 | 1.3.1 |
| Fundamentals | Noise Floor Analysis | 01 | 1.3.2 |
| System Identification | Visual Fingerprint (Lissajous Series) | 02 | 2.1 |
| System Identification | Fading Memory & Echo State Properties | 02 | 2.3 |
| System Identification | Separation Property | 02 | 2.4 |
| System Characterization | Consistency | 03 | 3.2 |
| System Characterization | Conditional Lyapunov Exponent | 03 | 3.3.1 |
| System Characterization | Power Spectral Density vs 1/f | 03 | 3.3.2 |
| System Characterization | Linear Memory Capacity | 03 | 3.4 |
| System Characterization | Information Processing Capacity | 03 | 3.5 |
| System Characterization | Measure of Nonlinearity | 03 | 3.6 |
| Benchmarks | NARMA-10 | 04 | 4.1 |
| Benchmarks | Mackey-Glass System | 04 | 4.2 |
| Benchmarks | Lorenz'63 Attractor | 04 | 4.3 |
| Benchmarks | Sunspot Numbers | 04 | 4.4 |
| Benchmarks | XOR Task | 04 | 4.5 |

MNIST and Double Pole Balancing are deferred (hardware I/O constraints) and not
implemented in this release.

---

## What the results mean: a plain-language guide

### After Notebook 1 — Fundamentals

**V_safe** is the maximum safe input amplitude. All subsequent tests use inputs scaled
to this value. If V_safe is very small (close to zero), the device saturates or distorts
at very low amplitudes — it may be too fragile, or the electrode configuration may need
revision.

**Distortion products** (H_odd, H_even) appearing in the output confirm the device is
nonlinear — a necessary condition for reservoir computing. If only H_linear is present,
the device is operating as a linear filter, which is not sufficient.

**Noise floor** shows how much of the output varies randomly from period to period. Some
noise is expected and can actually be beneficial (stochastic resonance). If noise dominates
the signal, characterization in Section 3 will be unreliable.

*Decision after Notebook 1:*
- V_safe > 0, distortion products present, noise floor below signal level → proceed to Notebook 2.
- V_safe ≈ 0 → device saturates immediately; reconsider electrode placement or input scaling.
- No distortion products → device may be linear; continue to Notebook 2 to confirm, but
  expect low IPC and nonlinearity scores in Notebook 3.
- Very high noise floor → noise swamps the signal; check connections and shielding before continuing.

---

### After Notebook 2 — System Identification

**Visual fingerprint** gives a qualitative picture of the device's dynamical character.
Ordered dynamics show clean, nested trajectories. Chaotic dynamics show dense, space-filling
trajectories. Critical dynamics fall between. No single appearance is universally best —
this depends on your application.

**FMP/ESP test**: if all trial outputs converge to the same trajectory under the near-zero
probe, fading memory holds. If outputs remain persistently different across trials,
the device may not be forgetting its initial conditions — a sign the ESP is violated
at this input amplitude.

**Separation property**: a sustained nonzero d̄(t) after the displaced spike confirms the
device can distinguish similar inputs over time. If d̄(t) decays back to zero quickly, the
device is too ordered and will not separate distinct inputs reliably.

**Observational orthogonality** (reported for each test) measures how independently the
output electrodes are sampling the reservoir. Low orthogonality means the electrodes are
reading redundant information; repositioning electrodes or adding more may help.

*Decision after Notebook 2:*
- FMP/ESP converging, SP sustained, orthogonality moderate-to-high → proceed to Notebook 3.
- FMP/ESP not converging → reservoir may be unstable at this amplitude; try reducing input
  to V_safe × 0.5 and re-running.
- SP decays to zero → reservoir is too ordered; if the DUT parameters are adjustable,
  try increasing spectral radius (for LI-ESN) or connectivity (for Ag2S-NWN).
- SP diverges erratically (σ/d̄ large) → reservoir is too chaotic; reduce spectral radius.
- Orthogonality near zero → electrode configuration is poor; rethink placement.

---

### After Notebook 3 — System Characterization

**Consistency γ²**: close to 1.0 means the device produces repeatable responses from
different initial conditions — a quantitative confirmation of ESP. Values below 0.5
indicate significant variability; the readout will struggle to learn stable mappings.

**CLE λ̂**:
- λ̂ < 0 (high R²): ordered dynamics — good stability, limited memory richness.
- λ̂ ≈ 0 (high R²): critical / edge-of-chaos — best balance of stability and complexity.
- λ̂ > 0 (high R²): chaotic — high separation but poor consistency.
- Low R²: dynamics are not well-described by a simple exponential; the device may have
  mixed or non-stationary behavior. Interpret other metrics more heavily.

**PSD exponent β̂**: β̂ ≈ 1 is associated with critical dynamics; β̂ ≈ 2 with
random-walk/chaotic behavior. Treat this as supporting evidence alongside λ̂, not
as a standalone diagnostic.

**Linear MC**: total memory capacity. Higher is better for tasks requiring long-term memory.
A sharp drop at small k indicates limited short-term memory.

**IPC total**: the IPC at d=1 should match linear MC. The additional capacity at d≥2
reflects nonlinear memory — the reservoir's ability to compute nonlinear functions of
past inputs. IPC > MC indicates meaningful nonlinear processing.

**Nonlinearity NL_k**: NL_k = 1 − MC_k, reported per delay k (not a single number) —
how nonlinear the device's recall of u(t−k) is at each delay. NL_k close to 0 means
near-linear recall at that delay; close to 1 means strongly nonlinear. NL at k=1 is
the traditionally-cited single value (nonlinearity of the immediate past).
A device with high NL_k but low IPC is interesting — it is nonlinear but the
nonlinearity is not productively separating information.

*Interpreting the full picture:*
A good PRC candidate shows: high γ² (consistent), λ̂ near 0 (critical), moderate-to-high
MC (memory), IPC > MC (nonlinear processing), and sustained SP from Notebook 2.
No single metric is decisive. The toolkit is designed to give a complete qualitative
and quantitative picture, not a single pass/fail score.

---

### After Notebook 4 — Benchmarks

Notebook 4 reports task performance rather than diagnostic metrics: it answers "how well
does this device actually compute" rather than "what kind of dynamical system is this."
Lower NRMSE is better for the four regression tasks (NARMA-10, Mackey-Glass, Lorenz'63,
Sunspot Numbers); higher accuracy is better for XOR.

**NARMA-10** and **XOR** are the most memory- and nonlinearity-demanding of the set —
weak scores here alongside strong Notebook 3 metrics (high MC, high IPC) suggest the
DUT's raw capacity isn't being captured well by the readout, not that the capacity is
absent. **Mackey-Glass** and **Sunspot** are one-step-ahead prediction tasks and tend to
score well whenever fading memory (Notebook 2) holds. **Lorenz** is the hardest task
(reconstructing a 3D attractor from a single scalar observable) — some error is expected
even for a strong reservoir.

Absolute NRMSE/accuracy values depend heavily on the DUT's specific configuration (this
notebook reuses whatever DUT_PARAMS was established in Notebook 1, not a benchmark-tuned
configuration), so these numbers are most useful for comparing configurations or DUTs
against each other, not as a standalone pass/fail score.

---

## Known software limitations

**IPC computation scales poorly with K_MAX and IPC_MAX_DEGREE.** The number of basis
functions grows as C(K_MAX + d, d). With K_MAX=50 and degree=3, this is ~23,000 basis
functions, requiring a very large matrix solve. Default is K_MAX=10, degree=2 (65 basis
functions). Do not increase K_MAX beyond 20 without checking memory and runtime first.

**Orthogonality requires N_h ≥ 2.** If the DUT has only one output electrode, the
pairwise orthogonality metric is undefined and returns NaN. This is expected behavior.

**Settling may not converge.** For chaotic reservoirs, `run_until_settled()` may reach
its sample limit without converging. A warning is printed; the subsequent test proceeds
from the unsettled state. For the visual fingerprint test, settling is intentionally
skipped — the unsettled transient is informative.

**CLE exponential fit may be poor (low R²).** This is expected for reservoirs with mixed
or non-stationary dynamics. Low R² is flagged with a warning; λ̂ should be interpreted
cautiously in this case.

**Column D of the visual fingerprint (State LP) is simulation only.** It requires access
to the hidden reservoir state x(t), which is unavailable for hardware DUTs. The notebook
renders columns A, B, C only when Wx_scalar is None.

**Hardware reset is not instantaneous.** The `reset()` method works perfectly for
simulated DUTs. For physical hardware, forcing the device to a reproducible rest state
via a periodic entrainment signal is recommended (see SM.1 of the paper). The current
toolkit does not implement hardware reset procedures.

**Input convention: scalar only for Sections 1–3.** All tests in Sections 1–3 use a
single input electrode (N_u=1). The DUT interface supports multiple inputs (N_u>1) and
this will be used in the benchmarking section (Section 4). For now, all signal generators
produce shape (T,) scalars that are reshaped to (T,1) via `to_input_seq()` before
passing to the DUT.

**Ag2S-NWN performance.** The simultaneous voltage solve runs at every timestep and
scales as O(N_wires³). At N_wires=20, simulation of 30s at 100 Hz takes on the order
of seconds. Beyond 20 wires, expect significant slowdowns.

**Bidirectional inputs for Ag2S-NWN.** Signal generators produce bipolar signals by
default. When using the Ag2S-NWN, wrap input signals with `bias_positive()` to map
to non-negative voltages, consistent with unidirectional current flow in the physical
substrate.

---

## Results file schema

All results are JSON files in `results/`.

**`section1_results.json`**
```
V_SAFE                        float  Safe operating amplitude
H_linear_power                float  Spectral power at input frequencies
H_odd_power                   float  Spectral power at odd intermod. products
H_even_power                  float  Spectral power at even intermod. products
total_spectral_power          float  Total output spectral power
noise_floor_peak_sigma2_H     float  Peak noise variance across frequency bins
DUT_params                    dict   DUT configuration (passed to Notebooks 2 and 3)
```

**`section2_results.json`**
```
ortho_fingerprint_mean        float  Mean orthogonality across fingerprint sweep
ortho_fmp / ortho_esp         float  Orthogonality for FMP and ESP tests
ortho_sp                      float  Orthogonality for SP test
sp_post_spike_d_mean          float  Mean separation after displaced spike
sp_post_spike_sigma_mean      float  Mean σ(t) after displaced spike
sp_sigma_over_d               float  σ/d̄ ratio (lower = more stable separation)
test_params                   dict   Test configuration
```

**`section3_results.json`**
```
gamma2_global                 float  Global consistency γ²
gamma2_per_electrode          list   Per-electrode γ²_i
ortho_shared_run              float  Orthogonality from shared trial run
ortho_mc                      float  Orthogonality during MC computation
lambda_hat                    float  CLE estimate λ̂
R2_cle                        float  Goodness of fit for CLE exponential model
regime_estimate               str    "ordered" / "critical/edge" / "chaotic"
beta_hat                      float  PSD power law exponent β̂
R2_psd                        float  Goodness of fit for PSD power law
MC_total                      float  Total linear memory capacity
MC_k                          list   Memory capacity per delay k
IPC_total                     float  Total information processing capacity
IPC_by_degree                 dict   IPC contribution per polynomial degree
NL_k                          list   Measure of nonlinearity per delay k (1 − MC_k)
test_params                   dict   Test configuration
```

**`section4_results.json`**
```
nrmse_narma10                 float  NARMA-10 imitation NRMSE
nrmse_mackey_glass            float  Mackey-Glass one-step-ahead prediction NRMSE
nrmse_lorenz                  float  Lorenz'63 mean NRMSE across x, y, z
nrmse_sunspot                 float  Sunspot number one-step-ahead prediction NRMSE
xor_accuracy                  float  XOR task classification accuracy
xor_lag                       int    Lag d used for the XOR task
mg_horizon                    int    Prediction horizon (steps) used for Mackey-Glass
test_params                   dict   Test configuration
```
