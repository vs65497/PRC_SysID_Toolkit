# Prompt — Notebook Fixes (Post-Review Pass)

## Context

The PRC toolkit notebooks have been reviewed against real output. This prompt
describes a focused set of fixes to apply. Do NOT restructure notebooks, add
new imports beyond what is specified, or change anything not listed here.
Make the minimum change needed for each item.

Read `build_notes.md` before starting. Run `python -m pytest tests/ -v` after
all changes and confirm all tests still pass before finishing.

---

## Fix 1 — Section 1.1: Automated device disqualification checks

**File:** `01_fundamentals.ipynb`, the existing Section 1.1 code cell.

After the existing DC ramp plot, add automated fit-based checks for wire,
resistor, capacitor, and inductor. Do NOT add new plots. Output is print
statements only. The diode check remains visual-only (existing print statement,
do not modify it).

Use the DC ramp data (`u_ramp`, `h_ramp_scalar`) already computed above for
wire and resistor checks. Use the multisine FFT data from Section 1.3.1 for
capacitor and inductor checks — but those won't exist yet when 1.1 runs. So
for capacitor and inductor: add a note in a print statement that these checks
require running Section 1.3.1 first, and implement them at the END of the
Section 1.3.1 code cell instead (after H_fft is computed).

### Wire check (in 1.1 cell, after plot)
Fit a line through (u_ramp, h_ramp_scalar) via np.polyfit(u_ramp, h_ramp_scalar, 1).
Compute R² of that fit.
```
slope, intercept = np.polyfit(u_ramp, h_ramp_scalar, 1)
h_fit = slope * u_ramp + intercept
ss_res = np.sum((h_ramp_scalar - h_fit)**2)
ss_tot = np.sum((h_ramp_scalar - h_ramp_scalar.mean())**2)
r2_linear = 1 - ss_res / (ss_tot + 1e-12)
```
Print results:
- If r2_linear > 0.99 AND abs(slope - 1.0) < 0.05 AND abs(intercept) < 0.05:
  "1.1 Wire check: LIKELY A WIRE (R²={r2_linear:.4f}, slope≈1, intercept≈0). Do not proceed."
- Elif r2_linear > 0.99:
  "1.1 Resistor check: LIKELY A RESISTOR (R²={r2_linear:.4f}, slope={slope:.3f}). Do not proceed."
- Else:
  "1.1 Linear device check: Not a wire or resistor (R²={r2_linear:.4f}). Proceeding."

### Capacitor and inductor checks (add to END of Section 1.3.1 code cell)
After H_fft and bins_lin are computed. Use the linear frequency bins only.

```python
# ── 1.1 Capacitor / Inductor check (requires 1.3.1 FFT data) ─────────
# For a capacitor: |H(f)| / |U(f)| ∝ f  (impedance 1/2πfC, so output ∝ f)
# For an inductor: |H(f)| / |U(f)| ∝ 1/f
# Test using the 4 linear frequency bins (f=1,3,7,11 Hz).
# Input multisine has equal amplitude at each frequency by construction,
# so |U(f)| is approximately constant and we just check |H(f)| vs f.

f_vals  = np.array([1.0, 3.0, 7.0, 11.0])
h_at_f  = H_fft[bins_lin]   # |H| at linear frequencies

# Fit log|H| vs log(f): slope≈+1 → capacitor, slope≈-1 → inductor
log_f = np.log(f_vals)
log_h = np.log(h_at_f + 1e-30)
slope_cap, _ = np.polyfit(log_f, log_h, 1)

if abs(slope_cap - 1.0) < 0.3:
    print(f"1.1 Capacitor check: LIKELY A CAPACITOR (log-log slope={slope_cap:.3f}≈+1). Do not proceed.")
elif abs(slope_cap + 1.0) < 0.3:
    print(f"1.1 Inductor check: LIKELY AN INDUCTOR (log-log slope={slope_cap:.3f}≈-1). Do not proceed.")
else:
    print(f"1.1 Capacitor/Inductor check: Not a capacitor or inductor (log-log slope={slope_cap:.3f}). Proceeding.")
```

---

## Fix 2 — Mean subtraction before FFT in Sections 1.3.1 and 1.3.2

**File:** `01_fundamentals.ipynb`

### Section 1.3.1
Find the line that computes `h_period` (the last period of h_ms used for FFT).
Immediately after that line, add mean subtraction:
```python
h_period = h_period - h_period.mean()   # remove DC offset before FFT
# Note: ‖h‖₂ is always non-negative, so h_scalar has a large positive mean.
# Mean subtraction isolates the AC dynamics for spectral analysis.
```

### Section 1.3.2
Find the per-period FFT loop. Inside the loop, before computing the FFT of
each period of h_ms, subtract the mean of that period:
```python
h_period_slice = h_ms[start:end]
h_period_slice = h_period_slice - h_period_slice.mean()   # remove DC per period
H_periods.append(np.fft.rfft(h_period_slice) / T_period)
```
Also do the same for u_ms periods (for consistency, subtract per-period mean
of u_ms slice before FFT). u_ms is already zero-mean by design but this makes
the treatment symmetric.

---

## Fix 3 — SP test: trim convolution edge artifact from plot

**File:** `02_system_identification.ipynb`, Section 2.4 code cell.

Find where d_mean and d_std are plotted against t_sp. Trim the last `window`
samples from all three arrays before plotting to remove the convolution
boundary artifact that makes d_mean collapse at the end of the time series:

```python
trim = window   # same window used for running mean
t_plot   = t_sp[:-trim]
d_plot   = d[:-trim]
dm_plot  = d_mean[:-trim]
ds_plot  = d_std[:-trim]
```

Then replace all references to `t_sp`, `d`, `d_mean`, `d_std` in the plot
calls (axes[2]) with `t_plot`, `d_plot`, `dm_plot`, `ds_plot`.
Also trim the same range from axes[0] and axes[1] for visual consistency —
replace their `t_sp` with `t_plot` and slice u_template, u_variant,
h0_scalar, h1_scalar to `[:-trim]`.

Also update the post-spike mask to use t_plot:
```python
post_spike_mask = t_plot > t_spike_sec
d_post_mean = float(dm_plot[post_spike_mask].mean())
d_post_std  = float(ds_plot[post_spike_mask].mean())
```

---

## Fix 4 — FMP/ESP: update markdown cells to clarify FMP vs ESP distinction

**File:** `02_system_identification.ipynb`

### Markdown cell above the FMP test cell
Replace the existing markdown content for Section 2.3 with:

```markdown
## 2.3 Fading Memory Property (FMP) and Echo State Property (ESP)

Two related but distinct properties are tested here with two separate experiments.

**FMP experiment (Cell below):** Each trial establishes a different initial
condition by driving the DUT with a distinct multisine u_A^(i). The input is
then replaced with a near-zero DC probe u_B (amplitude ≈ 1% of V_safe). If
outputs from all trials converge toward the same trajectory under this
near-zero input, the reservoir is *forgetting its initial conditions on its
own* — this is the Fading Memory Property. The near-zero input is chosen
deliberately so that the convergence is driven by the reservoir's intrinsic
decay, not by a common forcing signal.

**ESP experiment (Next cell):** After settling to a distinct initial condition
with u_A^(i), the input is swapped to a common multisine u_C. If all trials
converge to the same output under u_C regardless of initial condition, the
reservoir is being *driven to a common state by the input* — this is the Echo
State Property. The distinction from FMP is that here, convergence is
input-driven rather than intrinsic.

Both experiments run N_TRIALS=2 trials.

**I/O convention:** u(t) scalar, shape (T,). H(t) vector shape (T, N_h),
plotted as ‖h‖₂ and per-electrode. Orthogonality computed from full vector H.
```

### Markdown cell above the ESP test cell
Replace its existing content with:

```markdown
### ESP Experiment

The same DUT is driven to distinct initial conditions with u_A^(i), then
switched to a common input u_C (different frequencies and mean amplitude from
u_A). Convergence under u_C demonstrates ESP: the input, not the initial
condition, determines the output trajectory. Compare with the FMP experiment
above — there, convergence occurred under near-zero input (intrinsic decay);
here, convergence is actively driven by the shared input signal.
```

---

## Fix 5 — XOR: add markdown note explaining ideal output

**File:** `04_benchmarks.ipynb`

Find the markdown cell for Section 4.5 (XOR Task). At the end of that
markdown cell, append:

```markdown
**What good XOR performance looks like:** In the ideal case the predicted
signal (dashed) exactly overlaps the target (solid) in the test plot below —
a binary square wave at the XOR lag. Accuracy near 1.0 indicates the
reservoir successfully computed the nonlinear combination u(t) XOR u(t−d).
Accuracy near 0.5 (chance) means the readout cannot separate the two classes,
typically because the reservoir lacks sufficient nonlinear memory at delay d.
XOR performance is sensitive to dynamical regime: critical and mildly chaotic
reservoirs generally outperform ordered ones on this task, since XOR requires
both memory (to retain u(t−d)) and nonlinearity (to compute the XOR function).
With only N_h output electrodes driving a linear classifier, performance also
degrades when N_h is small — increasing N_h or the spectral radius may
substantially improve accuracy.
```

---

## Fix 6 — Section 2.3 markdown: remove N_TRIALS config mention

**File:** `02_system_identification.ipynb`

In the updated markdown for Section 2.3 (written in Fix 4 above), the text
already says "N_TRIALS=2 trials" as a hardcoded statement. Confirm that no
cell in notebook 02 imports or references N_TRIALS from config. If it does,
replace that import/reference with the literal value 2 and a comment:
```python
N_TRIALS = 2   # hardcoded — the FMP/ESP test is designed for exactly 2 trials
```

---

## Verification

After all changes:
1. Run `python -m pytest tests/ -v` — all tests must pass.
2. Run `nbformat.validate()` on all four notebooks — must pass.
3. Do NOT execute the notebooks. Write only.
4. Do NOT change any imports, results cells, or anything not listed above.
