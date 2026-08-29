# Build Notes

## Ag2S-NWN (prompts/00b_ag2s_nwn.md, fixed by prompts/00c_ag2s_nwn_ground_fix.md) — floating-ground bug, now resolved

**Status: RESOLVED.** `prompts/00c_ag2s_nwn_ground_fix.md` fixed the floating-ground bug described
below by reserving node `N_in` as an explicit 0V ground reference electrode in the voltage solve
(`self.ground_node = N_in`, included in `known_nodes`/`known_voltages` alongside the driven input
nodes in `step()`'s Step 3). This gives the reduced Laplacian a genuine reference distinct from the
input, for all `N_in >= 1`.

Verified per the prompt's own checklist (`N_wires=20, N_in=1, N_out=3, connectivity=0.3, seed=42`):
1. `dut.V` now shows a genuine gradient across nodes after `step()` (e.g. values from 0.0 to 0.5
   under a 0.5V input), not the uniform "every node = input voltage" collapse.
2. `dut.V[dut.ground_node]` is exactly `0.0` after every `step()` and after `reset()`.
3. After 100 steps of sinusoidal input, `dut.L` shows a real mix of values across active junctions
   (~15% ON, ~12% OFF, the rest at intermediate growth — 35 distinct values), instead of staying at
   all-zero.

The node-count assertion was tightened from `N_in + N_out <= N_wires` to `N_in + N_out < N_wires` to
guarantee an interior node is always available for ground. This is a stricter constraint than before;
existing configurations in this toolkit (`N_wires=20, N_in=1, N_out=3` in the notebooks;
`N_wires=10, N_in=1, N_out=2` in the smoke test) all satisfy it with room to spare.

Regression tests for all three checks above were added to `tests/test_ag2s_nwn.py` (the original
`test_ag2s_nwn_shapes` smoke test was left unmodified, per the fix prompt's instructions, and still
passes).

---

## Trial: Ag2S-NWN as the active DUT in Notebooks 01–03 (post ground-fix)

**Update — both issues below are now RESOLVED** by `prompts/00d_readout_bias.md` (corrected; see
its own section below) and `prompts/00e_sp_test_washout.md`. Re-running this trial against the
fixed notebooks confirms both directly:

- **Notebook 02 / SP test**: no `AttributeError` — the washout-based initialization
  (`dut.reset()` + `run_until_settled()` + `dut.step()`, all part of the generic `BaseDUT`
  interface) works unmodified against `Ag2SNWN`. No DUT-specific workaround needed at all,
  unlike the earlier `isinstance(dut, Ag2SNWN)` patch described below.
- **Notebook 03 / MC**: `MC_total` went from **-60.4** to **+0.046** (small positive; every
  `MC_k` is now a small positive number, not ≈-1.2). `IPC_total=22.1` with `NL=0.996` —
  interesting in its own right: Ag2S-NWN shows almost no *linear* memory but substantial
  nonlinear processing capacity, consistent with its threshold-switching physics.

CLE (`R2_cle=0.0013`) and PSD (`beta_hat=-0.21`) are unaffected by either fix (they don't use the
trained readout) and remain exactly the `growth_rate`/`V_THRESH` tuning question already noted in
the Status section below — a separate, lower-priority physics follow-up, not a code bug.

The original trial write-up (pre-fix) is kept below for the record.

After the ground-node fix above, Ag2S-NWN was trialed as the active DUT in patched copies of all
three notebooks (`Ag2SNWN(N_wires=20, N_in=1, N_out=3, connectivity=0.3, sigma_process=0.01,
sigma_measure=0.005, growth_rate=1e-10, seed=42)`, every bipolar signal wrapped in
`bias_positive()`). These trial copies are **not** committed — only findings are recorded here. All
three executed end-to-end with zero cell errors, a major improvement over the pre-fix state where
the device produced no dynamics at all. Two real issues were found in the process, both about the
notebooks'/library's implicit LI-ESN-shaped assumptions rather than the ground-node fix itself.

### Notebook 01 (Fundamentals): works cleanly

Distortion analysis shows genuine nonlinear content (H_linear 14.8%, H_odd 0.1%, H_even 68.1% of
spectral power — the odd term confirms real nonlinearity, not just DC/even artifacts of the ‖·‖₂
reduction). Noise floor is non-degenerate. No issues.

### Notebook 02 (System Identification): runs cleanly, but exposed a cross-DUT interface gap

The Separation Property test (Section 2.4) saves a common initial condition with `dut.x.copy()`
and restores it via `dut.reset(x0=x_common)` — this is LI-ESN-specific: `x` is its reservoir state
vector. Ag2S-NWN has no `x` attribute; its analogous state lives in `L`/`G`/`V`/`V_prev`, and
`Ag2SNWN.reset(x0=...)` explicitly ignores `x0` (by design, per prompt 00b). Calling the notebook
as originally written against `Ag2SNWN` throws `AttributeError: 'Ag2SNWN' object has no attribute
'x'`.

Worked around in the trial copy only, by branching on `isinstance(dut, Ag2SNWN)` and saving/restoring
`(L, G, V, V_prev)` directly rather than going through `reset(x0=...)`. With that workaround, the
test runs and produces non-degenerate orthogonality values (0.06–0.61), but the separation ratio
σ/d̄ = 4.26 is much worse than LI-ESN's 0.72 — separation is present but noisy/unstable relative to
its magnitude at these default `growth_rate`/`V_THRESH` values.

This points to a real, general gap: nothing in `BaseDUT` provides a generic "save/restore state"
operation, so any test that needs a shared initial condition across trials (as SP does) is
currently written against LI-ESN's specific `x` attribute. A generic `get_state()`/`set_state()` on
`BaseDUT` (or teaching `Ag2SNWN.reset(x0=...)` to accept its own state tuple) would fix this
properly for any DUT — not attempted here since it's a design decision beyond the scope of the
ground-node fix.

### Notebook 03 (System Characterization): runs, but produces nonsensical negative memory capacity

`MC_total = -60.4`, with every `MC_k` around -1.2 (R² this negative on **in-sample** data — the
same data used to fit the readout — should not normally happen for a plain least-squares fit).
Diagnosed directly: `train_readout_ridge` fits `H @ W ≈ Z` with **no intercept/bias column**. This
never mattered for LI-ESN, whose i.i.d. uniform drive and reservoir outputs are zero-mean, so a
zero-intercept linear combination can already center itself near the target's mean. Ag2S-NWN
*requires* `bias_positive()`-shifted inputs (`u_drive` here has mean ≈0.41V, not 0), and its 3
output electrodes don't happen to span that offset either — confirmed by hand: without a bias
column, `H0 @ W` for delay k=1 has a predicted mean of 0.239 against a true target mean of 0.406,
giving R²=-1.28 regardless of ridge `alpha` (tried 1e-4, 1e-2, 1.0 — identical result, ruling out
regularization strength as the cause). Adding an explicit ones-column to `H0` before fitting
recovers a sane R²≈0.003 at k=1 — still very weak memory, but no longer catastrophically negative.

Two separate takeaways:
1. **The readout has a latent bug for any non-zero-mean signal**, which `train_readout_ridge` /
   `train_readout_ols` don't currently guard against or document. This affects any future DUT (or
   LI-ESN configuration) driven by a non-zero-mean signal, not just this trial. Not fixed here,
   since adding a bias term is a shared-library change affecting the already-shipped LI-ESN
   pipeline and its committed results — a decision for the user, not something to slip in
   silently.
2. Even accounting for that, actual memory capacity at `growth_rate=1e-10` appears genuinely weak:
   the bridge dynamics (SM.2's ion-migration timescale) may simply be much slower than a 100Hz
   i.i.d.-noise drive changes, so there's little encoded memory of recent input to recover.
   `gamma2_global=0.34` (moderate-low consistency), CLE `R2_cle=0.0013` (the exponential
   divergence/convergence model does not describe trajectory distance at all — h0/h1 don't
   converge or diverge cleanly), and PSD `beta_hat=-0.21` with `R2_psd=0.16` (a poor, unusual fit —
   negative β is not physically typical) are all consistent with a substrate whose electrical
   response is dominated by fast resistive-network behavior rather than the slow bridge memory at
   this drive rate. `growth_rate`, `V_THRESH`, and/or the drive's timescale likely need deliberate
   retuning for Ag2S-NWN to act as a productive PRC substrate — this is exactly the manual tuning
   the original prompt (00b) flagged as expected ("← TUNE THIS") but was moot before the ground
   fix, since no bridge dynamics activated at all until now.

### Status (as originally filed, prior to the 00d/00e fixes)

Ground-node fix is confirmed working and is a clear improvement (dynamics now activate, Notebooks
01–02 already produce sane results). Ag2S-NWN is **not yet** switched on as the active DUT in the
committed notebooks — two follow-up decisions are needed first: (1) whether/how to fix the missing
readout bias term in `prc_toolkit/analysis/readout.py`, and (2) whether/how to give `BaseDUT` a
generic state save/restore so Section 2.4 works against any DUT, not just LI-ESN. Growth-rate/
threshold retuning for Ag2S-NWN is a separate, lower-priority follow-up on top of those two.

**Both decisions above are now resolved** — see "Update" note at the top of this section. Ag2S-NWN
is still not switched on as the active DUT in the committed notebooks (that remains a separate
decision from fixing these bugs), but nothing code-level is blocking it anymore. The only
remaining open item for Ag2S-NWN specifically is `growth_rate`/`V_THRESH` retuning, which is a
physics/parameter question, not a bug.

---

## Fix: readout bias term (prompts/00d_readout_bias.md)

**Status: RESOLVED, with a correction.** The prompt's literal fix (append a bias column while
fitting, then strip and discard it before returning, keeping `H @ W_ext` unchanged at prediction
time) does not work — a fitted intercept only corrects predictions if it survives to prediction
time. Confirmed by running the prompt's own verification script against its own literal
implementation: it asserts R² > 0.99 but actually produces R² = -14.7.

Implemented the corrected version instead (confirmed with the user): `train_readout_ols` and
`train_readout_ridge` now return `W_ext` of shape `(N_h + 1, K)`, with the fitted intercept as the
last row, and every caller appends a matching bias column to `H` before predicting
(`H_bias = np.hstack([H, np.ones((T, 1))]); pred = H_bias @ W_ext`). Updated the two call sites in
`03_system_characterization.ipynb` (MC in 3.4, IPC in 3.5) and `tests/test_analysis.py`'s
assertions accordingly. `train_readout_classify` needed no direct changes (it calls
`train_readout_ols` internally, as the prompt anticipated), but its callers face the same
bias-column requirement.

Verified: full test suite passes (38/38, including a new permanent regression test for the
nonzero-mean case), and a fresh end-to-end run of Notebook 03 with LI-ESN stays numerically
consistent with the pre-fix baseline (MC_total 1.77 vs 1.69, IPC_total 32.3 vs 29.6, NL 0.73 vs
0.73 — small differences from the intercept now correctly absorbing LI-ESN's tiny nonzero-mean
residuals).

---

## Fix: SP test hardware-compatible washout (prompts/00e_sp_test_washout.md)

**Status: RESOLVED.** Section 2.4's Separation Property test previously saved a common initial
condition via `dut.x.copy()` and restored it via `dut.reset(x0=x_common)` — a simulation-only
shortcut with no hardware equivalent, and one that broke for Ag2S-NWN (see the `AttributeError`
above). Replaced with an independent sine washout before each trial: reset to blank, drive with a
fixed 5Hz/`V_safe` sine via `run_until_settled()` until settled, then step through the trial
sequence directly (not via `run()`, which would reset the DUT again). The ESP guarantees both
trials converge to approximately the same driven state without saving or restoring any internal
attribute — this works against any `BaseDUT` implementation, not just LI-ESN.

Checked Section 2.3 (FMP/ESP) for the same risk Sonnet flagged: neither test saves/restores state
or touches `dut.x` — each trial already gets its own independent `dut.reset()` + distinct settling
input, so no changes were needed there.

Verified: a fresh end-to-end run with LI-ESN produces non-degenerate SP output close to the
pre-fix baseline (`d̄` 0.0188 vs 0.0182, `σ/d̄` 0.73 vs 0.72, as expected given LI-ESN's strong
ESP), and the re-trial against Ag2S-NWN above confirms the washout works generically with no
DUT-specific code needed.

---

## Historical record: the original floating-ground bug (now fixed above)

**Status (as originally filed, prior to the 00c fix):** Implemented per spec
(`prc_toolkit/dut/ag2s_nwn.py`). The literal smoke test given in the
prompt (`tests/test_ag2s_nwn.py::test_ag2s_nwn_shapes` — checks output shape, absence of NaN, and
`0 <= L <= L_GAP`) **passes**. However, the prompt's own recommended physical validation ("Check
`dut.L` after a run — it should show a mix of values between 0 and `L_GAP`, not all zeros or all
`L_GAP`") **fails** under the default / toolkit-standard configuration (`N_in=1`).

### Finding

With exactly one input electrode (`N_in=1`) and no explicit ground/reference node, the voltage
solve in `step()` (Step 2–3 of the prompt: build nodal Laplacian `G_node`, solve
`G_hat @ V_hat = -G_node[unknown, in] @ u` via `lstsq`) always returns the trivial solution
`V_hat = u * ones` — i.e. **every floating node in the network is set to the same voltage as the
single input node**, regardless of graph topology, connectivity, or seed.

This is a property of the math, not an implementation slip: for any graph, the full Laplacian's
rows sum to zero, so `(L_uu + L_uk) @ 1 = 0` always holds. Substituting `V_hat = u * 1` into
`L_uu @ V_hat = -L_uk @ u` gives `u * (L_uu + L_uk) @ 1 = 0`, which is satisfied identically. With
a single voltage source and no second fixed-voltage (ground) node providing a return path, "zero
current everywhere, every node at the source voltage" is a *valid* solution to the linear system —
and it is what `lstsq` returns.

Consequence: `V_junc = |V[i] - V[j]|` is 0 V for every junction, every timestep, for the entire
run. Since `V_junc < V_THRESH` always holds, `step()`'s Step 4 always takes the `dl = -rate` branch
(annihilation) — but `L` starts at 0 and is clamped at 0, so it simply stays at 0 forever. No bridge
ever switches ON. This holds regardless of `growth_rate` or `V_THRESH` tuning; it is not a
sensitivity/tuning issue as the prompt's inline comments assume for those two parameters.

### Verification

Confirmed empirically (see session transcript) across four seeds (0–3) and two connectivities
(0.3, 0.4) with `N_wires=20, N_in=1`: after any `step()` call, `dut.V` is numerically identical to
the input voltage at every node. A 3000-timestep multisine-driven run (`N_wires=20`,
default `growth_rate=1e-10`) left `dut.L` at exactly zero across all 60 active junctions.

As a control, giving the device **two** input electrodes at *different* fixed voltages
(`N_in=2`, e.g. `u=[0.8, 0.2]`) does produce a genuine voltage gradient across the network
(`V` ranges from 0.2 to 0.8), confirming the root cause is specifically the single-voltage-source
/ no-ground configuration, not a bug in the Laplacian assembly or lstsq call.

This matches (but is more severe than) the prompt's own "Known limitations → Floating ground" note,
which describes the missing ground node but asserts "the floating ground does not affect
correctness of the bridge model." That assertion does not hold for the `N_in=1` case used
throughout Sections 1–3 of this toolkit (see README "Input convention: scalar only for Sections
1–3").

---

## Hardware mode (prompts/00g_hardware_mode.md) — demo dataset, plus two spec/code mismatches

**Status: RESOLVED**, with two deviations from the prompt found by cross-checking it against the
actual notebooks before implementing.

**Deviation 1 — Notebook 03's file-naming scheme assumed a structure that doesn't exist.** The
prompt names three separate hardware output files for Section 3 (`consistency_trial_{i}`,
`shared_run`, `mc_run`), implying three independent `dut` collection points. In the actual
notebook there is exactly **one** data-collection cell — it produces two trials (`H0`, `H1`) that
every later section (3.2 through 3.6) reuses unmodified. Used two files instead
(`section3_shared_run_trial00.csv` / `trial01.csv`), dropping the two fictitious ones.

**Deviation 2 — Notebook 01 contradicted itself.** Change 1 says "no code changes to notebook 01,"
but the operator instructions for `section1_multisine.csv` said to record it "before running
Section 1.3" — implying 1.3 would load it, which nothing wired up. Resolved conservatively at the
time (kept notebook 01 unwired, softened the instruction to "manual reference only"); later
superseded by direct request — see the notebook 01 section below, where 1.3 is now genuinely wired.

**Design not in the prompt at all: a committed demo dataset.** Added at the user's request, on top
of the prompt: `data/demo/hardware_outputs/` is a committed, pre-generated set of outputs (captured
by literally running the LI-ESN simulation with a `SAVE_DEMO_HARDWARE_DATA` flag flipped on) so
`DUT_MODEL="hardware"` works out of the box with no physical DUT attached. `HARDWARE_DATA_PATH`
defaults to this demo path; a real operator repoints it at `data/hardware/outputs` (the real,
initially-empty target `00_hardware_signals.ipynb` creates) once they've recorded actual hardware
data. `data/hardware/` is gitignored (operator-specific, regenerable); `data/demo/` is tracked.

Verified: full pytest; LI-ESN chain (01→02→03→04) byte-identical to the pre-change baseline;
`DUT_MODEL="hardware"` against the committed demo data reproduces the LI-ESN results exactly for
every notebook, including a clean `FileNotFoundError` (not a crash) when pointed at the real, empty
path instead.

---

## Column D for Ag2S-NWN — correcting the state-proxy metric (Sonnet's `V` proposal vs. `L`)

**Status: RESOLVED.** Column D (State LP, `d(Wx)/dt` vs `Wx(t)`) was implemented for LI-ESN first
(`Wx_scalar = ‖W @ x‖`, straightforward — `x` is the reservoir's actual persistent hidden state).
For Ag2S-NWN, Sonnet proposed the analog `Wx_scalar = ‖dut.V‖` (node voltages).

**This does not hold up against the DUT's own code.** `Ag2SNWN.step()` re-solves `V` from scratch
every timestep via a KCL least-squares solve, as an algebraic function of the current input `u` and
the current conductance matrix `G` — it carries no memory of its own. `Ag2SNWN.reset()`'s own
docstring says this explicitly: *"the Ag2S-NWN state is fully described by L (and derived G), not a
hidden state vector."* `V` also includes the input nodes (driven directly by `u`) and output nodes
(already shown in columns A–C), so `‖V‖` would mostly just re-plot signals already visible
elsewhere in the fingerprint grid — structurally it's closer to LI-ESN's `h` than to `x`.

The actual analog of `x` — the thing that persists across timesteps and gives this substrate fading
memory in the first place — is `L`, the bridge-length matrix. Implemented `Wx_scalar = ‖L‖_F`
instead. Verified: LI-ESN's fingerprint output is unaffected (byte-identical); Ag2S-NWN now renders
a real, non-trivial Column D (values on the order of 1e-8, consistent with `L_GAP = 1.53e-9`)
instead of omitting it.

---

## Codebase cleanup (prompts/00h_cleanup.md) — two more prompt bugs found

**Status: RESOLVED**, with the same pattern as before: cross-checking the prompt against the actual
code before implementing caught two bugs that would have shipped silently.

1. **`prepare_input()`'s factory signature was missing its required `amplitude` argument.** The
   prompt specifies `prepare_input(u_seq, dut_model)`, but every real call site
   (`V_SAFE`, `V_SAFE*0.5`, `V_SAFE*0.4`, `WASHOUT_AMP`, `lz_amplitude`, ...) needs an amplitude to
   correctly rescale `[-amplitude, amplitude] -> [0, amplitude]` for Ag2S-NWN — the amplitude bound
   varies per call site and can't be inferred. Implemented as
   `prepare_input(u_seq, amplitude, dut_model)` instead, preserving the existing per-notebook
   behavior exactly, just centralized in the new `prc_toolkit/dut/factory.py`.
2. **The suggested FMP/ESP dedup helper used the wrong RNG seed formula.** The prompt's example
   seeds trials with `SEED + trial_idx + rng_offset`; the actual code seeds with
   `trial_i + rng_offset` (no `SEED` involved at all). Using the prompt's formula would have
   silently shifted every FMP/ESP RNG stream and changed results with no functional justification.
   Extracted the shared helper (`make_settle_input()`) preserving the real formula.

Also skipped the prompt's Change 2 entirely (collapsing `Wx_scalar`'s dead `if/else` back to an
unconditional `None`) — by the time this prompt was applied, Column D had already been implemented
for both DUT types (see above); applying Change 2 literally would have deleted that work.

Everything else applied as specified: `make_dut()`/`prepare_input()` factory adopted by all four
notebooks in place of copy-pasted instantiation blocks; `from collections import Counter` moved out
of notebook 03's ~23,000-iteration IPC loop; the dead `multisine` import removed from notebook 02;
the `DUT_CFG`/`DUT_PARAMS` double-assignment collapsed to one line in notebooks 02 and 03; the
unsupported "very high MC at large k" sentence removed from the README.

Verified: 47/47 pytest; full LI-ESN chain byte-identical to baseline; Ag2S-NWN and hardware-mode
smoke-tested on all four notebooks.

---

## Notebook 01: transient-tail fix (fingerprint + safe region) and hardware mode for Section 1.3

**Status: RESOLVED.**

**Transient tail in the Lissajous plots.** Both the visual fingerprint (notebook 02, §2.1) and the
Safe Region Sweep (notebook 01, §1.2) drove the DUT for a fixed duration and plotted the *entire*
trajectory, including the initial transient — this dominates the plot visually and makes the
converged limit-cycle shape hard to read. Fixed identically in both places: split the fixed
duration into a settle portion (driven, discarded) and a record portion (driven, plotted), same
total duration as before (`FP_SETTLE_PERIODS=2` / `FP_RECORD_PERIODS=3` for the fingerprint;
`SWEEP_SETTLE_PERIODS=2` / `SWEEP_RECORD_PERIODS=3` for the safe region sweep). The safe region
sweep previously called `run_until_settled()` before recording, but that criterion (mean output
magnitude rate of change) can flatten out well before the *phase-space trajectory* itself has
converged to its limit cycle — hence the tail persisting despite it. The fixed-discard approach
doesn't have that failure mode.

Side effect worth flagging: removing `run_until_settled()` from the safe region sweep changes how
many RNG draws are consumed by that cell (LI-ESN's `sigma_process`/`sigma_measure` noise draws from
a single `rng` object shared across the whole notebook run, and `dut.reset()` does not reset it).
This shifted Section 1.3's noise-driven metrics slightly (`H_linear_power` 0.0654→0.0643,
`noise_floor_peak` similarly small change) — expected and harmless, not a bug, but worth knowing
about since it moved the committed baseline numbers.

**Hardware mode for Section 1.3.** Notebook 01 previously had no `DUT_MODEL` variable at all (a
hardcoded `LIESN(...)` plus a commented-out manual Ag2S-NWN swap — the odd one out relative to
notebooks 02–04). Brought it in line: `DUT_MODEL`/`DUT_PARAMS` via the same `make_dut()` factory,
with `results/section1_results.json`'s `"DUT_params"` always serialized from a separate
`LIESN_PARAMS` dict regardless of which `DUT_MODEL` this notebook actually runs — preserving the
existing downstream handoff contract notebooks 02–04 depend on. Sections 1.1/1.2 remain
demonstration-only.

Found one real usability bug while testing "Run All" under `DUT_MODEL="hardware"`: the first
version guarded 1.1/1.2 with a plain `assert DUT_MODEL != "hardware"`, which halted execution
*before* ever reaching Section 1.3 — completely defeating the point of wiring 1.3 for hardware
mode, since cells run top-to-bottom. Fixed by having 1.1/1.2 print a message and skip gracefully
(`if DUT_MODEL == "hardware": print(...) else: <existing body>`) instead of asserting, so "Run All"
now proceeds straight through to Section 1.3 without manual cell-skipping.

Verified: full pytest; LI-ESN chain byte-identical (aside from the expected Section 1 RNG shift
above); Ag2S-NWN smoke-tested; full hardware-mode chain (01→02→03→04) runs end-to-end in one pass
against the committed demo dataset (extended with `section1_multisine.csv`) and reproduces the
LI-ESN results exactly.

### Decision (as originally made, prior to the 00c fix)

Per the project's build instructions: this counted as a 00b smoke-test failure. Ag2S-NWN was
**not** exercised as the active DUT in any of the Sections 1–3 notebooks. It remained in the
codebase exactly as specified by the prompt (code + its literal smoke test, both passing), and the
notebooks included it only as the prompt's own commented-out "Alternative DUT" block — never
uncommented or run — so no downstream test results were computed against a device known not to
produce bridge dynamics under this input convention. LI-ESN was the sole active DUT for Notebooks
01–03.

### Fix applied

See "RESOLVED" section at the top of this file. `prompts/00c_ag2s_nwn_ground_fix.md` implemented
exactly the fix anticipated below (an explicit ground/reference node at a fixed 0V), resolving this
finding.
