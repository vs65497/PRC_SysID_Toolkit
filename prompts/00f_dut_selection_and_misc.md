# Update Prompt: DUT Selection, Hardware Mode, Column D Gate, and NL Fix

**Files to update:**
- `01_fundamentals.ipynb`
- `02_system_identification.ipynb`
- `03_system_characterization.ipynb`

---

## Overview

This prompt makes four related changes across the three notebooks:

1. Add a top-level `DUT_MODEL` string parameter to notebooks 02 and 03 that controls
   which DUT is instantiated, with a guard that halts execution for unrecognized values.
2. Add a top-level `HARDWARE_MODE` boolean parameter to notebooks 02 and 03 that
   suppresses simulation-only outputs (currently: column D of the visual fingerprint).
3. Fix the measure of nonlinearity in notebook 03 Section 3.6.
4. Add commented-out Ag2S-NWN DUT declarations in notebooks 02 and 03.

Notebook 01 does not receive `DUT_MODEL` or `HARDWARE_MODE` — its simple disqualifying
models (wire, resistor, etc.) in Section 1.1 are pedagogical props hardcoded inline,
not selectable DUT candidates. Do not modify notebook 01 except as noted below.

---

## Change 1: `DUT_MODEL` parameter (Notebooks 02 and 03)

In each notebook, in the top configuration cell (where `V_SAFE`, `DT`, and similar
parameters are set), add:

```python
# --- DUT selection ---
# "liesn"    : Leaky Integrator Echo State Network (default simulated DUT)
# "ag2s_nwn" : Ag2S Nanowire Network (physically-derived simulated DUT)
# Add further entries here as new DUT models are implemented.
DUT_MODEL = "liesn"
```

Immediately after the configuration cell, add a guard cell:

```python
assert DUT_MODEL in ("liesn", "ag2s_nwn"), (
    f"DUT_MODEL '{DUT_MODEL}' is not supported in this notebook. "
    f"Supported values: 'liesn', 'ag2s_nwn'."
)
```

Then replace the existing hardcoded DUT instantiation block with a conditional:

```python
if DUT_MODEL == "liesn":
    from prc_toolkit.dut.liesn import LIESN
    dut = LIESN(**DUT_PARAMS)

elif DUT_MODEL == "ag2s_nwn":
    from prc_toolkit.dut.ag2s_nwn import Ag2SNWN
    dut = Ag2SNWN(**DUT_PARAMS)
```

Where `DUT_PARAMS` is the existing parameter dict already used for DUT construction.
If `DUT_PARAMS` is not currently a dict, refactor the DUT constructor call into one
at this point — it will be needed for the conditional anyway.

Do not change any DUT parameter values — only the instantiation structure.

---

## Change 2: `HARDWARE_MODE` parameter (Notebooks 02 and 03)

In the same top configuration cell, add:

```python
# --- Hardware mode ---
# False (default): simulation mode. Simulation-only outputs are shown
#                  (e.g. column D of the visual fingerprint, which requires
#                  access to the hidden reservoir state).
# True           : hardware mode. Simulation-only outputs are suppressed.
#                  Set this to True when running against a physical DUT.
HARDWARE_MODE = False
```

This flag is used in Change 3 below (column D gate in notebook 02). It should also
be applied to any other simulation-only output blocks encountered in the notebooks —
gate them with `if not HARDWARE_MODE:` and add a brief comment explaining why the
block is skipped in hardware mode.

---

## Change 3: Column D gate in Section 2.1 (Notebook 02)

In Section 2.1 (Visual Fingerprint), the column D block computes `Wx_scalar` from
the DUT's hidden state. This is simulation-only. Gate it as follows:

```python
if not HARDWARE_MODE:
    # Column D (State LP): requires access to hidden reservoir state.
    # Simulation only — omitted in hardware mode (HARDWARE_MODE = True).
    # For LI-ESN: Wx_scalar = np.linalg.norm(dut.W @ dut.x) at each step
    #   (requires run_with_state() — see note below).
    # For Ag2S-NWN: equivalent state is dut.V (node voltages).
    # Not yet implemented; Wx_scalar stays None and column D is omitted.
    Wx_scalar = None
else:
    Wx_scalar = None
```

If column D is already conditionally omitted when `Wx_scalar is None`, no further
changes to the fingerprint rendering code are needed — the existing None-check
already handles it. Just ensure the `Wx_scalar` assignment is inside the
`if not HARDWARE_MODE:` block as above.

---

## Change 4: Commented-out Ag2S-NWN DUT declarations (Notebooks 02 and 03)

After the `DUT_MODEL` conditional instantiation block added in Change 1, add a
comment block showing the recommended Ag2S-NWN configuration for reference:

```python
# --- Alternative DUT: Ag2S Nanowire Network ---
# To use, set DUT_MODEL = "ag2s_nwn" in the configuration cell above.
# Recommended configuration:
#
# DUT_PARAMS = dict(
#     N_wires=20,
#     N_in=1,
#     N_out=3,
#     connectivity=0.3,
#     sigma_process=0.01,
#     sigma_measure=0.005,
#     growth_rate=1e-10,
#     seed=42,
# )
#
# Note: Ag2S-NWN requires non-negative input voltages. Bipolar signals from
# the generators must be wrapped with bias_positive() before being passed
# to the DUT. The DUT_MODEL conditional above handles this automatically
# when DUT_MODEL = "ag2s_nwn" — see the input pipeline below.
```

Also add `bias_positive()` wrapping in the input pipeline for the Ag2S-NWN branch.
Wherever input signals are passed to the DUT in the notebook, wrap them as follows
if `DUT_MODEL == "ag2s_nwn"`:

```python
# Note: locate the correct import path for bias_positive() from existing
# usage in the codebase — do not guess the path.
from prc_toolkit.signals import bias_positive

def prepare_input(u_seq):
    """Apply bias_positive for rectifying substrates; identity for others."""
    if DUT_MODEL == "ag2s_nwn":
        return bias_positive(u_seq)
    return u_seq
```

Then replace bare `u_seq` arguments to `dut.step()` / `run_until_settled()` /
`run_sp_trial()` (Section 2.4) with `prepare_input(u_seq)`. Do not apply
`bias_positive()` inside the DUT itself — keep it in the notebook pipeline so
hardware operators can see and adjust it.

---

## Change 5: Fix measure of nonlinearity in Section 3.6 (Notebook 03)

In Section 3.6, find the line that computes `NL` (measure of nonlinearity).
Replace whatever is currently there with:

```python
NL = 1.0 - MC_total
```

Update the results dict entry accordingly:

```python
results["NL"] = float(NL)
```

If the section currently prints or displays `NL`, ensure the updated value is used.
No other changes to Section 3.6 are needed.

---

## What not to change

- Do not modify notebook 01 beyond any incidental formatting. The simple
  disqualifying DUT models in Section 1.1 (wire, resistor, etc.) are hardcoded
  inline and are not part of the `DUT_MODEL` selection system.
- Do not change any DUT parameter values, signal generator configurations, or
  analysis logic.
- Do not modify `BaseDUT`, `LIESN`, `Ag2SNWN`, or any shared library file —
  all changes are notebook-level only.
- Do not uncomment or activate Ag2S-NWN as the default DUT — `DUT_MODEL = "liesn"`
  remains the default in all committed notebooks.

---

## Verification

After applying all changes:

1. Run notebook 02 end-to-end with `DUT_MODEL = "liesn"`, `HARDWARE_MODE = False`.
   Confirm column D cell executes without error and `Wx_scalar` is set (even if None
   pending `run_with_state()` implementation).

2. Run notebook 02 with `HARDWARE_MODE = True`. Confirm column D block is skipped
   cleanly with no errors.

3. Run notebook 03 end-to-end with `DUT_MODEL = "liesn"`. Confirm `NL = 1.0 -
   MC_total` and that the value written to `section3_results.json` is updated.

4. In both notebooks, temporarily set `DUT_MODEL = "ag2s_nwn"` and confirm the
   guard passes, the Ag2S-NWN DUT is instantiated without error, and
   `bias_positive()` is applied to input signals. Then revert to `"liesn"` before
   committing.

5. Confirm `assert DUT_MODEL in (...)` fires with a clear message for an unrecognized
   value (e.g. `DUT_MODEL = "resistor"`).
