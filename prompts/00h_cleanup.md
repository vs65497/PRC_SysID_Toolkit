# Update Prompt: Codebase Cleanup

**Files to update:**
- `prc_toolkit/dut/factory.py` (new)
- `02_system_identification.ipynb`
- `03_system_characterization.ipynb`
- `04_benchmarks.ipynb`
- `README.md`

---

## Before making any changes — read and confirm

Read through notebooks 02, 03, and 04 and locate every occurrence of:
- The `DUT_MODEL`/`HARDWARE_MODE` configuration comments
- The `assert DUT_MODEL in (...)` guard
- The `if/elif` DUT instantiation block
- The "Alternative DUT: Ag2S Nanowire Network" comment block
- The `prepare_input()` function definition and docstring

Confirm they are substantively identical across all three notebooks. If any notebook
has a meaningfully different version (not just cosmetic wording differences), flag it
before proceeding.

Also locate in notebook 02 the two cells for FMP and ESP (Section 2.3) that each
generate `u_A` via a multisine-with-random-phases block. Confirm they are
copy-pasted duplicates differing only in RNG offset.

---

## Overview

This prompt makes the following targeted cleanup changes:

1. Extract the duplicated DUT instantiation logic into a shared factory function.
2. Fix the no-op `if/else` for `Wx_scalar` in notebook 02.
3. Fix the `from collections import Counter` inside a loop in notebook 03.
4. Remove a stale `multisine` import in notebook 02.
5. Deduplicate the FMP/ESP multisine block in notebook 02.
6. Remove the redundant `DUT_CFG`/`DUT_PARAMS` double-assignment in notebooks
   02 and 03.
7. Delete the unsupported "very high MC at large k" sentence from `README.md`.

Changes are surgical — do not rewrite, reformat, or restructure anything beyond
what is described here.

---

## Change 1: DUT factory function

Create `prc_toolkit/dut/factory.py`:

```python
"""DUT factory — centralised instantiation and input preparation.

Notebooks import make_dut() and prepare_input() from here instead of
copy-pasting the instantiation block and prepare_input() helper.
"""

import numpy as np


def make_dut(dut_model: str, dut_params: dict):
    """
    Instantiate and return a DUT from a model name and parameter dict.

    dut_model: one of "liesn", "ag2s_nwn", "hardware".
    dut_params: dict of constructor kwargs for the chosen model.
                Ignored when dut_model == "hardware".

    Returns the instantiated DUT, or None for hardware mode.
    """
    if dut_model == "liesn":
        from prc_toolkit.dut.liesn import LIESN
        return LIESN(**dut_params)

    elif dut_model == "ag2s_nwn":
        from prc_toolkit.dut.ag2s_nwn import Ag2SNWN
        return Ag2SNWN(**dut_params)

    elif dut_model == "hardware":
        return None  # data loaded from filesystem; no simulated DUT

    else:
        raise ValueError(
            f"Unknown DUT_MODEL '{dut_model}'. "
            f"Supported values: 'liesn', 'ag2s_nwn', 'hardware'."
        )


def prepare_input(u_seq: np.ndarray, dut_model: str) -> np.ndarray:
    """
    Apply any DUT-specific input conditioning to u_seq.

    For Ag2S-NWN: applies bias_positive() to map bipolar signals to
    non-negative voltages, consistent with unidirectional current flow.
    For all other models: returns u_seq unchanged.

    u_seq: array, shape (T, N_u).
    Returns: array, shape (T, N_u).
    """
    if dut_model == "ag2s_nwn":
        # Locate bias_positive from existing usage in the codebase.
        from prc_toolkit.signals import bias_positive
        return bias_positive(u_seq)
    return u_seq
```

Then in each of notebooks 02, 03, and 04:

- Replace the entire duplicated DUT instantiation block (guard, `if/elif`,
  "Alternative DUT" comment, `prepare_input()` definition) with:

```python
from prc_toolkit.dut.factory import make_dut, prepare_input

assert DUT_MODEL in ("liesn", "ag2s_nwn", "hardware"), (
    f"DUT_MODEL '{DUT_MODEL}' is not supported. "
    f"Supported values: 'liesn', 'ag2s_nwn', 'hardware'."
)

dut = make_dut(DUT_MODEL, DUT_PARAMS)
```

- Replace all existing calls to `prepare_input(u_seq)` in each notebook with
  `prepare_input(u_seq, DUT_MODEL)` to match the updated signature.

- Remove the inline `prepare_input()` function definition from each notebook
  entirely — it now lives in the factory module.

Do not remove the "Alternative DUT: Ag2S Nanowire Network" comment block from the
notebooks if it contains recommended parameter values that are not captured in
`DUT_PARAMS` — move those values into a comment near the `DUT_PARAMS` definition
instead, so the operator can still see them. If `DUT_PARAMS` already captures them,
remove the comment block entirely.

---

## Change 2: Fix no-op Wx_scalar if/else (Notebook 02, Section 2.1)

Locate the block:

```python
if not HARDWARE_MODE:
    ...
    Wx_scalar = None
else:
    Wx_scalar = None
```

Both branches assign `None`. Replace with a single unconditional assignment and a
clarifying comment:

```python
# Column D (State LP) requires access to the hidden reservoir state — simulation
# only, not yet implemented. Wx_scalar = None causes the fingerprint renderer to
# omit column D automatically. Set HARDWARE_MODE = False and implement
# run_with_state() to enable this column in future.
Wx_scalar = None
```

---

## Change 3: Fix import inside loop (Notebook 03, Section 3.5 IPC)

Locate `from collections import Counter` inside the IPC loop body. Move it to the
top of that cell, alongside the other imports. Do not change anything else in
the cell.

---

## Change 4: Remove stale multisine import (Notebook 02)

Locate `from prc_toolkit.signals.generators import multisine` (or equivalent) in
notebook 02's import cell. Confirm `multisine` is not called anywhere in the
notebook. If confirmed unused, remove that import line. Do not remove any other
imports.

---

## Change 5: Deduplicate FMP/ESP multisine block (Notebook 02, Section 2.3)

Locate the two cells for FMP and ESP that each contain an identical multisine
generation block for `u_A`, differing only in RNG offset. Extract the shared
generation logic into a helper defined once before both cells:

```python
def make_trial_input(trial_idx: int, rng_offset: int = 0) -> np.ndarray:
    """
    Generate a multisine trial input with randomised phases.
    Returns u_seq: shape (T, 1).
    """
    rng_trial = np.random.default_rng(SEED + trial_idx + rng_offset)
    # ... existing multisine generation code, parameterised on rng_trial ...
    return prepare_input(u_seq, DUT_MODEL)
```

Then replace each copy-pasted block with a call to `make_trial_input(trial_idx,
rng_offset=...)` using the appropriate offset for FMP vs ESP. Do not change any
other logic in those cells.

---

## Change 6: Remove redundant DUT_CFG/DUT_PARAMS double-assignment
(Notebooks 02 and 03)

Locate any pattern of the form:

```python
DUT_CFG = s1["DUT_params"]
DUT_PARAMS = DUT_CFG
```

Replace with a single line:

```python
DUT_PARAMS = s1["DUT_params"]
```

Update any remaining references to `DUT_CFG` in the notebook to use `DUT_PARAMS`.
If `DUT_CFG` is not used anywhere after the assignment, simply remove it.

---

## Change 7: README — remove unsupported MC sentence

In `README.md`, locate the paragraph under "Linear MC" in the "What the results
mean" section. Find and delete the following sentence (exact wording may vary
slightly — identify it by meaning):

> "Very high MC at large k is unusual and may indicate the reservoir is memorizing
> the random input rather than processing it — check that the input signal was
> truly i.i.d."

Delete only that sentence. Do not change any surrounding text.

---

## What not to change

- Do not modify `BaseDUT`, `LIESN`, `Ag2SNWN`, or any other shared library file
  beyond the new `factory.py`.
- Do not reformat, rename, or restructure any notebook cell beyond the specific
  changes listed above.
- Do not change any parameter values, analysis logic, or plotting code.
- Do not change `DUT_MODEL = "liesn"` defaults.
- Cosmetic inconsistencies (header style, comment wording, column D explanation
  differences between notebooks) are out of scope — do not touch them.

---

## Verification

After applying all changes:

1. Run notebooks 02, 03, and 04 end-to-end with `DUT_MODEL = "liesn"` and confirm
   all results are numerically identical to the pre-change baseline.

2. Confirm `from prc_toolkit.dut.factory import make_dut, prepare_input` resolves
   correctly in all three notebooks.

3. Confirm `prepare_input(u_seq, DUT_MODEL)` is called consistently — no remaining
   calls to the old single-argument `prepare_input(u_seq)` signature.

4. Confirm `multisine` is not imported anywhere in notebook 02 (unless it is
   actually called somewhere — double-check before removing).

5. Confirm `Wx_scalar = None` appears exactly once in Section 2.1 of notebook 02,
   with no surrounding if/else.

6. Confirm `from collections import Counter` appears at the top of the IPC cell in
   notebook 03, not inside any loop.
