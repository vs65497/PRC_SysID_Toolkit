# Update Prompt: Separation Property Test — Sine Washout Initialization

**File to update:** The Section 2.4 cell(s) in `02_system_identification.ipynb`

## Background

The current Section 2.4 implementation saves the DUT's internal state after a shared
run and restores it via `dut.reset(x0=x_common)` before each trial. This is a
simulation shortcut that has no hardware equivalent — a physical DUT's internal state
cannot be read or written directly. It also breaks for any DUT that does not expose
a single state vector `x` (e.g. Ag2S-NWN, whose state lives in `L`, `G`, `V`,
`V_prev`).

Replace this with a hardware-compatible initialization procedure: before each trial,
reset the DUT to its blank state and drive it with a fixed sine washout sequence until
settled. Because the ESP guarantees convergence to a driven attractor, both trials
will reach approximately the same state after the washout, which is sufficient for
the separation property test. The two post-washout states will not be bit-identical
(especially with process noise), which is the honest hardware-faithful behavior.

`dut.reset()` with no arguments (blank state) is an acceptable simulation shortcut
and should be retained — physical reset procedures vary by substrate and are outside
the scope of this toolkit.

`run_until_settled()` already accepts an arbitrary input sequence and loops it
internally — no changes to that function are needed.

---

## Change: Replace the trial initialization block in Section 2.4

### Remove

Any code that:
- Calls `dut.x.copy()` or accesses any other DUT-internal state attribute to save
  a snapshot.
- Calls `dut.reset(x0=...)` with a non-None argument to restore a snapshot.
- Runs a single shared initialization run and then forks state into two trials.

### Replace with

The following structure. Insert it before the two-trial loop, replacing the
snapshot/restore logic:

```python
# --- Sine washout sequence (hardware-compatible initialization) ---
# A fixed 5 Hz sine at V_safe amplitude is used to drive the DUT to a
# repeatable settled state before each trial. This mimics what a hardware
# operator would do: apply a known signal from a function generator, wait
# for the device to settle, then trigger the test sequence.
# The washout is run independently before each trial (not shared) so that
# no internal state needs to be saved or restored — making this procedure
# directly portable to hardware.

WASHOUT_FREQ = 5.0          # Hz — fixed, hardware-operator-friendly
WASHOUT_AMP  = V_SAFE       # match the operating amplitude used in the test

t_washout = np.arange(0, 1.0, DT)   # 1-second template, looped by run_until_settled
u_washout = WASHOUT_AMP * np.sin(2 * np.pi * WASHOUT_FREQ * t_washout)
u_washout_seq = u_washout.reshape(-1, 1)   # shape (T_template, 1) per BaseDUT convention


def run_sp_trial(dut, u_trial):
    """
    Reset DUT, run sine washout until settled, then run the trial sequence.
    Returns H: output history over the trial only, shape (T_trial, N_h).
    Hardware equivalent: power-cycle/reset DUT, apply sine from function
    generator until settled, then trigger test sequence.

    NOTE: do NOT call dut.run() here — run() resets the DUT internally before
    stepping, which would wipe out the settled state. Step manually instead.
    """
    dut.reset()
    run_until_settled(dut, u_washout_seq)   # DUT is now at settled state

    # Step through the trial sequence without resetting
    u_trial = np.asarray(u_trial)
    T_trial = u_trial.shape[0]
    H = None
    for t in range(T_trial):
        h = dut.step(u_trial[t])
        if H is None:
            H = np.empty((T_trial, h.shape[0]), dtype=h.dtype)
        H[t] = h
    return H
```

Then replace the two trial calls with:

```python
H0 = run_sp_trial(dut, u_trial_0)
H1 = run_sp_trial(dut, u_trial_1)
```

Where `u_trial_0` and `u_trial_1` are the Poisson spike train sequences already
constructed earlier in the cell (the original and the delayed-spike variant),
shaped `(T_trial, 1)` per the existing convention.

---

## What not to change

- Do not change how `u_trial_0` and `u_trial_1` are constructed (Poisson spike
  trains, delayed spike variant).
- Do not change how `d(t)`, `d_bar(t)`, and `σ(t)` are computed from `H0` and
  `H1` — those remain identical.
- Do not change `run_until_settled()` in the shared library.
- Do not change `dut.reset()` behavior — blank-state reset (no argument) is
  the correct and intended call here.
- Do not add `get_state()`/`set_state()` to `BaseDUT` — that approach is
  being deliberately avoided in favor of the washout procedure.

---

## Notes for the hardware operator (add as a markdown cell before the trial code)

Add or update the prose cell immediately before the trial code to include the
following note:

> **Hardware note:** Before each trial, apply a 5 Hz sine wave at V_safe amplitude
> to the DUT input and monitor the output until it has settled (output variance
> across a sliding window falls below the settling threshold). Once settled, trigger
> the trial input sequence immediately. Repeat this washout independently before
> each trial — do not attempt to preserve the DUT state between trials. The ESP
> guarantees that both trials will reach approximately the same driven state after
> a sufficiently long washout, which is the intended common starting condition for
> the separation test.

---

## Verification

After applying this change:

1. Run Section 2.4 end-to-end and confirm it produces non-degenerate `d_bar(t)`
   and `σ(t)` curves (not all-zero, not NaN).
2. Confirm the cell runs without accessing `dut.x` or any other DUT-internal
   attribute outside of `step()`, `run()`, and `reset()`.
3. Confirm the cell runs identically with both LI-ESN and Ag2S-NWN as the active
   DUT (swap in Ag2S-NWN temporarily to verify — no AttributeError should occur).
