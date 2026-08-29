# Update Prompt: Ag2S-NWN Ground Reference Node

**File to update:** `prc_toolkit/dut/ag2s_nwn.py`

## Background

The current implementation has a degenerate voltage solve when `N_in=1`. With a single
fixed-voltage node and no ground reference, the reduced Laplacian is rank-deficient and
`lstsq` returns the trivial solution — every node is set to the input voltage, all junction
voltages are zero, and no bridge dynamics occur. The fix is to reserve one interior node
as an explicit ground reference electrode, held at 0 V in the solver alongside the input
nodes.

Node `N_in` is chosen as the ground node because it is always the first non-input,
non-output node by index, making the assignment deterministic and independent of the RNG.
This corresponds to a physical ground electrode in a real device.

---

## Change 1: Reserve node `N_in` as the ground electrode

In `__init__`, after the line that sets `self.in_nodes`, add:

```python
self.ground_node = N_in  # first non-input node; held at 0 V as voltage reference
```

Tighten the existing node count assertion from `<=` to `<`, to ensure there is always
at least one interior node available for ground:

```python
assert N_in + N_out < N_wires, "N_in + N_out must be strictly less than N_wires to reserve a ground node"
```

Update the class docstring to note that node `N_in` is reserved as the ground reference
electrode and is neither driven nor read.

---

## Change 2: Include the ground node in the voltage solve

In `step()`, Step 3, replace the current block that builds `unknown_nodes` and solves
for `V_hat` with the following:

```python
known_nodes = np.append(self.in_nodes, self.ground_node)
known_voltages = np.append(u, 0.0)
unknown_nodes = np.array([i for i in range(self.N_wires) if i not in set(known_nodes.tolist())])

G_hat = G_node[np.ix_(unknown_nodes, unknown_nodes)]
I_input = -G_node[np.ix_(unknown_nodes, known_nodes)] @ known_voltages
V_hat = np.linalg.lstsq(G_hat, I_input, rcond=None)[0]
self.V[unknown_nodes] = V_hat
self.V[self.ground_node] = 0.0
```

The ground node is now part of the known-voltage set, providing a genuine reference
distinct from the driven input node(s). This resolves the rank deficiency for all
`N_in >= 1`.

---

## Change 3: Reset the ground node voltage in `reset()`

In `reset()`, after the existing reset lines, add:

```python
self.V[self.ground_node] = 0.0
self.V_prev[self.ground_node] = 0.0
```

This ensures the ground node is consistently at 0 V after a reset, not left as whatever
value it held from a previous run.

---

## Change 4: Update the disconnected graph warning

The existing rank check and warning message in `__init__` can remain as-is. Update its
warning text to mention that the ground node is node `N_in`, so the message is useful
when debugging:

```python
print(
    f"WARNING: Ag2SNWN graph is disconnected (rank={rank}, "
    f"N_wires={N_wires}). Ground node is node {N_in}. "
    f"Increase connectivity or use a different seed. "
    f"Voltage solve may be unreliable."
)
```

---

## What not to change

- Do not change `N_wires`, `in_nodes`, `out_nodes`, the bridge dynamics (Step 4), or
  the output readout (Step 5).
- Do not change the smoke test (`tests/test_ag2s_nwn.py::test_ag2s_nwn_shapes`). It
  checks output shape, absence of NaN, and `0 <= L <= L_GAP`, all of which should
  still pass.

---

## Verification

After applying this change, manually verify the fix by running a short simulation with
`N_in=1` and confirming:

1. `dut.V` shows a genuine voltage gradient across nodes after a `step()` call —
   values should differ from one another, not all equal the input voltage.
2. `dut.V[dut.ground_node]` is exactly `0.0` after every `step()` and after `reset()`.
3. After a multi-timestep run (e.g. 100 steps with a sinusoidal input), `dut.L` shows
   a mix of values between `0` and `L_GAP` across active junctions — not all zeros.

This replaces the physical validation note in `build_notes.md` which documented the
failure of the original implementation under `N_in=1`.
