# Prompt 00b — Ag2S Nanowire Network DUT (`prc_toolkit/dut/ag2s_nwn.py`)

## Context

This prompt adds a second simulated DUT to the shared library defined in Prompt 00.
The silver-sulfide nanowire network (Ag2S-NWN) is implemented as a liquid state machine (LSM)
with electrochemically actuated atomic switches as nonlinear activation. It is derived in SM.2
of the paper. This file should be placed at `prc_toolkit/dut/ag2s_nwn.py` and follows the
same `BaseDUT` interface defined in `prc_toolkit/dut/base.py`.

The Ag2S-NWN is a more physically realistic PRC substrate than the LI-ESN. It is slower
to simulate due to the simultaneous voltage solve required at each timestep.

---

## Network size guidance

```python
N_WIRES_DEFAULT = 20  # NOTE: 20 nanowires is approximately the practical maximum for
                       # real-time simulation performance. Beyond this, the simultaneous
                       # voltage solve (pseudoinverse of the conductance matrix) becomes
                       # a runtime bottleneck. Keep at 20 for demonstration purposes.
```

---

## Physical model summary (from SM.2)

The Ag2S-NWN is represented in its dual form: nanowires become neurons, and the junctions
between them (atomic switches) become resistive edges. The network is an undirected random
graph with adjacency matrix A.

At each timestep, silver-sulfide bridges grow or annihilate at each junction based on the
current flowing through them. When a bridge spans the gap completely, the junction switches
ON (conducting); otherwise it is OFF (insulating). The network voltages are solved
simultaneously using Kirchhoff's Current Law (KCL) and the Moore-Penrose pseudoinverse.

---

## Class: `Ag2SNWN(BaseDUT)`

### Physical constants (empirical, from Hasegawa2011 and Nayak2011)

These are fixed physical values, not tunable parameters. They reflect Ag2S gap-type
atomic switches with Ag⁺ cation migration.

```python
# From Hasegawa2011 and Nayak2011:
L_GAP    = 1.53e-9   # Gap distance: 1.53 nm (Ag2S gap-type atomic switch)
G_ON     = 1/12500   # ON conductance: 1/12.5 kΩ ≈ G₀ = n·2e²/h for n=1 (quantum conductance)
G_OFF    = 1e-9      # OFF conductance: ~1/100 MΩ (leakage)
BETA     = 32.9      # Exponential voltage sensitivity (V⁻¹) from Nayak2011.
                     # Controls how steeply switching speed increases with junction voltage.
                     # Used directly in bridge length update: dl ∝ exp(BETA * V_junc).
V_THRESH = 0.01      # Switching threshold voltage (V). Below this, bridges do not move.
                     # The empirical value from Hasegawa2011 is ~0.3 V for a two-terminal
                     # device under fixed bias. In a network, junction voltages are much
                     # smaller (divided across many nodes), so 0.3 V is too high and will
                     # prevent any switching. Start low (0.01 V) and tune upward manually
                     # until the desired sparsity of active switches is achieved.
                     # ← TUNE THIS for your network size and input amplitude.
```

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `N_wires` | int | 20 | Number of nanowires (neurons). Max ~20 for performance. |
| `N_in` | int | 1 | Number of input electrodes. |
| `N_out` | int | 3 | Number of output electrodes. These are the observable h(t) nodes. |
| `connectivity` | float | 0.3 | Edge probability for random graph A (Erdős–Rényi). |
| `dt` | float | DT | Timestep from config. |
| `sigma_process` | float | 0.0 | Process noise gain — std of Gaussian noise added to bridge length update. |
| `sigma_measure` | float | 0.0 | Measurement noise gain — std of Gaussian added to h(t) output. |
| `growth_rate` | float | 1e-10 | Scales the bridge length update. Effective rate is growth_rate * exp(BETA * V_junc). With BETA=32.9 and V_junc~0.1V, exp term~27, so effective rate~27*growth_rate per second. Increase growth_rate to speed up switching for testing; decrease to slow it down. ← TUNE THIS alongside V_THRESH. |
| `seed` | int | 42 | Random seed for graph generation and noise. |

Note: `D_growth` is removed as a constructor parameter. Direction of bridge growth is
determined internally from the sign of voltage change at each junction (see Step 5).

### Internal state

```python
self.A         # Adjacency matrix, shape (N_wires, N_wires). Binary, symmetric, zero diagonal.
self.L         # Bridge lengths, shape (N_wires, N_wires). Initialized to 0. Units: meters.
               # L[i,j] ∈ [0, L_GAP]. Clamped — cannot exceed gap distance or go below 0.
self.G         # Conductance matrix, shape (N_wires, N_wires).
               # G[i,j] = G_ON where A[i,j]=1 and L[i,j] >= L_GAP (bridge ON),
               # G[i,j] = G_OFF where A[i,j]=1 and L[i,j] < L_GAP (bridge OFF).
               # G[i,j] = 0 where A[i,j]=0 (no junction).
self.V         # Voltage at each node, shape (N_wires,). Initialized to 0. Units: Volts.
self.in_nodes  # Indices of input electrode nodes, shape (N_in,).
self.out_nodes # Indices of output electrode nodes, shape (N_out,).
               # in_nodes and out_nodes must not overlap.
self.V_prev    # Previous timestep voltages, shape (N_wires,). Used for dV direction detection.
               # Initialized to zeros in __init__ and reset().
self.rng           # np.random.default_rng(seed)
# Scalar parameters stored as instance attributes for use in step():
self.N_wires       # int
self.in_nodes      # np.ndarray — electrode indices (stored above, repeated here for clarity)
self.out_nodes     # np.ndarray
self.dt            # float — timestep
self.growth_rate   # float — bridge growth scaling factor
self.sigma_process # float — process noise gain
self.sigma_measure # float — measurement noise gain
```
All constructor parameters must be stored as `self.<name>` in `__init__`.

### Graph initialization (`__init__`)

1. Define module-level physical constants (outside the class):
   ```python
   L_GAP    = 1.53e-9   # meters
   G_ON     = 1/12500   # siemens
   G_OFF    = 1e-9      # siemens
   BETA     = 32.9      # V^-1 — voltage sensitivity, used in exp(BETA * V_junc)
   V_THRESH = 0.01      # volts — tunable, see physical constants note above
   ```

2. Generate symmetric Erdős–Rényi random graph:
   ```python
   rng = np.random.default_rng(seed)
   upper = rng.random((N_wires, N_wires)) < connectivity
   A = np.triu(upper, k=1)
   A = A + A.T   # symmetric, zero diagonal
   ```

   After generating A, verify the graph is connected. If not, warn the user:
   ```python
   import numpy.linalg as nla
   # A connected graph has exactly one zero eigenvalue in its Laplacian.
   # Simple check: use BFS/DFS or check that (A + I)^N_wires has no zero rows.
   # Practical shortcut for small N: check np.linalg.matrix_rank of the Laplacian.
   D = np.diag(A.sum(axis=1))
   lap = D - A
   rank = np.linalg.matrix_rank(lap)
   if rank < N_wires - 1:
       print(f"WARNING: Ag2SNWN graph is disconnected (rank={rank}, N_wires={N_wires}). "
             "Increase connectivity or use a different seed. Voltage solve may be unreliable.")
   # Also check that no output node is fully isolated (no edges).
   # An isolated output node will always read 0V silently.
   for node in self.out_nodes:
       if self.A[node, :].sum() == 0:
           print(f"WARNING: output node {node} has no edges in A. "
                 "It will always output 0V. Use a different seed or "
                 "increase connectivity.")
   ```

3. Assign electrode indices:
   ```python
   # assert N_in + N_out <= N_wires
   self.in_nodes  = np.arange(N_in)
   self.out_nodes = np.arange(N_wires - N_out, N_wires)
   ```

4. Initialize bridge lengths to zero, conductances to G_OFF for all active edges:
   ```python
   self.L = np.zeros((N_wires, N_wires))
   self.G = np.where(A == 1, G_OFF, 0.0)   # all bridges start OFF
   self.V = np.zeros(N_wires)
   ```

### `reset(x0=None)`

Reset bridge lengths to zero and recompute conductances. All bridges return to OFF state.

```python
self.L      = np.zeros((self.N_wires, self.N_wires))
self.G      = np.where(self.A == 1, G_OFF, 0.0)
self.V      = np.zeros(self.N_wires)
self.V_prev = np.zeros(self.N_wires)   # must reset V_prev or first-step dV_junc is stale
```

The `x0` parameter is accepted for interface compatibility but ignored — the Ag2S-NWN
state is fully described by L (and the derived G), not a hidden state vector.
Print a warning if x0 is not None: "Ag2SNWN: x0 ignored; state reset to L=0 (all bridges OFF)."

### `step(u: np.ndarray) -> np.ndarray`

`u` is shape `(N_in,)` — voltage applied at input electrodes (Volts, non-negative).
Returns `h` of shape `(N_out,)` — voltage measured at output electrodes.

**Step 1: Apply input voltages**

```python
self.V[self.in_nodes] = u
```

**Step 2: Build nodal conductance matrix**

**Note on corrected mathematics:** The original implementation had two errors in the
conductance matrix construction: (1) it only filled the upper triangle of the off-diagonal
entries, leaving the matrix asymmetric; (2) it contained a `if j == i+1: Gij = -Gij`
branch with no physical justification (debugging artifact). The correct nodal conductance
matrix (graph Laplacian) is:

```python
G_node = np.zeros((self.N_wires, self.N_wires))
for i in range(self.N_wires):
    for j in range(i + 1, self.N_wires):   # upper triangle only — prevents double-counting
        if self.A[i, j] == 1:
            g = self.G[i, j]
            G_node[i, i] += g    # diagonal: sum of conductances at node i
            G_node[j, j] += g    # diagonal: sum of conductances at node j
            G_node[i, j] -= g    # off-diagonal: negative (KCL)
            G_node[j, i] -= g    # symmetric
```

This is the standard nodal admittance matrix. It is symmetric by construction.
IMPORTANT: iterate upper triangle only (j > i). Processing both (i,j) and (j,i)
would double-count every conductance, corrupting the Laplacian.

**Step 3: Solve for unknown node voltages**

```python
# Use set for O(1) lookup; convert to sorted array for numpy indexing
in_set        = set(self.in_nodes.tolist())
unknown_nodes = np.array([i for i in range(self.N_wires) if i not in in_set])

G_hat   = G_node[np.ix_(unknown_nodes, unknown_nodes)]
I_input = -G_node[np.ix_(unknown_nodes, self.in_nodes)] @ u
V_hat   = np.linalg.lstsq(G_hat, I_input, rcond=None)[0]
self.V[unknown_nodes] = V_hat
```

**Note:** The original implementation constructed the `b` vector using `np.dot(ones, A1)`
(summing columns of the source/sink submatrices). The correct nodal injection vector is the
standard KCL formulation above: the current injected at unknown nodes by the known input
nodes is `−G_sub @ u`, where `G_sub` is the off-diagonal block between unknown and input
nodes. This is mathematically equivalent to the standard modified nodal analysis (MNA)
formulation and avoids the ambiguity in the original `b` construction.

**Step 4: Update bridge lengths — signed bidirectional dynamics**

Bridge growth and annihilation are modelled by a **signed dl**: the exponential
of the junction voltage sets the rate, and the sign is determined by whether the
**junction voltage is rising or falling** relative to the previous timestep.

Using the change in junction voltage (rather than node j alone) is physically correct:
the junction voltage |V[i]-V[j]| is the quantity that drives ion migration, and its
trend — rising or falling — determines which way ions drift. Using a single node's
voltage change is arbitrary for a symmetric edge and can give wrong direction when
nodes move in opposite directions.

Store `self.V_prev` — a copy of `self.V` from the previous timestep.

```python
for i in range(self.N_wires):
    for j in range(i + 1, self.N_wires):   # upper triangle only — prevents double-counting
        if self.A[i, j] != 1:
            continue

    V_junc      = abs(self.V[i] - self.V[j])          # current junction voltage (V)
    V_junc_prev = abs(self.V_prev[i] - self.V_prev[j]) # previous junction voltage (V)
    dV_junc     = V_junc - V_junc_prev                 # signed change in junction voltage

    # ── Signed bridge length update ───────────────────────────────────
    # dl > 0: bridge grows (extends toward gap, toward ON state)
    # dl < 0: bridge shrinks (retracts, toward OFF state / annihilation)
    #
    # Rate magnitude: growth_rate * exp(BETA * V_junc) * dt
    # exp(BETA * V_junc) increases steeply with junction voltage (BETA=32.9 V⁻¹),
    # so switching accelerates sharply above V_THRESH — faithful to Nayak2011.
    # Using junction voltage (not current) matches the Terabe single-junction model,
    # where current is implicitly V/R_gap (fixed R in their setup). In a network,
    # current depends on global conductance state and is a poor local proxy.
    # Junction voltage is the correct local driving quantity.
    #
    # Before bridge connects: G[i,j]=G_OFF (tiny), junction voltage is substantial
    #   → exp(BETA * V_junc) large → fast growth if above threshold.
    # After bridge connects: G[i,j]=G_ON, nodes equilibrate → V_junc → 0
    #   → exp(BETA * 0) = 1 → slow growth → bridge stabilizes at L_GAP naturally.

    rate = self.growth_rate * math.exp(min(BETA * V_junc, 20.0)) * self.dt
    # Cap exponent at 20 (exp(20)≈5e8) to prevent float64 overflow.
    # At V_junc=0.3V, BETA*V_junc≈9.87 → rate already ~19000× base.
    # Beyond that, the exact value doesn't matter — bridge switches fast.

    if V_junc < V_THRESH:
        # Below switching threshold: force annihilation regardless of trend.
        # Prevents noise-driven growth in quiescent junctions.
        dl = -rate   # negative → bridge retracts

    elif dV_junc > 0:
        # Junction voltage rising, above threshold: ions driven toward gap → growth
        dl = rate    # positive → bridge extends

    elif dV_junc < 0:
        # Junction voltage falling, above threshold: ions recede → annihilation
        dl = -rate   # negative → bridge retracts

    else:
        # No junction voltage change: bridge does not move.
        # Note: if sigma_process > 0, noise is added below and can
        # drive the bridge in either direction — this is intentional.
        dl = 0.0

    # ── Process noise ─────────────────────────────────────────────────
    if self.sigma_process > 0:
        dl += self.rng.normal(0, self.sigma_process) * self.dt

    # ── Apply update and enforce physical bounds ──────────────────────
    self.L[i, j] = np.clip(self.L[i, j] + dl, 0.0, L_GAP)
    self.L[j, i] = self.L[i, j]

    # ── Switch state update ───────────────────────────────────────────
    # ON: bridge has fully spanned the gap AND voltage sustains it.
    # OFF: everything else (including partially grown bridges).
    if (self.L[i, j] >= L_GAP) and (V_junc >= V_THRESH):
        self.G[i, j] = G_ON
        self.G[j, i] = G_ON
    else:
        self.G[i, j] = G_OFF
        self.G[j, i] = G_OFF
```

**At the end of `step()`, before returning h:**
```python
self.V_prev = self.V.copy()
```

Initialize `self.V_prev = np.zeros(N_wires)` in both `__init__` and `reset()`.

**Step 5: Return output with optional measurement noise**

```python
h = self.V[self.out_nodes].copy()   # shape (N_out,) — voltages at output nodes
if self.sigma_measure > 0:
    h += self.rng.normal(0, self.sigma_measure, size=h.shape)
return h
```

### `run(u_seq, x0=None) -> np.ndarray`

`u_seq` shape `(T, N_in)` — input voltage sequence in Volts (non-negative).
Returns `H` shape `(T, N_out)`.
Uses base class default (calls `step()` in a loop).
Override for performance if needed, but loop is acceptable for N_wires ≤ 20.

### Input biasing note

The Ag2S-NWN is a physically rectifying substrate: current flows preferentially from
input to output electrodes. For realistic simulation, input voltages should be
non-negative: `u ≥ 0`. The signal generators produce bipolar signals by default.
When using this DUT, callers should shift inputs: `u_biased = (u + amplitude) / 2`
to map [-amplitude, amplitude] → [0, amplitude]. Add a utility function
`bias_positive(u_seq, amplitude)` in `signals/generators.py` for this purpose.

The notebooks should call `bias_positive` when using Ag2SNWN as the DUT.

---

## Performance note

At each timestep, `step()` calls `np.linalg.lstsq` on a matrix of size
`(N_wires - N_in) × (N_wires - N_in)`. For N_wires=20, this is a 19×19 system —
fast enough for interactive use. For N_wires > 20, runtime grows as O(N³) and
simulation of 30 seconds at fs=100 Hz (3000 timesteps) may take several minutes.

---

## Additions to `signals/generators.py`

Add this function:

### `bias_positive(u_seq, amplitude) -> np.ndarray`

Map a bipolar signal `u_seq ∈ [-amplitude, amplitude]` to `[0, amplitude]`:
```python
return (u_seq + amplitude) / 2.0
```

Used when driving the Ag2S-NWN, which assumes non-negative input voltages.

---

## Additions to all three notebooks (Cells 2 and 3)

In every notebook's import cell, add:
```python
from prc_toolkit.dut.ag2s_nwn import Ag2SNWN
from prc_toolkit.signals.generators import bias_positive
```

In the DUT configuration cell, add a commented-out alternative DUT block:

```python
# ── Alternative DUT: Ag2S Nanowire Network ───────────────────────────
# Uncomment to use Ag2SNWN instead of LIESN.
# Physical constants (L_GAP, G_ON, G_OFF, V_THRESH) are fixed empirical values
# from Hasegawa2011/Nayak2011 — they are not constructor parameters.
# Input voltages must be non-negative (Volts). Use bias_positive() on all signals.
# N_wires <= 20 recommended for performance (see dut/ag2s_nwn.py).
#
# dut = Ag2SNWN(
#     N_wires=20,
#     N_in=1,
#     N_out=3,
#     connectivity=0.3,
#     sigma_process=0.0,
#     sigma_measure=0.0,
#     seed=42
# )
# # Shift bipolar signals to non-negative range before passing to this DUT:
# # u_biased = bias_positive(multisine(...), amplitude=V_SAFE)
```

---

## Known limitations and open questions

Document these honestly in code comments and in SM.2 of the paper.

- **Cold start / inrush:** At t=0, `V_prev = 0` and input jumps immediately to u[0].
  The first `dV_junc` is therefore `|u[0] - 0| = |u[0]|` — large and positive —
  potentially triggering spurious growth on all input-adjacent edges in the first
  timestep. In hardware, this corresponds to inrush current that can damage input
  electrodes. Mitigation: keep `growth_rate` small so the first-timestep dl is
  negligible, and ramp input amplitude slowly (as Section 1.2 prescribes).

- **Floating ground:** No explicit ground node is defined. The `lstsq` solve returns
  a least-norm voltage solution, meaning absolute voltages float. Only junction voltage
  *differences* V_junc = |V[i]-V[j]| are physically meaningful, and the bridge dynamics
  use only these differences — so the floating ground does not affect correctness of
  the bridge model. In hardware, output nodes are tied to ground via high-resistance
  dividers, providing a reference. This is a hardware compensation problem deferred
  to a later design iteration.

- **Direction detection at steady state:** When the network reaches a steady-state
  input (constant u, all node voltages settled), `dV_junc = 0` for all edges and
  `dl = 0` (ignoring noise). Bridges neither grow nor shrink. This is physically
  correct — a DC steady state has no ion migration — but means the network has no
  dynamics under constant input. Use time-varying inputs (as all toolkit tests do).

- **Validation:** Linear MC is the recommended first sanity check. A working Ag2S-NWN
  should show nonzero but finite MC with a decaying MC_k profile. If MC_k is flat
  (infinite memory) or all zero (no memory), the bridge dynamics are not functioning
  correctly. Check `dut.L` after a run — it should show a mix of values between 0
  and L_GAP, not all zeros or all L_GAP.

## Diagnostic cell (add to notebooks, toggled by flag)

In the DUT configuration cell of each notebook, add:

```python
ENABLE_DIAGNOSTICS = False   # ← set True to enable bridge state heatmap after each run
                              # WARNING: adds significant overhead. Use for debugging only.
```

After any `dut.run()` call in the notebook, wrap a diagnostic plot:

```python
if ENABLE_DIAGNOSTICS and isinstance(dut, Ag2SNWN):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dut.L / L_GAP, vmin=0, vmax=1, cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax, label='Bridge length / L_GAP')
    ax.set_title("Ag2S-NWN bridge state (0=OFF, 1=ON)")
    ax.set_xlabel("Node j"); ax.set_ylabel("Node i")
    plt.tight_layout(); plt.show()
    n_on = np.sum(dut.L >= L_GAP) // 2   # symmetric matrix, count upper triangle
    n_active = np.sum(dut.A) // 2
    print(f"Bridges ON: {n_on} / {n_active} active junctions "
          f"({100*n_on/max(n_active,1):.1f}%)")
```

This is the fastest way to confirm the bridge dynamics are working. Expect a mix of
ON and OFF bridges after a run. All-zero means no switching occurred (growth_rate or
V_THRESH too restrictive). All-one means all bridges saturated (growth_rate too large
or V_THRESH too low).

---

## Smoke test additions (`tests/`)

Add to the smoke test file:

```python
def test_ag2s_nwn_shapes():
    from prc_toolkit.dut.ag2s_nwn import Ag2SNWN
    dut = Ag2SNWN(N_wires=10, N_in=1, N_out=2, seed=0)
    # Inputs must be non-negative voltages. Use a small positive range for smoke test.
    u_seq = np.random.default_rng(1).uniform(0.0, 0.5, size=(50, 1))
    H = dut.run(u_seq)
    assert H.shape == (50, 2), f"Expected (50,2), got {H.shape}"
    assert not np.any(np.isnan(H)), "NaN in output"
    # Verify bridge lengths stay within physical bounds [0, L_GAP]
    from prc_toolkit.dut.ag2s_nwn import L_GAP
    assert np.all(dut.L >= 0.0), "Bridge length went negative"
    assert np.all(dut.L <= L_GAP + 1e-15), "Bridge length exceeded L_GAP"
    print("Ag2SNWN smoke test passed.")
```
