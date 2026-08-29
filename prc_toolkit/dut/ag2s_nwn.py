"""Ag2S Nanowire Network — physically-derived simulated DUT (SM.2 of the paper).

Represented in dual form: nanowires are neurons, junctions between them
(electrochemically actuated atomic switches) are resistive edges on an
undirected random graph. At each timestep, silver-sulfide bridges grow or
annihilate at each junction based on the trend of the local junction voltage;
once a bridge spans the gap the junction switches ON. Node voltages are
solved simultaneously via Kirchhoff's Current Law (KCL) using a
Moore-Penrose least-squares solve.

This substrate is a more physically realistic PRC candidate than the LI-ESN,
at the cost of a per-timestep O(N_wires^3) voltage solve.
"""

import math

import numpy as np

from prc_toolkit.dut.base import BaseDUT

# Physical constants (empirical, from Hasegawa2011 and Nayak2011). These are
# fixed physical values, not tunable parameters.
L_GAP = 1.53e-9      # Gap distance: 1.53 nm (Ag2S gap-type atomic switch).
G_ON = 1 / 12500     # ON conductance: 1/12.5 kOhm ~= quantum conductance G0.
G_OFF = 1e-9         # OFF conductance: ~1/100 MOhm (leakage).
BETA = 32.9          # Exponential voltage sensitivity (V^-1), from Nayak2011.
V_THRESH = 0.01      # Switching threshold voltage (V). Tuned low for network
                     # operation, since junction voltages are divided across
                     # many nodes (see prompts/00b_ag2s_nwn.md).

N_WIRES_DEFAULT = 20  # Practical maximum for real-time simulation performance.

# Single source of truth for the recommended Ag2S-NWN configuration. Notebooks
# reference this constant (rather than duplicating literal values) when
# offering Ag2S-NWN as an alternative DUT, so there is exactly one place to
# update if the recommendation changes. Differs from the bare constructor
# defaults only in sigma_process/sigma_measure (nonzero here, matching the
# noise settings used for LI-ESN elsewhere in the toolkit).
RECOMMENDED_PARAMS = dict(
    N_wires=20,
    N_in=1,
    N_out=3,
    connectivity=0.3,
    sigma_process=0.01,
    sigma_measure=0.005,
    growth_rate=1e-10,
    seed=42,
)


class Ag2SNWN(BaseDUT):
    """Ag2S-NWN simulated DUT.

    Input to step()/run(): vector, shape (N_in,) — non-negative electrode
    voltages (Volts). The substrate is physically rectifying; bipolar signals
    from the generators should be passed through bias_positive() first.
    Output of step()/run(): vector, shape (N_out,) — voltages at output nodes.

    Node N_in is reserved as the ground reference electrode: it is held at
    0V in the voltage solve and is neither driven nor read.
    """

    def __init__(
        self,
        N_wires=N_WIRES_DEFAULT,
        N_in=1,
        N_out=3,
        connectivity=0.3,
        dt=None,
        sigma_process=0.0,
        sigma_measure=0.0,
        growth_rate=1e-10,
        seed=42,
    ):
        if dt is None:
            from prc_toolkit.config import DT

            dt = DT

        assert N_in + N_out < N_wires, "N_in + N_out must be strictly less than N_wires to reserve a ground node"

        self.N_wires = N_wires
        self.N_in = N_in
        self.N_out = N_out
        self.connectivity = connectivity
        self.dt = dt
        self.sigma_process = sigma_process
        self.sigma_measure = sigma_measure
        self.growth_rate = growth_rate
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        upper = self.rng.random((N_wires, N_wires)) < connectivity
        A = np.triu(upper, k=1)
        A = A + A.T
        self.A = A.astype(float)

        D = np.diag(self.A.sum(axis=1))
        lap = D - self.A
        rank = np.linalg.matrix_rank(lap)
        if rank < N_wires - 1:
            print(
                f"WARNING: Ag2SNWN graph is disconnected (rank={rank}, "
                f"N_wires={N_wires}). Ground node is node {N_in}. "
                f"Increase connectivity or use a different seed. "
                f"Voltage solve may be unreliable."
            )

        self.in_nodes = np.arange(N_in)
        self.out_nodes = np.arange(N_wires - N_out, N_wires)
        self.ground_node = N_in  # first non-input node; held at 0 V as voltage reference

        for node in self.out_nodes:
            if self.A[node, :].sum() == 0:
                print(
                    f"WARNING: output node {node} has no edges in A. It "
                    f"will always output 0V. Use a different seed or "
                    f"increase connectivity."
                )

        self.L = np.zeros((N_wires, N_wires))
        self.G = np.where(self.A == 1, G_OFF, 0.0)
        self.V = np.zeros(N_wires)
        self.V_prev = np.zeros(N_wires)

    def reset(self, x0=None):
        """
        Reset bridge lengths to zero (all bridges OFF) and node voltages to
        zero. x0 is accepted for interface compatibility but ignored — the
        Ag2S-NWN state is fully described by L (and derived G), not a hidden
        state vector.
        """
        if x0 is not None:
            print("Ag2SNWN: x0 ignored; state reset to L=0 (all bridges OFF).")
        self.L = np.zeros((self.N_wires, self.N_wires))
        self.G = np.where(self.A == 1, G_OFF, 0.0)
        self.V = np.zeros(self.N_wires)
        self.V_prev = np.zeros(self.N_wires)
        self.V[self.ground_node] = 0.0
        self.V_prev[self.ground_node] = 0.0

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        u: vector, shape (N_in,) — non-negative input electrode voltages (V).
        Returns h: vector, shape (N_out,) — output electrode voltages (V).
        """
        u = np.atleast_1d(u)

        # Step 1: apply input voltages.
        self.V[self.in_nodes] = u

        # Step 2: build nodal conductance matrix (graph Laplacian).
        G_node = np.zeros((self.N_wires, self.N_wires))
        for i in range(self.N_wires):
            for j in range(i + 1, self.N_wires):
                if self.A[i, j] == 1:
                    g = self.G[i, j]
                    G_node[i, i] += g
                    G_node[j, j] += g
                    G_node[i, j] -= g
                    G_node[j, i] -= g

        # Step 3: solve for unknown node voltages via KCL. Node `ground_node`
        # is held at 0V alongside the driven input nodes, giving the reduced
        # Laplacian a genuine reference distinct from the input — without it,
        # a single voltage source with no ground yields the degenerate
        # solution where every floating node equals the input voltage and no
        # current ever flows (see build_notes.md).
        known_nodes = np.append(self.in_nodes, self.ground_node)
        known_voltages = np.append(u, 0.0)
        unknown_nodes = np.array([i for i in range(self.N_wires) if i not in set(known_nodes.tolist())])

        G_hat = G_node[np.ix_(unknown_nodes, unknown_nodes)]
        I_input = -G_node[np.ix_(unknown_nodes, known_nodes)] @ known_voltages
        V_hat = np.linalg.lstsq(G_hat, I_input, rcond=None)[0]
        self.V[unknown_nodes] = V_hat
        self.V[self.ground_node] = 0.0

        # Step 4: update bridge lengths — signed bidirectional dynamics.
        for i in range(self.N_wires):
            for j in range(i + 1, self.N_wires):
                if self.A[i, j] != 1:
                    continue

                V_junc = abs(self.V[i] - self.V[j])
                V_junc_prev = abs(self.V_prev[i] - self.V_prev[j])
                dV_junc = V_junc - V_junc_prev

                rate = self.growth_rate * math.exp(min(BETA * V_junc, 20.0)) * self.dt

                if V_junc < V_THRESH:
                    dl = -rate
                elif dV_junc > 0:
                    dl = rate
                elif dV_junc < 0:
                    dl = -rate
                else:
                    dl = 0.0

                if self.sigma_process > 0:
                    dl += self.rng.normal(0, self.sigma_process) * self.dt

                self.L[i, j] = np.clip(self.L[i, j] + dl, 0.0, L_GAP)
                self.L[j, i] = self.L[i, j]

                if (self.L[i, j] >= L_GAP) and (V_junc >= V_THRESH):
                    self.G[i, j] = G_ON
                    self.G[j, i] = G_ON
                else:
                    self.G[i, j] = G_OFF
                    self.G[j, i] = G_OFF

        self.V_prev = self.V.copy()

        # Step 5: return output with optional measurement noise.
        h = self.V[self.out_nodes].copy()
        if self.sigma_measure > 0:
            h += self.rng.normal(0, self.sigma_measure, size=h.shape)
        return h
