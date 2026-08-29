"""Leaky-Integrator Echo State Network — primary simulated DUT for Sections 1-3.

State update:
    x(n) = (1 - alpha) * x(n-1) + alpha * f(W_in @ [1; u(n)] + W @ x(n-1) + noise_process)
    h(n) = W_out @ x(n) + noise_measure

W_out is fixed at construction (the DUT's internal readout) and is never trained;
only an external readout W_ext (see analysis/readout.py) is fit against h(n).
"""

import numpy as np

from prc_toolkit.dut.base import BaseDUT


class LIESN(BaseDUT):
    """Leaky-integrator ESN simulated DUT.

    Input to step()/run(): vector, shape (N_u,) per timestep.
    Output of step()/run(): vector, shape (N_h,) per timestep.
    """

    def __init__(
        self,
        N_x=50,
        N_u=1,
        N_h=5,
        alpha=0.3,
        spectral_radius=1.1,
        sigma_process=0.0,
        sigma_measure=0.0,
        seed=42,
    ):
        self.N_x = N_x
        self.N_u = N_u
        self.N_h = N_h
        self.alpha = alpha
        self.spectral_radius = spectral_radius
        self.sigma_process = sigma_process
        self.sigma_measure = sigma_measure
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        self.W_in = self.rng.uniform(-1.0, 1.0, size=(N_x, 1 + N_u))

        W = self.rng.normal(0.0, 1.0, size=(N_x, N_x))
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / rho)

        self.W_out = self.rng.uniform(-1.0, 1.0, size=(N_h, N_x))

        self.x = np.zeros(N_x)

    def reset(self, x0=None):
        """Reset reservoir state. x0: vector shape (N_x,), or None for zeros."""
        self.x = np.zeros(self.N_x) if x0 is None else np.asarray(x0, dtype=float).copy()

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        u: vector, shape (N_u,).
        Returns h: vector, shape (N_h,).
        """
        u = np.atleast_1d(u)
        u_ext = np.concatenate(([1.0], u))

        if self.sigma_process > 0.0:
            noise_process = self.rng.normal(0.0, self.sigma_process, size=self.N_x)
        else:
            noise_process = 0.0

        pre_activation = self.W_in @ u_ext + self.W @ self.x + noise_process
        self.x = (1.0 - self.alpha) * self.x + self.alpha * np.tanh(pre_activation)

        if self.sigma_measure > 0.0:
            noise_measure = self.rng.normal(0.0, self.sigma_measure, size=self.N_h)
        else:
            noise_measure = 0.0

        h = self.W_out @ self.x + noise_measure
        return h
