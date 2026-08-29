"""Abstract DUT (device under test) interface.

I/O convention (enforced throughout the toolkit):
  - Input `u` per timestep is a vector of shape (N_u,). Scalar signals of shape (T,)
    from the generators are reshaped to (T, 1) via `to_input_seq()` before being
    passed to `run()`.
  - Output `h` from `step()` is always a vector of shape (N_h,), one value per
    output electrode. Callers needing a scalar take `np.linalg.norm(h, axis=-1)`.
"""

from abc import ABC, abstractmethod

import numpy as np


def to_input_seq(u):
    """
    Ensure input sequence has shape (T, N_u).
    Accepts shape (T,) and returns shape (T, 1).
    Accepts shape (T, N_u) and returns unchanged.
    """
    u = np.asarray(u)
    if u.ndim == 1:
        return u[:, np.newaxis]
    return u


class BaseDUT(ABC):
    """Abstract base class for every device under test (simulated or hardware)."""

    @abstractmethod
    def reset(self, x0=None):
        """
        Reset internal state.

        x0: vector initial condition, or None to reset to zeros.

        Note for hardware implementations: physical reset may not be instantaneous.
        Some substrates (e.g. a cup of coffee, mechanical tensegrity) require
        entrainment on a forcing periodic sequence to reach a reproducible rest
        state. For this iteration, all DUTs are software simulations and reset
        is exact.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Apply a single input sample.

        u: vector, shape (N_u,) — input electrode voltages. For single-input
           DUTs, shape is (1,). Use np.atleast_1d(u_scalar) to convert.
        Returns h: vector, shape (N_h,) — output electrode vector. Callers who
           need a scalar output take np.linalg.norm(h).
        """
        raise NotImplementedError

    def run(self, u_seq: np.ndarray, x0=None) -> np.ndarray:
        """
        Apply a sequence of inputs.

        u_seq: array, shape (T, N_u) — input sequence. For single-input DUTs,
           shape is (T, 1). Callers generating scalar signals of shape (T,)
           should reshape via `to_input_seq()` first.
        Returns H: array, shape (T, N_h) — output at each timestep.
        Resets state to x0 before running.
        """
        self.reset(x0)
        u_seq = np.asarray(u_seq)
        T = u_seq.shape[0]
        H = None
        for t in range(T):
            h = self.step(u_seq[t])
            if H is None:
                H = np.empty((T, h.shape[0]), dtype=h.dtype)
            H[t] = h
        return H
