"""Smoke tests for prc_toolkit.dut.base."""

import numpy as np

from prc_toolkit.dut.base import BaseDUT, to_input_seq


class _EchoDUT(BaseDUT):
    """Minimal concrete DUT for exercising the base run() loop."""

    def __init__(self, n_h=2):
        self.n_h = n_h
        self.x = None

    def reset(self, x0=None):
        self.x = 0.0 if x0 is None else float(x0)

    def step(self, u):
        self.x = self.x + u[0]
        return np.full(self.n_h, self.x)


def test_to_input_seq_scalar():
    u = np.arange(5.0)
    out = to_input_seq(u)
    assert out.shape == (5, 1)
    np.testing.assert_array_equal(out[:, 0], u)


def test_to_input_seq_already_vector():
    u = np.zeros((5, 3))
    out = to_input_seq(u)
    assert out is u


def test_base_run_loop_shapes():
    dut = _EchoDUT(n_h=3)
    u_seq = to_input_seq(np.ones(10))
    H = dut.run(u_seq)
    assert H.shape == (10, 3)
