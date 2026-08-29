"""Smoke tests for prc_toolkit.utils.settling."""

import numpy as np

from prc_toolkit.dut.base import BaseDUT
from prc_toolkit.utils.settling import is_settled, run_until_settled


class _ConstantDUT(BaseDUT):
    """Immediately settles: output is constant regardless of input."""

    def reset(self, x0=None):
        pass

    def step(self, u):
        return np.array([1.0, 1.0])


class _DriftingDUT(BaseDUT):
    """Never settles within a reasonable sample budget: output keeps growing."""

    def __init__(self):
        self.t = 0

    def reset(self, x0=None):
        self.t = 0

    def step(self, u):
        self.t += 1
        return np.array([float(self.t)])


def test_is_settled_true_for_constant_history():
    h_history = np.ones((50, 2))
    assert is_settled(h_history) is True


def test_is_settled_false_for_ramping_history():
    h_history = np.arange(50.0).reshape(-1, 1)
    assert is_settled(h_history, eps=1e-4) is False


def test_run_until_settled_constant_dut_settles_quickly():
    dut = _ConstantDUT()
    u_seq = np.zeros((10, 1))
    H = run_until_settled(dut, u_seq, max_samples=1000)
    assert H.shape[1] == 2
    assert H.shape[0] < 1000


def test_run_until_settled_drifting_dut_hits_max_samples(capsys):
    dut = _DriftingDUT()
    u_seq = np.zeros((10, 1))
    H = run_until_settled(dut, u_seq, max_samples=200)
    assert H.shape[0] == 200
    captured = capsys.readouterr()
    assert "max_samples reached" in captured.out
