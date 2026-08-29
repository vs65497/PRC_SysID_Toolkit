"""Smoke tests for prc_toolkit.dut.liesn."""

import numpy as np

from prc_toolkit.config import FS
from prc_toolkit.dut.base import to_input_seq
from prc_toolkit.dut.liesn import LIESN


def test_liesn_spectral_radius():
    dut = LIESN(N_x=50, spectral_radius=1.1, seed=42)
    rho = np.max(np.abs(np.linalg.eigvals(dut.W)))
    np.testing.assert_allclose(rho, 1.1, rtol=1e-6)


def test_liesn_run_shapes_one_second():
    dut = LIESN(N_x=50, N_u=1, N_h=5, seed=42)
    T = FS  # 1 second
    u_seq = to_input_seq(np.sin(2 * np.pi * 1.0 * np.arange(T) / FS))
    H = dut.run(u_seq)
    assert H.shape == (T, 5)
    assert np.all(np.isfinite(H))


def test_liesn_reset_reproducible():
    dut = LIESN(seed=1)
    u_seq = to_input_seq(np.ones(20))
    H1 = dut.run(u_seq)
    H2 = dut.run(u_seq)
    np.testing.assert_array_equal(H1, H2)


def test_liesn_step_output_shape():
    dut = LIESN(N_u=1, N_h=5, seed=0)
    dut.reset()
    h = dut.step(np.array([0.1]))
    assert h.shape == (5,)
