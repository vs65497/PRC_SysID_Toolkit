"""Smoke tests for prc_toolkit.dut.factory."""

import numpy as np
import pytest

from prc_toolkit.dut.ag2s_nwn import Ag2SNWN
from prc_toolkit.dut.factory import make_dut, prepare_input
from prc_toolkit.dut.liesn import LIESN


def test_make_dut_liesn():
    dut = make_dut("liesn", dict(N_x=10, N_h=2, seed=1))
    assert isinstance(dut, LIESN)


def test_make_dut_ag2s_nwn():
    dut = make_dut("ag2s_nwn", dict(N_wires=6, N_in=1, N_out=2, seed=1))
    assert isinstance(dut, Ag2SNWN)


def test_make_dut_hardware_returns_none():
    assert make_dut("hardware", {}) is None


def test_make_dut_unknown_raises():
    with pytest.raises(ValueError):
        make_dut("bogus", {})


def test_prepare_input_identity_for_liesn():
    u = np.array([-1.0, 0.0, 1.0])
    out = prepare_input(u, amplitude=1.0, dut_model="liesn")
    np.testing.assert_array_equal(out, u)


def test_prepare_input_bias_positive_for_ag2s_nwn():
    u = np.array([-1.0, 0.0, 1.0])
    out = prepare_input(u, amplitude=1.0, dut_model="ag2s_nwn")
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])
