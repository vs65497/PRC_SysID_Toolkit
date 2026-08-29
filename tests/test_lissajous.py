"""Smoke tests for prc_toolkit.analysis.lissajous."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from prc_toolkit.analysis.lissajous import (
    fingerprint_grid,
    lissajous_io,
    lissajous_residual,
    lissajous_response,
    lissajous_state,
)


def _make_series(T=100):
    t = np.linspace(0, 1, T)
    u = np.sin(2 * np.pi * 3 * t)
    h = 0.5 * u + 0.1 * u**2
    return u, h


def test_lissajous_response_runs():
    _, h = _make_series()
    ax = lissajous_response(h, label="test")
    assert ax is not None


def test_lissajous_io_runs():
    u, h = _make_series()
    ax = lissajous_io(h, u)
    assert ax is not None


def test_lissajous_residual_runs():
    u, h = _make_series()
    ax = lissajous_residual(h, u)
    assert ax is not None


def test_lissajous_state_runs():
    _, wx = _make_series()
    ax = lissajous_state(wx)
    assert ax is not None


def test_fingerprint_grid_without_state():
    sweep_results = []
    for i, amp_dB in enumerate([-20, -10, 0]):
        u, h = _make_series()
        sweep_results.append(
            {"amplitude_dB": amp_dB, "u": u, "h_scalar": h, "Wx_scalar": None}
        )
    fig = fingerprint_grid(sweep_results)
    assert fig is not None
    assert len(fig.axes) >= 3


def test_fingerprint_grid_with_state():
    sweep_results = []
    for amp_dB in [-20, -10, 0]:
        u, h = _make_series()
        sweep_results.append(
            {"amplitude_dB": amp_dB, "u": u, "h_scalar": h, "Wx_scalar": h.copy()}
        )
    fig = fingerprint_grid(sweep_results)
    assert fig is not None
