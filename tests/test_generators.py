"""Smoke tests for prc_toolkit.signals.generators."""

import numpy as np

from prc_toolkit.config import FS, V_MAX
from prc_toolkit.signals.generators import (
    bias_positive,
    delayed_spike_train,
    dc_near_zero,
    iid_uniform,
    multisine,
    poisson_spike_train,
    sine_sweep,
)


def test_multisine_shape_and_bins():
    T_sec = 4
    u = multisine(T_sec, amplitude=1.0)
    assert u.shape == (T_sec * FS,)
    # Energy should land on bins 1, 3, 7, 11 Hz.
    spec = np.abs(np.fft.rfft(u))
    freqs = np.fft.rfftfreq(u.shape[0], d=1.0 / FS)
    for f in (1, 3, 7, 11):
        idx = np.argmin(np.abs(freqs - f))
        assert spec[idx] > 0.5 * spec.max()


def test_iid_uniform_bounds_and_seed():
    u = iid_uniform(2, amplitude=0.5, seed=0)
    assert u.shape == (2 * FS,)
    assert np.all(u >= -0.5) and np.all(u <= 0.5)
    u2 = iid_uniform(2, amplitude=0.5, seed=0)
    np.testing.assert_array_equal(u, u2)


def test_dc_near_zero_default():
    u = dc_near_zero(1)
    assert u.shape == (FS,)
    np.testing.assert_allclose(u, 0.01 * V_MAX)


def test_sine_sweep_amplitudes():
    steps = sine_sweep(1, amplitude=1.0, n_steps=5)
    assert len(steps) == 5
    amps = [a for a, _ in steps]
    np.testing.assert_allclose(amps[-1], 1.0)
    np.testing.assert_allclose(amps[0], 10 ** (-20 / 20))
    for _, sig in steps:
        assert sig.shape == (FS,)


def test_poisson_spike_train_shape():
    u = poisson_spike_train(5, rate_hz=2.0, amplitude=1.0, seed=0)
    assert u.shape == (5 * FS,)
    assert np.any(u > 0)
    assert set(np.unique(u)) <= {0.0, 1.0}


def test_delayed_spike_train_shifts():
    u = np.zeros(100)
    u[10:12] = 1.0
    shifted = delayed_spike_train(u, spike_idx=10, delay_samples=20)
    assert np.all(shifted[10:12] == 0.0)
    assert np.all(shifted[30:32] == 1.0)


def test_bias_positive_maps_range():
    u = np.array([-1.0, 0.0, 1.0])
    biased = bias_positive(u, amplitude=1.0)
    np.testing.assert_allclose(biased, [0.0, 0.5, 1.0])
