"""Input signal generators. All functions return scalar time series, shape (T,)."""

import numpy as np

from prc_toolkit.config import FS, V_MAX


def multisine(duration, amplitude=1.0, fs=FS) -> np.ndarray:
    """
    Sum of sines at 1, 3, 7, 11 Hz — integer multiples of 1 Hz, so they land on
    exact FFT bins when duration is an integer number of seconds.

    Returns: scalar array, shape (T,).
    """
    T = int(duration * fs)
    t = np.arange(T) / fs
    freqs = (1.0, 3.0, 7.0, 11.0)
    u = np.zeros(T)
    for f in freqs:
        u += np.sin(2 * np.pi * f * t)
    return amplitude * u


def iid_uniform(duration, amplitude=1.0, fs=FS, seed=None) -> np.ndarray:
    """
    i.i.d. samples from U[-amplitude, amplitude]. Used for Section 3 tests
    (consistency, MC, IPC, PSD).

    Returns: scalar array, shape (T,).
    """
    T = int(duration * fs)
    rng = np.random.default_rng(seed)
    return rng.uniform(-amplitude, amplitude, size=T)


def dc_near_zero(duration, amplitude=None, fs=FS) -> np.ndarray:
    """
    Constant signal at `amplitude`. Default amplitude is 1% of V_MAX. Used as the
    probe signal in the FMP/ESP test after initial settling.

    Returns: scalar array, shape (T,).
    """
    if amplitude is None:
        amplitude = 0.01 * V_MAX
    T = int(duration * fs)
    return np.full(T, amplitude)


def sine_sweep(duration, amplitude, n_steps, fs=FS):
    """
    Returns a list of (amplitude_linear, signal) tuples. Amplitudes are spaced
    evenly in dB from -20 dB to 0 dB relative to `amplitude` (ceiling, typically
    V_MAX). Each signal is a pure 1 Hz sine: A * sin(2*pi*1*t).

    Used for the Section 1.2 safe region sweep.

    Returns: list[tuple[float, np.ndarray]] — signal arrays have shape (T,).
    """
    T = int(duration * fs)
    t = np.arange(T) / fs
    dB_steps = np.linspace(-20, 0, n_steps)
    amplitudes_linear = amplitude * 10 ** (dB_steps / 20)
    return [(A, A * np.sin(2 * np.pi * 1.0 * t)) for A in amplitudes_linear]


def poisson_spike_train(duration, rate_hz, amplitude=1.0, pulse_width_samples=2, fs=FS, seed=None) -> np.ndarray:
    """
    Discrete-time approximation of a Poisson spike train. Inter-spike intervals
    are drawn from Exponential(1/rate_hz) and converted to sample indices; each
    spike sets `pulse_width_samples` consecutive samples to `amplitude`.

    Returns: scalar array, shape (T,).
    """
    T = int(duration * fs)
    rng = np.random.default_rng(seed)
    u = np.zeros(T)

    t_sample = 0.0
    while True:
        isi = rng.exponential(1.0 / rate_hz)
        t_sample += isi
        idx = int(round(t_sample * fs))
        if idx >= T:
            break
        end = min(idx + pulse_width_samples, T)
        u[idx:end] = amplitude

    return u


def delayed_spike_train(u_template, spike_idx, delay_samples) -> np.ndarray:
    """
    Given a template spike train `u_template` (scalar, shape (T,)), shift the
    spike located at sample index `spike_idx` by `delay_samples` (positive =
    later). Returns the modified copy. Used to create u^(1) from u^(0) in the
    Separation Property test.

    Returns: scalar array, shape (T,).
    """
    u_template = np.asarray(u_template)
    T = u_template.shape[0]
    u_out = u_template.copy()

    # Find the extent of the spike starting at spike_idx (run of identical
    # nonzero values starting there).
    if u_template[spike_idx] == 0:
        return u_out

    value = u_template[spike_idx]
    end = spike_idx
    while end < T and u_template[end] == value:
        end += 1
    width = end - spike_idx

    u_out[spike_idx:end] = 0.0

    new_start = spike_idx + delay_samples
    new_end = min(new_start + width, T)
    if new_start < T:
        u_out[max(new_start, 0):new_end] = value

    return u_out


def bias_positive(u_seq, amplitude) -> np.ndarray:
    """
    Map a bipolar signal u_seq in [-amplitude, amplitude] to [0, amplitude].
    Used when driving the Ag2S-NWN, which assumes non-negative input voltages
    (the substrate is physically rectifying).

    Returns: array, same shape as u_seq.
    """
    return (np.asarray(u_seq) + amplitude) / 2.0
