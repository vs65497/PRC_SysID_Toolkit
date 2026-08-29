"""Lissajous plot rendering for the visual fingerprint test (Section 2.1)."""

import matplotlib.pyplot as plt
import numpy as np


def lissajous_response(h_seq, label="", color=None, ax=None):
    """
    Plot type A: dh/dt vs h(t).

    h_seq: scalar time series, shape (T,). Caller passes np.linalg.norm(H, axis=1)
    if H is a matrix.

    Returns the axes object.
    """
    if ax is None:
        _, ax = plt.subplots()
    dh = np.gradient(h_seq)
    ax.plot(h_seq, dh, label=label, color=color)
    ax.set_xlabel("h(t)")
    ax.set_ylabel("dh/dt")
    return ax


def lissajous_io(h_seq, u_seq, label="", color=None, ax=None):
    """
    Plot type B: h(t) vs u(t).

    h_seq, u_seq: scalar time series, shape (T,).

    Returns the axes object.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(u_seq, h_seq, label=label, color=color)
    ax.set_xlabel("u(t)")
    ax.set_ylabel("h(t)")
    return ax


def lissajous_residual(h_seq, u_seq, label="", color=None, ax=None):
    """
    Plot type C: (h(t) - u(t)) vs u(t).

    h_seq, u_seq: scalar time series, shape (T,).

    Returns the axes object.
    """
    if ax is None:
        _, ax = plt.subplots()
    residual = h_seq - u_seq
    ax.plot(u_seq, residual, label=label, color=color)
    ax.set_xlabel("u(t)")
    ax.set_ylabel("h(t) - u(t)")
    return ax


def lissajous_state(Wx_seq, label="", color=None, ax=None):
    """
    Plot type D: d(Wx)/dt vs Wx(t). Simulation only — not available for
    hardware DUTs.

    Wx_seq: scalar time series, shape (T,). Caller computes W @ x and passes
    np.linalg.norm of that.

    Returns the axes object.
    """
    if ax is None:
        _, ax = plt.subplots()
    dWx = np.gradient(Wx_seq)
    ax.plot(Wx_seq, dWx, label=label, color=color)
    ax.set_xlabel("Wx(t)")
    ax.set_ylabel("d(Wx)/dt")
    return ax


def fingerprint_grid(sweep_results, titles=("Response LP", "I/O LP", "Residual LP", "State LP")):
    """
    Compose the 4-column visual fingerprint grid (Figure 7 style).

    sweep_results: list of dicts, one per amplitude step, each with keys
    {'amplitude_dB': float, 'u': array, 'h_scalar': array, 'Wx_scalar': array or None}.

    Trajectories are colored by amplitude in dB using a yellow-to-purple
    colormap. Column D (State LP) is rendered only when Wx_scalar is not None
    for all sweep entries — it requires access to the hidden reservoir state,
    unavailable for hardware DUTs.

    Returns the figure.
    """
    has_state = all(entry.get("Wx_scalar") is not None for entry in sweep_results)
    n_cols = 4 if has_state else 3

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    amplitudes_dB = np.array([entry["amplitude_dB"] for entry in sweep_results])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=amplitudes_dB.min(), vmax=amplitudes_dB.max())

    for entry in sweep_results:
        color = cmap(norm(entry["amplitude_dB"]))
        u = entry["u"]
        h_scalar = entry["h_scalar"]

        lissajous_response(h_scalar, color=color, ax=axes[0])
        lissajous_io(h_scalar, u, color=color, ax=axes[1])
        lissajous_residual(h_scalar, u, color=color, ax=axes[2])
        if has_state:
            lissajous_state(entry["Wx_scalar"], color=color, ax=axes[3])

    for ax, title in zip(axes, titles[:n_cols]):
        ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Input Amplitude (dB)")

    return fig
