"""Settling detection for driving a DUT to a reproducible operating state."""

import numpy as np

from prc_toolkit.config import DT, SETTLE_EPS, SETTLE_WINDOW


def is_settled(h_history, eps=SETTLE_EPS, window=SETTLE_WINDOW) -> bool:
    """
    Given recent output history `h_history` of shape (window, N_h) (vector
    outputs), compute |delta h_bar / delta t| as the change in mean output
    magnitude between the first and second half of the window, divided by the
    elapsed time between those halves. Returns True if below `eps`.
    """
    h_history = np.asarray(h_history)
    magnitude = np.linalg.norm(h_history, axis=1)

    half = len(magnitude) // 2
    h_bar_first = np.mean(magnitude[:half])
    h_bar_second = np.mean(magnitude[half:])

    elapsed = half * DT
    if elapsed <= 0:
        return True

    rate = abs(h_bar_second - h_bar_first) / elapsed
    return bool(rate < eps)


def run_until_settled(dut, u_seq, max_samples=10000, eps=SETTLE_EPS, window=SETTLE_WINDOW) -> np.ndarray:
    """
    Drive `dut` with `u_seq` (vector input, shape (T, N_u)) repeated as needed
    until settled (per `is_settled` over the most recent `window` samples) or
    `max_samples` is reached.

    Returns H: vector output history, shape (T_actual, N_h).
    """
    u_seq = np.asarray(u_seq)
    T_template = u_seq.shape[0]

    h_list = []
    n = 0
    while n < max_samples:
        u = u_seq[n % T_template]
        h = dut.step(u)
        h_list.append(h)
        n += 1

        if n >= window and is_settled(np.array(h_list[-window:]), eps=eps, window=window):
            return np.array(h_list)

    print(
        f"run_until_settled: max_samples reached before settling criterion met "
        f"(eps={eps}). Proceeding with unsettled state. Consider increasing "
        f"max_samples or adjusting input."
    )
    return np.array(h_list)
