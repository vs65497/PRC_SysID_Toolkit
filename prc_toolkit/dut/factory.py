"""DUT factory — centralised instantiation and input preparation.

Notebooks import make_dut() and prepare_input() from here instead of
copy-pasting the instantiation block and prepare_input() helper.
"""

import numpy as np

from prc_toolkit.signals.generators import bias_positive


def make_dut(dut_model: str, dut_params: dict):
    """
    Instantiate and return a DUT from a model name and parameter dict.

    dut_model: one of "liesn", "ag2s_nwn", "hardware".
    dut_params: dict of constructor kwargs for the chosen model.
                Ignored when dut_model == "hardware".

    Returns the instantiated DUT, or None for hardware mode.
    """
    if dut_model == "liesn":
        from prc_toolkit.dut.liesn import LIESN
        return LIESN(**dut_params)

    elif dut_model == "ag2s_nwn":
        from prc_toolkit.dut.ag2s_nwn import Ag2SNWN
        return Ag2SNWN(**dut_params)

    elif dut_model == "hardware":
        return None  # data loaded from filesystem; no simulated DUT

    else:
        raise ValueError(
            f"Unknown DUT_MODEL '{dut_model}'. "
            f"Supported values: 'liesn', 'ag2s_nwn', 'hardware'."
        )


def prepare_input(u_seq: np.ndarray, amplitude: float, dut_model: str) -> np.ndarray:
    """
    Apply bias_positive() for rectifying substrates (Ag2S-NWN); identity for
    others. `amplitude` must match the true amplitude bound u_seq was
    generated with, since bias_positive() needs it to correctly map
    [-amplitude, amplitude] -> [0, amplitude]. Only apply this to genuinely
    bipolar signals -- already non-negative signals should be left unwrapped.

    u_seq: array, shape (T,) or (T, N_u).
    Returns: array, same shape as u_seq.
    """
    if dut_model == "ag2s_nwn":
        return bias_positive(u_seq, amplitude)
    return u_seq
