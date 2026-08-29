"""Smoke test for prc_toolkit.config."""

from prc_toolkit.config import DT, FS, N_TRIALS, RESULTS_DIR, SEED, SETTLE_EPS, SETTLE_WINDOW, V_MAX


def test_config_constants_present_and_sane():
    assert FS > 0
    assert DT == 1.0 / FS
    assert V_MAX > 0
    assert SETTLE_EPS > 0
    assert SETTLE_WINDOW > 0
    assert N_TRIALS > 0
    assert isinstance(RESULTS_DIR, str)
    assert isinstance(SEED, int)
