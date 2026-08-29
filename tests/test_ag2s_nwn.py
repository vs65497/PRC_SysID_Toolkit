"""Smoke tests for prc_toolkit.dut.ag2s_nwn, per prompts/00b_ag2s_nwn.md and
prompts/00c_ag2s_nwn_ground_fix.md."""

import numpy as np

from prc_toolkit.dut.ag2s_nwn import L_GAP, Ag2SNWN, RECOMMENDED_PARAMS


def test_ag2s_nwn_shapes():
    dut = Ag2SNWN(N_wires=10, N_in=1, N_out=2, seed=0)
    # Inputs must be non-negative voltages. Use a small positive range for smoke test.
    u_seq = np.random.default_rng(1).uniform(0.0, 0.5, size=(50, 1))
    H = dut.run(u_seq)
    assert H.shape == (50, 2), f"Expected (50,2), got {H.shape}"
    assert not np.any(np.isnan(H)), "NaN in output"
    # Verify bridge lengths stay within physical bounds [0, L_GAP]
    assert np.all(dut.L >= 0.0), "Bridge length went negative"
    assert np.all(dut.L <= L_GAP + 1e-15), "Bridge length exceeded L_GAP"
    print("Ag2SNWN smoke test passed.")


def test_ag2s_nwn_ground_node_gives_voltage_gradient():
    """Verification 1 from prompts/00c_ag2s_nwn_ground_fix.md: with N_in=1,
    node voltages must no longer collapse to the trivial all-equal-to-input
    solution (the floating-ground bug documented in build_notes.md)."""
    dut = Ag2SNWN(N_wires=20, N_in=1, N_out=3, connectivity=0.3, seed=42)
    dut.step(np.array([0.5]))
    assert not np.allclose(dut.V, 0.5), "Voltages collapsed to the input value (floating-ground bug)"
    assert len(np.unique(np.round(dut.V, 6))) > 1, "Expected a genuine voltage gradient across nodes"


def test_ag2s_nwn_ground_node_pinned_at_zero():
    """Verification 2 from prompts/00c_ag2s_nwn_ground_fix.md."""
    dut = Ag2SNWN(N_wires=20, N_in=1, N_out=3, connectivity=0.3, seed=42)
    dut.step(np.array([0.5]))
    assert dut.V[dut.ground_node] == 0.0
    dut.reset()
    assert dut.V[dut.ground_node] == 0.0
    assert dut.V_prev[dut.ground_node] == 0.0


def test_ag2s_nwn_bridges_show_mixed_states_after_run():
    """Verification 3 from prompts/00c_ag2s_nwn_ground_fix.md: after a
    multi-step sinusoidal run, bridge lengths should show a genuine mix of
    values, not be stuck at all-zero (no switching) as under the original
    floating-ground bug."""
    dut = Ag2SNWN(N_wires=20, N_in=1, N_out=3, connectivity=0.3, seed=42)
    t = np.arange(100)
    u = np.clip(0.3 * (np.sin(2 * np.pi * t / 100) + 1), 0, None)
    dut.reset()
    for ut in u:
        dut.step(np.array([ut]))

    L_active = dut.L[dut.A == 1]
    assert L_active.max() > 0.0, "No bridge growth occurred at all"
    assert L_active.min() < L_GAP, "Every active junction saturated to L_GAP"
    n_distinct = len(np.unique(np.round(L_active / L_GAP, 3)))
    assert n_distinct > 1, "Bridge lengths show no variation across junctions"


def test_ag2s_nwn_recommended_params_construct_and_run():
    """RECOMMENDED_PARAMS is the single source of truth notebooks reference
    (rather than duplicating literal config dicts) when offering Ag2S-NWN as
    an alternative DUT — it must actually construct and run cleanly."""
    dut = Ag2SNWN(**RECOMMENDED_PARAMS)
    u_seq = np.random.default_rng(1).uniform(0.0, 0.5, size=(50, 1))
    H = dut.run(u_seq)
    assert H.shape == (50, RECOMMENDED_PARAMS["N_out"])
    assert not np.any(np.isnan(H))
