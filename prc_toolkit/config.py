"""Global simulation parameters shared by every module and notebook."""

FS = 100            # Sampling rate in Hz. Satisfies Nyquist for multisine max freq of 11 Hz (2*11=22 Hz).
DT = 1.0 / FS       # Timestep in seconds.
V_MAX = 1.0         # Hard ceiling on input amplitude (hardware safety limit). Normalized units.
SETTLE_EPS = 1e-4   # Threshold for settling detection: |Δh̄/Δt| < SETTLE_EPS.
SETTLE_WINDOW = 50  # Number of samples over which to compute the settling criterion.
N_TRIALS = 2        # Number of trials for tests requiring repeated runs (FMP/ESP, consistency, CLE).
RESULTS_DIR = "results/"
SEED = 42           # Default random seed, shared across notebooks for reproducibility.
