"""Observational orthogonality metric."""

import numpy as np


def observational_orthogonality(H) -> float:
    """
    Mean pairwise (1 - |cosine similarity|) across all pairs i != j of
    electrode columns.

    H: vector output matrix, shape (T, N_h). Uses the full electrode vector,
    not a scalar reduction of it.

    Returns a scalar in [0, 1]. High value -> electrodes observe independent
    directions. If N_h < 2, there are no pairs to compare and NaN is returned.
    """
    H = np.asarray(H)
    N_h = H.shape[1]

    if N_h < 2:
        print(
            "observational_orthogonality: N_h < 2, cannot compute pairwise "
            "metric. Returning NaN."
        )
        return float("nan")

    scores = []
    for i in range(N_h):
        for j in range(N_h):
            if i == j:
                continue
            h_i = H[:, i]
            h_j = H[:, j]
            cos_ij = (h_i @ h_j) / (np.linalg.norm(h_i) * np.linalg.norm(h_j) + 1e-12)
            scores.append(1.0 - abs(cos_ij))

    return float(np.mean(scores))
