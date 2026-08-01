import numpy as np
import pandas as pd
import pytest

from analysis.entropy import entropy_rate, entropy_rate_by_subject, stationary_distribution
from analysis.io import TRANSITION_COLUMNS

# Two-state chain worked out by hand:
#   P = [[0.9, 0.1],
#        [0.2, 0.8]]
# Stationary distribution solves mu1*0.1 = mu2*0.2 with mu1+mu2=1 -> mu = [2/3, 1/3].
# Per-row entropies: H1 = -(0.9 ln 0.9 + 0.1 ln 0.1) = 0.325082973
#                    H2 = -(0.2 ln 0.2 + 0.8 ln 0.8) = 0.500402423
# Entropy rate = mu . H = 0.383522790 nats per unit time.
TWO_STATE_P = np.array([[0.9, 0.1], [0.2, 0.8]])
EXPECTED_MU = np.array([2 / 3, 1 / 3])
EXPECTED_RATE = 0.383522790


def test_stationary_distribution_two_state_chain():
    mu = stationary_distribution(TWO_STATE_P)
    assert np.allclose(mu, EXPECTED_MU, atol=1e-6)
    assert np.isclose(mu.sum(), 1.0)


def test_entropy_rate_two_state_chain():
    rate = entropy_rate(TWO_STATE_P)
    assert np.isclose(rate, EXPECTED_RATE, atol=1e-6)


def test_entropy_rate_fs_scaling():
    rate_nats = entropy_rate(TWO_STATE_P, fs=None)
    rate_per_second = entropy_rate(TWO_STATE_P, fs=4.0)
    assert np.isclose(rate_per_second, rate_nats * 4.0)


def test_entropy_rate_handles_zero_probability_cells():
    # A deterministic self-loop state (row of all zeros except the diagonal)
    # must not raise on log(0); its contribution to entropy is exactly 0.
    P = np.array([[1.0, 0.0], [0.3, 0.7]])
    rate = entropy_rate(P)
    assert np.isfinite(rate)


def test_entropy_rate_by_subject_matches_single_matrix_result():
    long_format = pd.DataFrame(
        [
            {"subject_id": 1, "from_state": i + 1, "to_state": j + 1, "probability": TWO_STATE_P[i, j]}
            for i in range(2)
            for j in range(2)
        ],
        columns=TRANSITION_COLUMNS,
    )
    result = entropy_rate_by_subject(long_format, fs=None)
    assert list(result["subject_id"]) == [1]
    assert np.isclose(result.loc[0, "entropy_rate"], EXPECTED_RATE, atol=1e-6)


def test_stationary_distribution_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        stationary_distribution(np.ones((2, 3)))
