import numpy as np
import pandas as pd
import pytest
from scipy.stats import shapiro

from analysis.transforms import log_transform, rank_based_inverse_normal_transform


def test_rank_int_improves_normality_of_a_skewed_distribution():
    rng = np.random.default_rng(0)
    skewed = pd.Series(rng.beta(0.5, 8, size=200))  # heavily right-skewed, bounded like FO

    _, p_before = shapiro(skewed)
    transformed = rank_based_inverse_normal_transform(skewed)
    _, p_after = shapiro(transformed)

    assert p_before < 0.01  # confirms the fixture is actually non-normal
    assert p_after > 0.5   # rank-INT output should look normal


def test_rank_int_is_monotonic():
    x = pd.Series([5.0, 1.0, 3.0, 100.0, 2.0])
    transformed = rank_based_inverse_normal_transform(x)

    # order of transformed values must match order of original values
    assert list(transformed.sort_values().index) == list(x.sort_values().index)


def test_rank_int_ties_get_equal_values():
    x = pd.Series([1.0, 2.0, 2.0, 3.0])
    transformed = rank_based_inverse_normal_transform(x)

    assert transformed[1] == transformed[2]


def test_log_transform_improves_normality_of_lognormal_data():
    rng = np.random.default_rng(0)
    durations = pd.Series(rng.lognormal(mean=1.0, sigma=1.5, size=200), name="lifetime_sec")

    _, p_before = shapiro(durations)
    transformed = log_transform(durations)
    _, p_after = shapiro(transformed)

    assert p_before < 0.01
    assert p_after > 0.05


def test_log_transform_rejects_non_positive_values():
    x = pd.Series([1.0, 2.0, 0.0, 3.0], name="interval_sec")
    with pytest.raises(ValueError, match="interval_sec"):
        log_transform(x)
