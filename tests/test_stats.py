import pytest

from agent_evals.core.stats import bootstrap_mean_diff_ci, cohens_kappa, pearson


def test_bootstrap_detects_clear_regression():
    current, baseline = [0.5] * 16, [1.0] * 16
    lo, hi = bootstrap_mean_diff_ci(current, baseline)
    assert hi < 0  # significant regression


def test_bootstrap_no_regression_on_equal_samples():
    lo, hi = bootstrap_mean_diff_ci([1.0] * 16, [1.0] * 16)
    assert lo == hi == 0.0


def test_bootstrap_noise_is_not_significant():
    current = [1, 0, 1, 1, 0, 1, 1, 1]
    baseline = [1, 1, 0, 1, 1, 0, 1, 1]
    lo, hi = bootstrap_mean_diff_ci(current, baseline)
    assert lo < 0 < hi  # CI spans zero -> not significant


def test_cohens_kappa():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_pearson():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
