"""Tests for the pure calibration reducers (no camera)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calibration as cal


def test_swipe_velocity_is_fraction_of_median_peak():
    # median peak = 0.2, fraction 0.55 -> 0.11
    assert cal.recommend_swipe_velocity([0.1, 0.2, 0.3], fraction=0.55) == 0.11


def test_swipe_velocity_clamped_to_floor_and_cap():
    assert cal.recommend_swipe_velocity([0.001], fraction=0.55) == cal.SWIPE_VELOCITY_FLOOR
    assert cal.recommend_swipe_velocity([5.0], fraction=0.55) == cal.SWIPE_VELOCITY_CAP


def test_swipe_velocity_none_when_empty():
    assert cal.recommend_swipe_velocity([]) is None
    assert cal.recommend_swipe_velocity([None, None]) is None


def test_pinch_sensitivity_ignores_zero_and_none():
    # nonzero abs deltas: 0.2, 0.4 -> median 0.3 * 0.5 = 0.15
    assert cal.recommend_pinch_sensitivity([0.0, None, 0.2, -0.4], fraction=0.5) == 0.15


def test_pinch_sensitivity_clamped():
    assert cal.recommend_pinch_sensitivity([0.001], fraction=0.5) == cal.PINCH_SENSITIVITY_FLOOR
    assert cal.recommend_pinch_sensitivity([9.0], fraction=0.5) == cal.PINCH_SENSITIVITY_CAP


def test_pinch_sensitivity_none_when_no_motion():
    assert cal.recommend_pinch_sensitivity([0.0, 0.0]) is None
    assert cal.recommend_pinch_sensitivity([]) is None


def test_pinch_sensitivity_floored_above_swipe_noise():
    # Deliberate pinch deltas are small (would give 0.05), but swipe jitter
    # peaked at 0.12 -> threshold is floored above the jitter so swiping can't
    # trip a false zoom.
    got = cal.recommend_pinch_sensitivity([0.1, 0.1], fraction=0.5,
                                          noise_deltas=[0.05, 0.12, 0.0, None])
    assert got == 0.12


def test_pinch_sensitivity_noise_ignored_when_empty():
    # Empty/zero noise leaves the normal floor behavior intact.
    got = cal.recommend_pinch_sensitivity([0.2, 0.4], fraction=0.5, noise_deltas=[0.0])
    assert got == 0.15


def test_pinch_threshold_between_clusters():
    # closed median 0.2, open median 0.8, bias 0.5 -> 0.5
    assert cal.recommend_pinch_threshold([0.18, 0.2, 0.22], [0.78, 0.8, 0.82]) == 0.5


def test_pinch_threshold_bias_shifts_toward_open():
    # 0.2 + 0.75*(0.8-0.2) = 0.65
    assert cal.recommend_pinch_threshold([0.2], [0.8], bias=0.75) == 0.65


def test_pinch_threshold_none_when_not_separable():
    # open not larger than closed -> can't separate
    assert cal.recommend_pinch_threshold([0.6], [0.4]) is None
    assert cal.recommend_pinch_threshold([], [0.8]) is None
    assert cal.recommend_pinch_threshold([0.2], []) is None


def test_merge_thresholds_skips_none():
    existing = {"swipe_velocity": 0.07, "pinch_threshold": 0.3, "cooldown_ms": 900}
    measured = {"swipe_velocity": 0.05, "pinch_threshold": None}
    merged = cal.merge_thresholds(existing, measured)
    assert merged["swipe_velocity"] == 0.05  # overwritten
    assert merged["pinch_threshold"] == 0.3  # None left untouched
    assert merged["cooldown_ms"] == 900  # unrelated preserved
    # original not mutated
    assert existing["swipe_velocity"] == 0.07
