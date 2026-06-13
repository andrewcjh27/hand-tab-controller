"""Unit tests for pure gesture math. No camera / mediapipe required."""

import numpy as np
import pytest

from gestures import (
    GestureType,
    VelocityTracker,
    classify_pinch_change,
    count_extended_fingers,
    detect_swipe,
    is_open_palm,
    is_point,
    is_v_sign,
    pinch_distance,
    two_hand_pinch_distance,
)


def make_hand(open_fingers=4, pinch=0.6):
    """Build a synthetic 21-landmark hand.

    Lays the wrist at the bottom and fingers pointing up. ``open_fingers``
    controls how many of the four fingers are extended; ``pinch`` sets the
    thumb-index tip separation.
    """
    pts = np.zeros((21, 2), dtype=float)
    pts[0] = (0.5, 0.9)          # wrist
    # MCPs (row near 0.6)
    pts[5] = (0.40, 0.6)         # index mcp
    pts[9] = (0.50, 0.6)         # middle mcp
    pts[13] = (0.60, 0.6)        # ring mcp
    pts[17] = (0.68, 0.6)        # pinky mcp
    # finger tips: extended -> high up (small y), folded -> near mcp
    finger_specs = [(8, 0.40), (12, 0.50), (16, 0.60), (20, 0.68)]
    for i, (tip, x) in enumerate(finger_specs):
        if i < open_fingers:
            pts[tip] = (x, 0.2)   # extended
        else:
            pts[tip] = (x, 0.62)  # folded (near mcp, closer to wrist)
    # thumb: index tip is at (0.40, 0.2) when extended; place thumb tip so the
    # thumb-index distance matches the requested pinch (scaled by hand size).
    index_tip = pts[8]
    pts[4] = (index_tip[0] + pinch, index_tip[1])
    return pts


def test_open_palm_detection():
    assert is_open_palm(make_hand(open_fingers=4))
    assert not is_open_palm(make_hand(open_fingers=1))


def test_point_detection():
    assert is_point(make_hand(open_fingers=1))
    assert not is_point(make_hand(open_fingers=4))


def test_v_sign_detection():
    # index + middle extended, ring + pinky folded
    assert is_v_sign(make_hand(open_fingers=2))
    assert not is_v_sign(make_hand(open_fingers=1))  # only index -> point
    assert not is_v_sign(make_hand(open_fingers=4))  # open palm
    # a V-sign must not also register as a point
    assert not is_point(make_hand(open_fingers=2))


def test_extended_count():
    assert count_extended_fingers(make_hand(open_fingers=3)) == 3


def test_pinch_distance_scales():
    near = pinch_distance(make_hand(pinch=0.05))
    far = pinch_distance(make_hand(pinch=0.6))
    assert far > near


def test_swipe_right_from_velocity():
    tracker = VelocityTracker(window=5)
    for x in (0.2, 0.35, 0.5, 0.65, 0.8):
        tracker.update((x, 0.5))
    assert detect_swipe(tracker, velocity_threshold=0.04) == GestureType.SWIPE_RIGHT


def test_swipe_left_from_velocity():
    tracker = VelocityTracker(window=5)
    for x in (0.8, 0.65, 0.5, 0.35, 0.2):
        tracker.update((x, 0.5))
    assert detect_swipe(tracker, velocity_threshold=0.04) == GestureType.SWIPE_LEFT


def test_slow_motion_is_not_a_swipe():
    tracker = VelocityTracker(window=5)
    for x in (0.50, 0.51, 0.52, 0.53, 0.54):
        tracker.update((x, 0.5))
    assert detect_swipe(tracker, velocity_threshold=0.04) is None


def test_vertical_motion_rejected():
    tracker = VelocityTracker(window=5)
    for y in (0.2, 0.4, 0.6, 0.8, 0.95):
        tracker.update((0.5, y))
    assert detect_swipe(tracker, velocity_threshold=0.04) is None


def test_velocity_tracker_travel_and_sample_count():
    tracker = VelocityTracker(window=3)
    assert tracker.sample_count == 0
    assert tracker.horizontal_travel() == 0.0  # too few samples
    for x in (0.2, 0.5, 0.9):
        tracker.update((x, 0.5))
    assert tracker.sample_count == 3  # capped at window
    assert tracker.horizontal_travel() == pytest.approx(0.7)
    tracker.reset()
    assert tracker.sample_count == 0


def test_pinch_in_and_spread_out():
    assert classify_pinch_change(0.6, 0.4, sensitivity=0.05) == GestureType.PINCH
    assert classify_pinch_change(0.4, 0.6, sensitivity=0.05) == GestureType.SPREAD
    assert classify_pinch_change(0.5, 0.51, sensitivity=0.05) is None
    assert classify_pinch_change(None, 0.5, sensitivity=0.05) is None


def test_two_hand_pinch_distance():
    a = make_hand()
    b = make_hand()
    b = b + np.array([0.3, 0.0])
    assert two_hand_pinch_distance(a, b) > 0.25
