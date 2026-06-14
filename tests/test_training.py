"""Tests for the pure training plan + progress tracker (no camera)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestures import GestureType
import training


def test_default_plan_covers_every_gesture_once():
    plan = training.default_plan()
    covered = {s.gesture for s in plan}
    assert covered == {g.value for g in GestureType}
    assert len(plan) == len(GestureType)  # one step per gesture, no dupes


def test_tracker_counts_reps_and_advances():
    plan = [training.TrainingStep("SWIPE_RIGHT", "R", "i", "e", reps=2),
            training.TrainingStep("PINCH", "P", "i", "e", reps=1)]
    t = training.TrainingTracker(plan=plan)
    assert t.current.gesture == "SWIPE_RIGHT"
    assert t.record("SWIPE_RIGHT") is True
    assert t.progress() == (1, 2, 1, 2)  # step 1/2, 1 of 2 reps
    assert t.record("SWIPE_RIGHT") is True  # hits 2 -> advances
    assert t.current.gesture == "PINCH"
    assert t.progress() == (2, 2, 0, 1)


def test_tracker_ignores_nonmatching_gesture():
    t = training.TrainingTracker(plan=[training.TrainingStep("PINCH", "P", "i", "e", reps=2)])
    assert t.record("SWIPE_LEFT") is False  # wrong gesture doesn't count
    assert t.progress()[2] == 0


def test_tracker_completes_after_last_step():
    plan = [training.TrainingStep("POINT", "P", "i", "e", reps=1)]
    t = training.TrainingTracker(plan=plan)
    assert not t.is_complete
    t.record("POINT")
    assert t.is_complete
    assert t.current is None
    assert t.record("POINT") is False  # no-op once complete
    assert t.progress() == (1, 1, 0, 0)


def test_tracker_skip_advances_without_reps():
    plan = [training.TrainingStep("GRAB", "G", "i", "e", reps=5),
            training.TrainingStep("POINT", "P", "i", "e", reps=1)]
    t = training.TrainingTracker(plan=plan)
    t.skip()
    assert t.current.gesture == "POINT"
    t.skip()
    assert t.is_complete
    t.skip()  # safe past the end
    assert t.is_complete
