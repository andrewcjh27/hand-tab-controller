"""Guided gesture training plan and progress tracking.

Pure logic (no cv2/mediapipe) so it is unit testable. ``main.py --train`` drives
the camera, renders the demo tab workspace so the user *sees each gesture act on
the tabs*, feeds detected gestures into a :class:`TrainingTracker` to count reps,
and collects calibration samples along the way.

Each :class:`TrainingStep` trains one movement: it names the gesture, tells the
user what to do and what it does to the tabs, asks for several reps, and (for the
threshold-driven gestures) tags which calibration samples to gather.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from gestures import GestureType


@dataclass
class TrainingStep:
    """One movement to train.

    Attributes:
        gesture: The :class:`GestureType` value to detect (e.g. ``"SWIPE_RIGHT"``).
        label: Short human name shown in the banner ("Swipe right").
        instruction: What the user should physically do.
        effect: What it does to the tabs (shown so they connect motion to result).
        reps: How many clean detections to require before advancing.
        sample: Calibration sample to collect during this step, one of
            ``"swipe"``, ``"pinch_motion"``, ``"pinch_open"``, ``"pinch_closed"``,
            or ``None`` for pose-only gestures that need no threshold.
        two_handed: True for gestures that need both hands (just informational).
    """

    gesture: str
    label: str
    instruction: str
    effect: str
    reps: int = 5
    sample: Optional[str] = None
    two_handed: bool = False


def default_plan() -> List[TrainingStep]:
    """The full training curriculum, in teaching order."""
    return [
        TrainingStep("SWIPE_RIGHT", "Swipe right",
                     "Move your open hand to the right", "next tab",
                     sample="swipe"),
        TrainingStep("SWIPE_LEFT", "Swipe left",
                     "Move your open hand to the left", "previous tab",
                     sample="swipe"),
        TrainingStep("SPREAD", "Spread",
                     "Hold your hand still; move thumb and index apart",
                     "grows the active tab", sample="pinch_motion"),
        TrainingStep("PINCH", "Pinch",
                     "Hold your hand still; bring thumb and index together",
                     "shrinks the active tab", sample="pinch_motion"),
        TrainingStep("POINT", "Point",
                     "Extend only your index finger", "starts moving a tab"),
        TrainingStep("GRAB", "Grab",
                     "Close your hand into a fist", "drags the tab with your hand",
                     sample="pinch_closed"),
        TrainingStep("OPEN_PALM", "Open palm",
                     "Open your whole hand flat", "releases the tab",
                     sample="pinch_open"),
        TrainingStep("V_SIGN", "Peace sign",
                     "Make a V with index and middle fingers", "toggles split view"),
        TrainingStep("TWO_HAND_PINCH", "Two-hand resize",
                     "Show both hands; move them closer / apart",
                     "resizes by the gap between hands", reps=3, two_handed=True),
    ]


@dataclass
class TrainingTracker:
    """Counts reps of the current step and advances through the plan."""

    plan: List[TrainingStep] = field(default_factory=default_plan)
    index: int = 0
    count: int = 0

    @property
    def is_complete(self) -> bool:
        return self.index >= len(self.plan)

    @property
    def current(self) -> Optional[TrainingStep]:
        return None if self.is_complete else self.plan[self.index]

    def record(self, gesture_value: str) -> bool:
        """Register a detected gesture. Returns True if it counted as a rep.

        A rep counts only when the detected gesture matches the current step;
        reaching the step's ``reps`` advances to the next step (and resets the
        counter). Detections that don't match the current step are ignored so
        stray gestures don't derail training.
        """
        if self.is_complete:
            return False
        step = self.plan[self.index]
        if gesture_value != step.gesture:
            return False
        self.count += 1
        if self.count >= step.reps:
            self.index += 1
            self.count = 0
        return True

    def skip(self) -> None:
        """Skip the current step (e.g. user can't perform it)."""
        if not self.is_complete:
            self.index += 1
            self.count = 0

    def progress(self) -> tuple[int, int, int, int]:
        """Return (step_index, step_total, reps_done, reps_required).

        ``step_index`` is 1-based for display and clamped to the plan length when
        complete.
        """
        total = len(self.plan)
        if self.is_complete:
            return total, total, 0, 0
        return self.index + 1, total, self.count, self.plan[self.index].reps
