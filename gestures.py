"""Hand-gesture recognition.

This module deliberately keeps all gesture *math* in pure functions that take
numpy landmark arrays. That makes them unit-testable without a camera or
MediaPipe installed. MediaPipe / OpenCV are only needed by the live pipeline
(``main.py``); nothing here imports them.

MediaPipe Hands returns 21 landmarks per hand. We use this standard indexing:

    0  : WRIST
    1-4: THUMB  (CMC, MCP, IP, TIP)
    5-8: INDEX  (MCP, PIP, DIP, TIP)
    9-12: MIDDLE
    13-16: RING
    17-20: PINKY

Each landmark is an (x, y) or (x, y, z) coordinate normalized to [0, 1] with the
origin at the top-left of the (un-mirrored) image.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]


# ---------------------------------------------------------------------------
# Gesture event types
# ---------------------------------------------------------------------------
class GestureType(str, Enum):
    """All recognized gesture types."""

    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    PINCH = "PINCH"
    SPREAD = "SPREAD"
    POINT = "POINT"
    GRAB = "GRAB"
    OPEN_PALM = "OPEN_PALM"
    TWO_HAND_PINCH = "TWO_HAND_PINCH"
    V_SIGN = "V_SIGN"


@dataclass
class Gesture:
    """A normalized gesture event.

    Attributes:
        type: The :class:`GestureType`.
        magnitude: Gesture-specific scalar (swipe velocity, pinch scale, etc.).
        position: Normalized (x, y) anchor point, typically the palm centroid.
        hand: Hand label ("Left", "Right", or "Both" for two-hand gestures).
    """

    type: GestureType
    magnitude: float = 0.0
    position: tuple[float, float] = (0.5, 0.5)
    hand: str = "Right"


# ---------------------------------------------------------------------------
# Pure-logic helpers (unit-testable, no cv2/mediapipe)
# ---------------------------------------------------------------------------
def as_array(landmarks: Sequence) -> np.ndarray:
    """Coerce a sequence of landmarks into an ``(N, 2)`` float array.

    Accepts a numpy array, a list of (x, y[, z]) tuples, or anything indexable.
    Only the first two coordinates (x, y) are kept.
    """
    arr = np.asarray(landmarks, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("landmarks must be shape (N, 2) or (N, 3)")
    return arr[:, :2]


def palm_centroid(landmarks: np.ndarray) -> np.ndarray:
    """Return the (x, y) centroid of the palm.

    Uses wrist plus the MCP joints of the four fingers, which is more stable
    than the wrist alone.
    """
    pts = landmarks[[WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP], :]
    return pts.mean(axis=0)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def hand_scale(landmarks: np.ndarray) -> float:
    """A scale-invariant reference length for the hand.

    Distance from wrist to the middle-finger MCP. Used to normalize pinch
    distances so they don't depend on how close the hand is to the camera.
    """
    return max(distance(landmarks[WRIST], landmarks[MIDDLE_MCP]), 1e-6)


def pinch_distance(landmarks: np.ndarray) -> float:
    """Normalized distance between thumb tip and index tip.

    Returns the raw distance divided by :func:`hand_scale` so the value is
    roughly comparable across hand sizes and camera distances.
    """
    raw = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    return raw / hand_scale(landmarks)


def finger_extended(landmarks: np.ndarray, tip: int, mcp: int) -> bool:
    """Heuristic: is the finger extended?

    A finger is considered extended when its tip is farther from the wrist than
    its MCP joint is (works regardless of hand orientation in 2D, good enough
    for our coarse gesture set).
    """
    wrist = landmarks[WRIST]
    return distance(landmarks[tip], wrist) > distance(landmarks[mcp], wrist)


def count_extended_fingers(landmarks: np.ndarray) -> int:
    """Count how many of the four (non-thumb) fingers are extended."""
    pairs = [
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ]
    return sum(finger_extended(landmarks, tip, mcp) for tip, mcp in pairs)


def is_open_palm(landmarks: np.ndarray) -> bool:
    """True when all four fingers are extended."""
    return count_extended_fingers(landmarks) >= 4


def is_grab(landmarks: np.ndarray, pinch_threshold: float) -> bool:
    """True when the hand is a closed fist / pinch-grab.

    Detected as: few fingers extended AND thumb-index pinch is closed.
    """
    return (
        count_extended_fingers(landmarks) <= 1
        and pinch_distance(landmarks) < pinch_threshold
    )


def is_point(landmarks: np.ndarray) -> bool:
    """True when only the index finger is extended (pointing)."""
    index = finger_extended(landmarks, INDEX_TIP, INDEX_MCP)
    others = (
        finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP)
        + finger_extended(landmarks, RING_TIP, RING_MCP)
        + finger_extended(landmarks, PINKY_TIP, PINKY_MCP)
    )
    return index and others == 0


def is_v_sign(landmarks: np.ndarray) -> bool:
    """True for a "V"/peace sign: index and middle extended, ring and pinky down.

    Used as the default gesture for toggling a split-window layout.
    """
    index = finger_extended(landmarks, INDEX_TIP, INDEX_MCP)
    middle = finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP)
    others = (
        finger_extended(landmarks, RING_TIP, RING_MCP)
        + finger_extended(landmarks, PINKY_TIP, PINKY_MCP)
    )
    return index and middle and others == 0


def two_hand_pinch_distance(
    landmarks_a: np.ndarray, landmarks_b: np.ndarray
) -> float:
    """Distance between the two palms' centroids (for two-hand resize)."""
    return distance(palm_centroid(landmarks_a), palm_centroid(landmarks_b))


# ---------------------------------------------------------------------------
# Velocity / swipe detection
# ---------------------------------------------------------------------------
@dataclass
class VelocityTracker:
    """Tracks the recent centroid positions of a hand to estimate velocity.

    The swipe detector looks at horizontal displacement over the buffered
    window. ``magnitude`` of the resulting swipe is the average horizontal
    velocity (normalized units per frame).
    """

    window: int = 5
    _xs: Deque[float] = field(default_factory=deque, init=False)
    _ys: Deque[float] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        self._xs = deque(maxlen=self.window)
        self._ys = deque(maxlen=self.window)

    @property
    def sample_count(self) -> int:
        """Number of buffered samples (at most ``window``)."""
        return len(self._xs)

    def update(self, position: Sequence[float]) -> None:
        """Push a new (x, y) palm centroid."""
        self._xs.append(float(position[0]))
        self._ys.append(float(position[1]))

    def reset(self) -> None:
        self._xs.clear()
        self._ys.clear()

    def horizontal_velocity(self) -> float:
        """Average horizontal velocity over the window (per frame).

        Positive = moving right, negative = moving left.
        """
        if len(self._xs) < 2:
            return 0.0
        return (self._xs[-1] - self._xs[0]) / (len(self._xs) - 1)

    def horizontal_travel(self) -> float:
        """Absolute horizontal distance covered over the buffered window."""
        if len(self._xs) < 2:
            return 0.0
        return abs(self._xs[-1] - self._xs[0])

    def vertical_spread(self) -> float:
        """Vertical travel over the window (used to reject diagonal motion)."""
        if len(self._ys) < 2:
            return 0.0
        return abs(self._ys[-1] - self._ys[0])

    def total_speed(self) -> float:
        """Overall palm speed: net displacement magnitude per frame over window.

        Used to tell a *translating* hand (a swipe) from a *stationary* hand
        whose fingers are moving (a pinch/zoom).
        """
        if len(self._xs) < 2:
            return 0.0
        dx = self._xs[-1] - self._xs[0]
        dy = self._ys[-1] - self._ys[0]
        return ((dx * dx + dy * dy) ** 0.5) / (len(self._xs) - 1)


def detect_swipe(
    tracker: VelocityTracker, velocity_threshold: float
) -> Optional[GestureType]:
    """Return SWIPE_LEFT / SWIPE_RIGHT if the tracked velocity is large enough.

    Pure logic over a :class:`VelocityTracker`; no camera needed. Rejects motion
    that is mostly vertical (more vertical than horizontal travel).
    """
    vx = tracker.horizontal_velocity()
    if abs(vx) < velocity_threshold:
        return None
    if tracker.vertical_spread() > tracker.horizontal_travel():
        return None
    return GestureType.SWIPE_RIGHT if vx > 0 else GestureType.SWIPE_LEFT


def arbitrate_dynamic(
    horizontal_velocity: float,
    palm_speed: float,
    pinch_net: float,
    swipe_velocity: float,
    pinch_sensitivity: float,
    horizontal_travel: float,
    vertical_travel: float,
    still_factor: float = 0.6,
) -> Optional[GestureType]:
    """Pick at most one *dynamic* gesture: SWIPE vs PINCH/SPREAD, never both.

    The two are separated by how the whole hand moves, which is what stops a
    swipe from also registering as a zoom (and vice versa):

    * **Swipe** — the palm is *translating* (``palm_speed >= swipe_velocity``),
      mostly horizontally (horizontal travel dominates vertical).
    * **Zoom** — the palm is roughly *stationary*
      (``palm_speed <= swipe_velocity * still_factor``) while the thumb-index
      distance changes by at least ``pinch_sensitivity`` over the window.

    Motion that is neither clearly translating nor clearly still (the ambiguous
    middle band) returns ``None`` so nothing fires by accident.
    """
    if abs(horizontal_velocity) >= swipe_velocity and palm_speed >= swipe_velocity:
        if vertical_travel <= horizontal_travel:
            return GestureType.SWIPE_RIGHT if horizontal_velocity > 0 else GestureType.SWIPE_LEFT
    if palm_speed <= swipe_velocity * still_factor and abs(pinch_net) >= pinch_sensitivity:
        return GestureType.SPREAD if pinch_net > 0 else GestureType.PINCH
    return None


def classify_pinch_change(
    prev_distance: Optional[float],
    curr_distance: float,
    sensitivity: float,
) -> Optional[GestureType]:
    """Classify a change in pinch distance as PINCH (in) or SPREAD (out).

    Args:
        prev_distance: Previous normalized pinch distance (or None on first frame).
        curr_distance: Current normalized pinch distance.
        sensitivity: Minimum absolute change to register.

    Returns:
        GestureType.SPREAD if fingers moved apart, GestureType.PINCH if they
        moved together, else None.
    """
    if prev_distance is None:
        return None
    delta = curr_distance - prev_distance
    if abs(delta) < sensitivity:
        return None
    return GestureType.SPREAD if delta > 0 else GestureType.PINCH


# ---------------------------------------------------------------------------
# Live wrapper (stateful) — used by the camera pipeline
# ---------------------------------------------------------------------------
class HandLandmarks:
    """Wraps a single hand's 21 MediaPipe landmarks.

    Constructed from a MediaPipe ``landmark`` list or any sequence of (x, y[, z]).
    Exposes convenience accessors that delegate to the pure functions above.
    """

    def __init__(self, landmarks: Sequence, label: str = "Right") -> None:
        self.points = as_array(landmarks)
        self.label = label

    @classmethod
    def from_task_landmarks(cls, landmarks, label: str = "Right") -> "HandLandmarks":
        """Build from a MediaPipe Tasks ``HandLandmarker`` result.

        The Tasks API yields a flat list of landmarks (each with ``x/y/z``)
        rather than the legacy ``.landmark`` wrapper.
        """
        pts = [(lm.x, lm.y, lm.z) for lm in landmarks]
        return cls(pts, label=label)

    @property
    def centroid(self) -> tuple[float, float]:
        c = palm_centroid(self.points)
        return float(c[0]), float(c[1])

    def pinch(self) -> float:
        return pinch_distance(self.points)

    def is_open_palm(self) -> bool:
        return is_open_palm(self.points)

    def is_point(self) -> bool:
        return is_point(self.points)

    def is_v_sign(self) -> bool:
        return is_v_sign(self.points)


class GestureRecognizer:
    """Stateful per-frame recognizer for one or two hands.

    Holds velocity trackers and prior pinch distances so it can emit SWIPE and
    PINCH/SPREAD events across frames. Static-pose gestures (POINT, GRAB,
    OPEN_PALM, TWO_HAND_PINCH) are emitted from the current frame.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.swipe_velocity = float(cfg.get("swipe_velocity", 0.04))
        self.pinch_sensitivity = float(cfg.get("pinch_sensitivity", 0.05))
        self.pinch_threshold = float(cfg.get("pinch_threshold", 0.4))
        self.window = int(cfg.get("smoothing_window", 5))
        self._trackers: dict[str, VelocityTracker] = {}
        self._pinch_wins: dict[str, Deque[float]] = {}

    def _tracker(self, label: str) -> VelocityTracker:
        if label not in self._trackers:
            self._trackers[label] = VelocityTracker(window=self.window)
        return self._trackers[label]

    def _pinch_window(self, label: str) -> Deque[float]:
        if label not in self._pinch_wins:
            self._pinch_wins[label] = deque(maxlen=self.window)
        return self._pinch_wins[label]

    def update(self, hands: List[HandLandmarks]) -> List[Gesture]:
        """Process the current frame's detected hands and return gestures."""
        events: List[Gesture] = []

        # Two-hand pinch (resize) takes priority when two hands are present.
        if len(hands) >= 2:
            a, b = hands[0], hands[1]
            dist = two_hand_pinch_distance(a.points, b.points)
            mid = (
                (a.centroid[0] + b.centroid[0]) / 2,
                (a.centroid[1] + b.centroid[1]) / 2,
            )
            events.append(
                Gesture(GestureType.TWO_HAND_PINCH, magnitude=dist,
                        position=mid, hand="Both")
            )

        for hand in hands:
            label = hand.label
            centroid = hand.centroid
            tracker = self._tracker(label)
            tracker.update(centroid)
            pinch_win = self._pinch_window(label)
            curr_pinch = hand.pinch()
            pinch_win.append(curr_pinch)
            pinch_net = pinch_win[-1] - pinch_win[0] if len(pinch_win) >= 2 else 0.0

            # One dynamic gesture at most: swipe (hand translating) XOR
            # pinch/zoom (hand still, fingers moving). This is what keeps a
            # swipe from also reading as a zoom and vice versa.
            dynamic = arbitrate_dynamic(
                tracker.horizontal_velocity(), tracker.total_speed(), pinch_net,
                self.swipe_velocity, self.pinch_sensitivity,
                horizontal_travel=tracker.horizontal_travel(),
                vertical_travel=tracker.vertical_spread(),
            )

            if dynamic in (GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT):
                # A swipe is an *open-hand* translation; a moving fist is a drag.
                if hand.is_open_palm():
                    events.append(
                        Gesture(dynamic, magnitude=abs(tracker.horizontal_velocity()),
                                position=centroid, hand=label)
                    )
                    tracker.reset()
                    pinch_win.clear()
                    continue
            elif dynamic in (GestureType.PINCH, GestureType.SPREAD):
                events.append(
                    Gesture(dynamic, magnitude=curr_pinch, position=centroid, hand=label)
                )
                pinch_win.clear()
                continue

            # No dynamic gesture this frame -> fall back to a single static pose.
            if is_grab(hand.points, self.pinch_threshold):
                events.append(
                    Gesture(GestureType.GRAB, magnitude=curr_pinch,
                            position=centroid, hand=label)
                )
            elif hand.is_v_sign():
                events.append(
                    Gesture(GestureType.V_SIGN, position=centroid, hand=label)
                )
            elif hand.is_point():
                events.append(
                    Gesture(GestureType.POINT, position=centroid, hand=label)
                )
            elif hand.is_open_palm():
                events.append(
                    Gesture(GestureType.OPEN_PALM, position=centroid, hand=label)
                )

        return events
