"""Derive recognizer thresholds from recordings of the user's own hand motion.

This module is **pure** (no cv2/mediapipe) so the calibration math is unit
testable. ``main.py --calibrate`` does the live capture (prompting the user,
reading frames) and feeds the collected samples into the ``recommend_*``
reducers here, then writes the merged thresholds back to ``gestures.json``.

The goal is *lower, personalized* sensitivity: each trigger threshold is placed
comfortably below the user's natural motion so real gestures fire reliably while
incidental movement does not. Safety floors/caps prevent a noisy or sparse
recording from producing a threshold that makes a gesture impossible (too high)
or one that fires constantly (too low).
"""

from __future__ import annotations

from statistics import median
from typing import Dict, List, Optional, Sequence

# Bounds keep an odd recording from yielding a degenerate threshold.
SWIPE_VELOCITY_FLOOR = 0.03
SWIPE_VELOCITY_CAP = 0.30
PINCH_SENSITIVITY_FLOOR = 0.04
PINCH_SENSITIVITY_CAP = 0.40
PINCH_THRESHOLD_FLOOR = 0.10
PINCH_THRESHOLD_CAP = 0.90


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clean(samples: Sequence[Optional[float]]) -> List[float]:
    """Drop ``None`` entries and coerce to float."""
    return [float(s) for s in samples if s is not None]


def recommend_swipe_velocity(
    peak_velocities: Sequence[Optional[float]], fraction: float = 0.55
) -> Optional[float]:
    """Swipe trigger = ``fraction`` of the median peak swipe speed.

    ``peak_velocities`` is the peak absolute horizontal palm velocity (normalized
    units/frame) observed during each of the user's practice swipes. Returns
    ``None`` when there are no samples to learn from.
    """
    vals = [abs(v) for v in _clean(peak_velocities)]
    if not vals:
        return None
    return round(_clamp(median(vals) * fraction, SWIPE_VELOCITY_FLOOR, SWIPE_VELOCITY_CAP), 4)


def recommend_pinch_sensitivity(
    frame_deltas: Sequence[Optional[float]],
    fraction: float = 0.5,
    noise_deltas: Optional[Sequence[Optional[float]]] = None,
) -> Optional[float]:
    """Min per-frame pinch-distance change to register a PINCH/SPREAD.

    Derived from the absolute frame-to-frame pinch-distance changes observed
    while the user deliberately pinches and spreads. Zero/None deltas (a still
    hand) are ignored. Returns ``None`` when there are no usable samples.

    ``noise_deltas`` are the pinch-distance changes seen while the user was
    *swiping* (fingers wobbling, not deliberately pinching). When supplied, the
    threshold is floored just above that jitter so swiping no longer trips a
    false zoom — the main swipe/zoom confusion fix on the calibration side.
    """
    deltas = [abs(d) for d in _clean(frame_deltas) if d != 0]
    if not deltas:
        return None
    floor = PINCH_SENSITIVITY_FLOOR
    if noise_deltas:
        noise = [abs(d) for d in _clean(noise_deltas) if d != 0]
        if noise:
            floor = max(floor, max(noise))
    return round(_clamp(median(deltas) * fraction, floor, PINCH_SENSITIVITY_CAP), 4)


def recommend_pinch_threshold(
    closed_samples: Sequence[Optional[float]],
    open_samples: Sequence[Optional[float]],
    bias: float = 0.5,
) -> Optional[float]:
    """Grab threshold placed between the closed-fist and open-hand pinch clusters.

    ``is_grab`` fires when the normalized thumb-index distance is *below* this
    threshold, so it must sit above the user's closed-fist distance and below
    their open-hand distance. ``bias`` (0..1) slides it between the two clusters;
    0.5 is the midpoint. Returns ``None`` if either cluster is empty or the
    recording can't separate them (open not larger than closed).
    """
    closed = _clean(closed_samples)
    opened = _clean(open_samples)
    if not closed or not opened:
        return None
    c, o = median(closed), median(opened)
    if o <= c:
        return None
    threshold = c + bias * (o - c)
    return round(_clamp(threshold, PINCH_THRESHOLD_FLOOR, PINCH_THRESHOLD_CAP), 4)


def merge_thresholds(
    existing: Dict[str, float], measured: Dict[str, Optional[float]]
) -> Dict[str, float]:
    """Overlay non-``None`` measured thresholds onto the existing ones.

    A gesture the user skipped (or that couldn't be measured) yields ``None`` and
    leaves its existing/default threshold untouched.
    """
    out = dict(existing)
    for key, value in measured.items():
        if value is not None:
            out[key] = value
    return out
