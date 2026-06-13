"""Configuration loading for the hand-tab controller.

The JSON config has two sections:

``mappings``  -- maps a gesture-type name (see :class:`gestures.GestureType`)
                 to an *action name* understood by :mod:`actions`.
``thresholds``-- tunable numeric parameters:

    swipe_velocity     Minimum horizontal palm velocity (normalized units per
                       frame) to count as a swipe. Lower = more sensitive.
    pinch_sensitivity  Minimum change in normalized thumb-index distance per
                       frame to register a PINCH / SPREAD.
    pinch_threshold    Normalized thumb-index distance below which the hand is
                       considered "pinched" (used for GRAB detection).
    smoothing_window   Number of recent frames used for velocity estimation.
    cooldown_ms        Debounce time (ms) between repeated firings of the same
                       action, so one swipe doesn't switch many tabs.
    move_speed         Pixels moved per drag step.
    resize_step        Fractional scale change per PINCH / SPREAD step.

No camera or MediaPipe imports here -- this module is pure and testable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List

from gestures import GestureType  # pure import: no cv2/mediapipe

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gestures.json")

DEFAULT_MAPPINGS: Dict[str, str] = {
    "SWIPE_LEFT": "prev_tab",
    "SWIPE_RIGHT": "next_tab",
    "PINCH": "resize_shrink",
    "SPREAD": "resize_grow",
    "POINT": "begin_move",
    "GRAB": "drag_move",
    "OPEN_PALM": "release",
    "TWO_HAND_PINCH": "resize_two_hand",
    "V_SIGN": "toggle_double_view",
}

# When ``backend`` is "os", the gesture *type* names stay the same but they are
# routed to the OS window controller via these action names. This is the set the
# OSActionRouter understands.
DEFAULT_OS_MAPPINGS: Dict[str, str] = {
    "SWIPE_LEFT": "prev_app",
    "SWIPE_RIGHT": "next_app",
    "PINCH": "resize_shrink",
    "SPREAD": "resize_grow",
    "POINT": "begin_move",
    "GRAB": "drag_move",
    "OPEN_PALM": "release",
    "TWO_HAND_PINCH": "resize_two_hand",
    "V_SIGN": "toggle_split",
}

# Valid backends: "os" drives real macOS windows; "canvas" is the in-app demo.
DEFAULT_BACKEND = "os"
DEFAULT_CAMERA_INDEX = 0

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "swipe_velocity": 0.04,
    "pinch_sensitivity": 0.05,
    "pinch_threshold": 0.4,
    "smoothing_window": 5,
    "cooldown_ms": 600,
    "move_speed": 40,
    "resize_step": 0.1,
}


# Valid gesture names are exactly the GestureType enum values.
VALID_GESTURES: frozenset[str] = frozenset(g.value for g in GestureType)

# Valid action names per backend = the set of values of that backend's default
# mapping. Kept explicit so the validation check is obvious and testable.
VALID_ACTIONS_OS: frozenset[str] = frozenset(DEFAULT_OS_MAPPINGS.values())
VALID_ACTIONS_CANVAS: frozenset[str] = frozenset(DEFAULT_MAPPINGS.values())

# Sane ranges for each threshold. Each entry: (kind, low, high) where kind is
# "float" or "int". ``low``/``high`` are inclusive/exclusive per the comment;
# bounds are encoded as (low_exclusive, low, high, high_inclusive).
#   (low_exclusive, low_bound, high_bound, high_inclusive)
# A bound of None means unbounded on that side.
THRESHOLD_RANGES: Dict[str, tuple] = {
    # name:            (kind,  low_excl, low,  high, high_incl)
    "swipe_velocity": ("float", True, 0.0, 1.0, True),  # (0, 1]
    "pinch_sensitivity": ("float", True, 0.0, 1.0, True),  # (0, 1]
    "pinch_threshold": ("float", True, 0.0, 1.0, True),  # (0, 1]
    "smoothing_window": ("int", False, 1, None, False),  # integer >= 1
    "cooldown_ms": ("float", False, 0, None, False),  # >= 0
    "move_speed": ("float", True, 0.0, None, False),  # > 0
    "resize_step": ("float", True, 0.0, 1.0, True),  # (0, 1]
}


def _range_label(low_excl: bool, low, high, high_incl: bool) -> str:
    """Build a human-readable range label like ``(0, 1]`` or ``>= 1``."""
    if low is not None and high is not None:
        lb = "(" if low_excl else "["
        rb = "]" if high_incl else ")"
        return f"{lb}{low}, {high}{rb}"
    if high is None and low is not None:
        return f"> {low}" if low_excl else f">= {low}"
    return "valid range"


def validate_config(data: dict, backend: str) -> List[str]:
    """Validate a raw config dict and return a list of problem strings.

    An empty list means the config is clean. This is a pure function with no
    side effects: it neither logs nor raises. ``backend`` selects which set of
    action names is considered valid (the active backend's default mapping).

    Detected problems:
      * unknown gesture names in ``mappings`` keys
      * unknown action names in ``mappings`` values
      * out-of-range / wrong-type / unknown ``thresholds``
      * invalid ``backend`` (not "os"/"canvas")
      * invalid ``camera_index`` (negative or non-integer)
    """
    problems: List[str] = []

    raw_backend = data.get("backend", DEFAULT_BACKEND)
    if str(raw_backend).lower() not in ("os", "canvas"):
        problems.append(
            f"invalid backend {raw_backend!r} (expected 'os' or 'canvas')"
        )

    if "camera_index" in data:
        cam = data["camera_index"]
        if isinstance(cam, bool) or not isinstance(cam, int):
            problems.append(
                f"camera_index={cam!r} is not an integer"
            )
        elif cam < 0:
            problems.append(f"camera_index={cam} is negative")

    valid_actions = VALID_ACTIONS_OS if backend == "os" else VALID_ACTIONS_CANVAS
    mappings = data.get("mappings", {})
    if isinstance(mappings, dict):
        for gesture, action in mappings.items():
            if gesture not in VALID_GESTURES:
                problems.append(f"unknown gesture {gesture!r} in mappings")
            if action not in valid_actions:
                problems.append(
                    f"unknown action {action!r} for gesture {gesture!r} in mappings"
                )
    else:
        problems.append("'mappings' must be an object")

    thresholds = data.get("thresholds", {})
    if isinstance(thresholds, dict):
        for name, value in thresholds.items():
            spec = THRESHOLD_RANGES.get(name)
            if spec is None:
                problems.append(f"unknown threshold {name!r}={value!r}")
                continue
            kind, low_excl, low, high, high_incl = spec
            label = _range_label(low_excl, low, high, high_incl)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(
                    f"threshold {name!r}={value!r} is not a number"
                )
                continue
            if kind == "int" and not isinstance(value, int):
                problems.append(
                    f"threshold {name!r}={value!r} must be an integer"
                )
                continue
            ok = True
            if low is not None:
                ok = ok and (value > low if low_excl else value >= low)
            if high is not None:
                ok = ok and (value <= high if high_incl else value < high)
            if not ok:
                problems.append(
                    f"threshold {name!r}={value} out of range {label}"
                )
    else:
        problems.append("'thresholds' must be an object")

    return problems


@dataclass
class Config:
    """Resolved configuration with defaults applied."""

    mappings: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MAPPINGS))
    thresholds: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    backend: str = DEFAULT_BACKEND
    camera_index: int = DEFAULT_CAMERA_INDEX

    def action_for(self, gesture_type: str) -> str | None:
        """Return the action name mapped to a gesture type, or None."""
        return self.mappings.get(gesture_type)

    def threshold(self, name: str) -> float:
        """Return a threshold value, falling back to the default."""
        return float(self.thresholds.get(name, DEFAULT_THRESHOLDS[name]))


def resolve_camera_index(config_index: int, cli_index: int | None) -> int:
    """Resolve the camera index: a valid CLI override wins over the config.

    ``cli_index`` is ``None`` when ``--camera`` was not passed. Negative indices
    are treated as "unset" and fall back to the config value.
    """
    if cli_index is not None and cli_index >= 0:
        return int(cli_index)
    return int(config_index)


def load_config(path: str | None = None) -> Config:
    """Load configuration from JSON, merging over defaults.

    Missing file or missing keys fall back to the documented defaults.
    """
    path = path or DEFAULT_CONFIG_PATH
    data: Dict = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

    backend = str(data.get("backend", DEFAULT_BACKEND)).lower()
    if backend not in ("os", "canvas"):
        backend = DEFAULT_BACKEND

    # Validate against the resolved backend and warn (non-fatal) for any
    # problems. The app continues with graceful defaults for bad entries.
    for problem in validate_config(data, backend):
        logger.warning("config: %s", problem)

    try:
        camera_index = int(data.get("camera_index", DEFAULT_CAMERA_INDEX))
        if camera_index < 0:
            camera_index = DEFAULT_CAMERA_INDEX
    except (TypeError, ValueError):
        camera_index = DEFAULT_CAMERA_INDEX

    # Pick the base mapping set for the active backend, then overlay any
    # user-supplied mappings so per-gesture overrides still work.
    base = dict(DEFAULT_OS_MAPPINGS) if backend == "os" else dict(DEFAULT_MAPPINGS)
    base.update(data.get("mappings", {}))

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(data.get("thresholds", {}))
    return Config(
        mappings=base,
        thresholds=thresholds,
        backend=backend,
        camera_index=camera_index,
    )


def write_default_config(path: str | None = None) -> str:
    """Write the default gestures.json to disk and return the path."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "backend": DEFAULT_BACKEND,
                "camera_index": DEFAULT_CAMERA_INDEX,
                "mappings": DEFAULT_OS_MAPPINGS,
                "thresholds": DEFAULT_THRESHOLDS,
            },
            fh,
            indent=2,
        )
    return path


if __name__ == "__main__":
    written = write_default_config()
    print(f"Wrote default config to {written}")
