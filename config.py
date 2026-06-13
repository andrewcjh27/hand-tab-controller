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
import os
from dataclasses import dataclass, field
from typing import Dict

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
    camera_index = int(data.get("camera_index", DEFAULT_CAMERA_INDEX))

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
