"""OpenCV rendering of the workspace and camera overlay.

``cv2`` is import-guarded so this module imports even when OpenCV is missing
(``CV2_AVAILABLE`` reports whether drawing is possible). All drawing functions
no-op gracefully when cv2 or numpy frames are unavailable.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

try:  # pragma: no cover - exercised only with cv2 installed
    import cv2

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from workspace import Workspace

# Colors (BGR)
BG = (32, 32, 36)
TAB = (70, 70, 78)
TAB_ACTIVE = (40, 140, 240)
TEXT = (235, 235, 235)
ACCENT = (0, 200, 120)


def blank_canvas(width: int, height: int):
    """Create a blank BGR canvas. Returns None if numpy is unavailable."""
    if np is None:
        return None
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = BG
    return canvas


def _draw_tab(canvas, tab) -> None:
    color = TAB_ACTIVE if tab.active else TAB
    cv2.rectangle(canvas, (tab.x, tab.y), (tab.x + tab.w, tab.y + tab.h),
                  color, thickness=-1)
    cv2.rectangle(canvas, (tab.x, tab.y), (tab.x + tab.w, tab.y + tab.h),
                  ACCENT if tab.active else (110, 110, 118), thickness=2)
    cv2.putText(canvas, f"{tab.id}: {tab.title}", (tab.x + 12, tab.y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)


def render_workspace(workspace: Workspace, help_lines: Optional[Sequence[str]] = None):
    """Render all tabs onto a fresh canvas and return it (or None)."""
    if not CV2_AVAILABLE or np is None:
        return None
    canvas = blank_canvas(workspace.canvas_w, workspace.canvas_h)
    mode = "DOUBLE" if workspace.double_view else "SINGLE"
    cv2.putText(canvas, f"Workspace [{mode}]  tabs={len(workspace.tabs)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)
    for tab in workspace.tabs:
        _draw_tab(canvas, tab)
    if help_lines:
        y = workspace.canvas_h - 18 * (len(help_lines) + 1)
        for line in help_lines:
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 180, 190), 1, cv2.LINE_AA)
            y += 18
    return canvas


def draw_landmarks(frame, hands_points: List, color=ACCENT):
    """Draw simple landmark dots for each hand onto a camera frame."""
    if not CV2_AVAILABLE:
        return frame
    h, w = frame.shape[:2]
    for pts in hands_points:
        for (x, y) in pts:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, color, -1)
    return frame


def wrap_lines(lines: Sequence[str], max_chars: int) -> List[str]:
    """Word-wrap each input line to at most ``max_chars`` characters.

    Pure helper (no cv2/numpy). Words longer than ``max_chars`` are hard-broken
    so a single long token never overflows the panel. Empty strings are
    preserved as empty lines. A non-positive ``max_chars`` disables wrapping and
    returns the input lines unchanged.
    """
    if max_chars <= 0:
        return list(lines)
    wrapped: List[str] = []
    for line in lines:
        if line == "":
            wrapped.append("")
            continue
        cur = ""
        for word in line.split(" "):
            # Hard-break any word that is itself wider than the limit.
            while len(word) > max_chars:
                if cur:
                    wrapped.append(cur)
                    cur = ""
                wrapped.append(word[:max_chars])
                word = word[max_chars:]
            if not cur:
                cur = word
            elif len(cur) + 1 + len(word) <= max_chars:
                cur = f"{cur} {word}"
            else:
                wrapped.append(cur)
                cur = word
        wrapped.append(cur)
    return wrapped


def overlay_text_panel(frame, lines: List[str], max_chars: int = 0):
    """Draw a semi-transparent help/log panel over a (mirrored) camera frame.

    Used in OS window-control mode where there is no tab canvas — the user sees
    the camera feed plus what gesture/action fired. Returns the frame (or None).

    ``max_chars`` optionally word-wraps long lines (see :func:`wrap_lines`); the
    default of 0 preserves the original one-line-per-entry behavior.
    """
    if not CV2_AVAILABLE or frame is None:
        return frame
    mirrored = cv2.flip(frame, 1)
    if not lines:
        return mirrored
    lines = wrap_lines(lines, max_chars)
    h, w = mirrored.shape[:2]
    panel_h = 22 * len(lines) + 16
    overlay = mirrored.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), BG, thickness=-1)
    cv2.addWeighted(overlay, 0.6, mirrored, 0.4, 0, mirrored)
    y = 24
    for line in lines:
        cv2.putText(mirrored, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, TEXT, 1, cv2.LINE_AA)
        y += 22
    return mirrored


def overlay_camera(canvas, frame, scale: float = 0.25):
    """Mirror the camera frame and paste it into the canvas top-right corner."""
    if not CV2_AVAILABLE or canvas is None or frame is None:
        return canvas
    mirrored = cv2.flip(frame, 1)
    fh, fw = mirrored.shape[:2]
    tw, th = int(fw * scale), int(fh * scale)
    thumb = cv2.resize(mirrored, (tw, th))
    ch, cw = canvas.shape[:2]
    x0, y0 = cw - tw - 10, 10
    canvas[y0:y0 + th, x0:x0 + tw] = thumb
    cv2.rectangle(canvas, (x0, y0), (x0 + tw, y0 + th), ACCENT, 1)
    return canvas
