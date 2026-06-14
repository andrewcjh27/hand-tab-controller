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
MUTED = (150, 150, 158)


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
    """Draw minimal landmark dots for each hand onto a camera frame.

    Kept deliberately sparse (small dots, single accent color) for a clean look.
    """
    if not CV2_AVAILABLE:
        return frame
    h, w = frame.shape[:2]
    for pts in hands_points:
        for (x, y) in pts:
            cv2.circle(frame, (int(x * w), int(y * h)), 2, color, -1, cv2.LINE_AA)
    return frame


def _draw_translucent_rect(img, pt1, pt2, alpha: float, color=BG) -> None:
    """Blend a filled rectangle over ``img`` in place at the given opacity."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def overlay_minimal(
    frame,
    status: Optional[str] = None,
    action: Optional[str] = None,
    hint: str = "h: help",
    show_help: bool = False,
    help_lines: Optional[Sequence[str]] = None,
    max_chars: int = 0,
):
    """Minimal HUD over a (mirrored) camera frame.

    A single thin translucent bar at the bottom shows the current detected
    gesture on the left and the most recent action (accent-colored) on the right;
    a faint hint sits top-right. Pressing the help key reveals the full mapping
    list in a translucent top panel. Returns the mirrored frame (or ``None``).
    """
    if not CV2_AVAILABLE or frame is None:
        return frame
    mirrored = cv2.flip(frame, 1)
    h, w = mirrored.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Bottom status bar.
    bar_h = 46
    _draw_translucent_rect(mirrored, (0, h - bar_h), (w, h), 0.55)
    label = status if status else "-"
    cv2.putText(mirrored, label, (16, h - 16), font, 0.72, TEXT, 2, cv2.LINE_AA)
    if action:
        (tw, _), _ = cv2.getTextSize(action, font, 0.6, 1)
        cv2.putText(mirrored, action, (w - tw - 16, h - 16), font, 0.6, ACCENT, 1, cv2.LINE_AA)

    if show_help and help_lines:
        lines = wrap_lines(help_lines, max_chars)
        panel_h = 22 * len(lines) + 16
        _draw_translucent_rect(mirrored, (0, 0), (w, panel_h), 0.7)
        y = 24
        for line in lines:
            cv2.putText(mirrored, line, (12, y), font, 0.5, TEXT, 1, cv2.LINE_AA)
            y += 22
    elif hint:
        (hw, _), _ = cv2.getTextSize(hint, font, 0.45, 1)
        cv2.putText(mirrored, hint, (w - hw - 12, 22), font, 0.45, MUTED, 1, cv2.LINE_AA)

    return mirrored


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


def overlay_prompt(frame, title: str, subtitle: str = ""):
    """Centered instruction overlay (used during calibration) on a mirrored frame."""
    if not CV2_AVAILABLE or frame is None:
        return frame
    mirrored = cv2.flip(frame, 1)
    h, w = mirrored.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    _draw_translucent_rect(mirrored, (0, h // 2 - 52), (w, h // 2 + 44), 0.6)
    (tw, _), _ = cv2.getTextSize(title, font, 0.9, 2)
    cv2.putText(mirrored, title, ((w - tw) // 2, h // 2), font, 0.9, TEXT, 2, cv2.LINE_AA)
    if subtitle:
        (sw, _), _ = cv2.getTextSize(subtitle, font, 0.55, 1)
        cv2.putText(mirrored, subtitle, ((w - sw) // 2, h // 2 + 30), font,
                    0.55, MUTED, 1, cv2.LINE_AA)
    return mirrored


def overlay_training(canvas, label: str, instruction: str, effect: str,
                     reps_done: int, reps_total: int,
                     step_index: int, step_total: int, complete: bool = False):
    """Draw the training banner on top of the workspace canvas.

    Shows the current movement, what to do, what it does to the tabs, and rep
    progress — so the user connects each gesture to its on-screen effect.
    """
    if not CV2_AVAILABLE or canvas is None:
        return canvas
    w = canvas.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    _draw_translucent_rect(canvas, (0, 0), (w, 96), 0.78)
    if complete:
        cv2.putText(canvas, "Training complete - thresholds saved", (16, 40),
                    font, 0.7, ACCENT, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Press q to exit", (16, 70), font, 0.5, MUTED, 1, cv2.LINE_AA)
        return canvas
    cv2.putText(canvas, f"Step {step_index}/{step_total}:  {label}", (16, 30),
                font, 0.7, TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, instruction, (16, 56), font, 0.5, TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"-> {effect}", (16, 78), font, 0.5, ACCENT, 1, cv2.LINE_AA)
    # Rep progress dots on the right.
    dot_r, gap = 7, 22
    x0 = w - reps_total * gap - 16
    for i in range(reps_total):
        cx = x0 + i * gap
        filled = i < reps_done
        cv2.circle(canvas, (cx, 30), dot_r, ACCENT if filled else MUTED,
                   -1 if filled else 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{reps_done}/{reps_total}", (x0, 58), font, 0.5, MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "s: skip   q: quit", (w - 170, 80), font, 0.45, MUTED, 1, cv2.LINE_AA)
    return canvas


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
