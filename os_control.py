"""macOS window controller driven by ``osascript`` (AppleScript + System Events).

This module drives **real** macOS windows/applications on screen — cycling the
frontmost app, moving/resizing the front window, and tiling windows left/right —
so hand gestures can manipulate Notes, Terminal, VS Code, Figma, etc.

Design for testability
-----------------------
AppleScript *string construction* is split into module-level **pure functions**
(``build_*`` and the cycle-index math) from the *execution*, which is a single
``_run`` method that shells out to ``osascript``. The pure builders take no
permissions and run no subprocess, so they are fully unit-testable.

Runtime requirements
---------------------
Driving other apps' windows via System Events requires **Accessibility**
permission for the host process (your Terminal/IDE). Grant it under
System Settings -> Privacy & Security -> Accessibility, then quit & reopen the
terminal. Camera permission is separate (needed by ``main.py``).

Nothing here imports cv2 or mediapipe.
"""

from __future__ import annotations

import subprocess
from typing import Callable, List, Optional, Tuple

# Seconds to wait for an ``osascript`` invocation before giving up.
OSASCRIPT_TIMEOUT_S = 5

# Default assumed screen size (pixels) until ``refresh_screen_size`` queries the
# real display. Matches a common 1440x900 laptop panel.
DEFAULT_SCREEN_W = 1440
DEFAULT_SCREEN_H = 900

# Minimum window size (pixels) enforced by ``clamp_size`` so a window can't be
# shrunk into nothing.
MIN_WINDOW_W = 200
MIN_WINDOW_H = 150
# Effectively-unbounded maximum used when no screen size is supplied.
MAX_WINDOW_DIM = 100000


# ---------------------------------------------------------------------------
# Pure cycle-index math (unit-testable, no osascript)
# ---------------------------------------------------------------------------
def next_index(current: int, count: int) -> int:
    """Index of the next item in a wrapping cycle of ``count`` items."""
    if count <= 0:
        return -1
    return (current + 1) % count


def prev_index(current: int, count: int) -> int:
    """Index of the previous item in a wrapping cycle of ``count`` items."""
    if count <= 0:
        return -1
    return (current - 1) % count


def clamp_size(
    w: float,
    h: float,
    min_w: int = MIN_WINDOW_W,
    min_h: int = MIN_WINDOW_H,
    max_w: int = MAX_WINDOW_DIM,
    max_h: int = MAX_WINDOW_DIM,
) -> Tuple[int, int]:
    """Clamp a width/height to sane minimums and the screen maximum."""
    cw = int(max(min_w, min(max_w, round(w))))
    ch = int(max(min_h, min(max_h, round(h))))
    return cw, ch


def map_normalized_to_screen(
    nx: float, ny: float, screen_w: int, screen_h: int, mirror_x: bool = True
) -> Tuple[int, int]:
    """Map a normalized [0,1] hand centroid to screen pixel coordinates.

    The camera preview is mirrored for the user, so by default we mirror x so
    moving the hand right moves the window right on screen.
    """
    x = (1.0 - nx) if mirror_x else nx
    px = int(max(0, min(screen_w, round(x * screen_w))))
    py = int(max(0, min(screen_h, round(ny * screen_h))))
    return px, py


# ---------------------------------------------------------------------------
# Pure AppleScript string builders (unit-testable, no osascript)
# ---------------------------------------------------------------------------
def build_list_apps_script() -> str:
    """AppleScript returning visible, non-background app process names."""
    return (
        'tell application "System Events" to get name of every application '
        "process whose visible is true and background only is false"
    )


def build_frontmost_app_script() -> str:
    """AppleScript returning the name of the frontmost application process."""
    return (
        'tell application "System Events" to get name of first application '
        "process whose frontmost is true"
    )


def build_activate_app_script(app_name: str) -> str:
    """AppleScript bringing ``app_name`` to the front."""
    safe = app_name.replace('"', '\\"')
    return (
        'tell application "System Events" to set frontmost of '
        f'(first application process whose name is "{safe}") to true'
    )


def build_get_position_script() -> str:
    """AppleScript returning the front window's position as ``x, y``."""
    return (
        'tell application "System Events" to get position of first window '
        "of (first application process whose frontmost is true)"
    )


def build_get_size_script() -> str:
    """AppleScript returning the front window's size as ``w, h``."""
    return (
        'tell application "System Events" to get size of first window '
        "of (first application process whose frontmost is true)"
    )


def build_set_position_script(x: int, y: int) -> str:
    """AppleScript setting the front window's position to ``(x, y)``."""
    return (
        'tell application "System Events" to set position of first window '
        "of (first application process whose frontmost is true) to "
        f"{{{int(x)}, {int(y)}}}"
    )


def build_set_size_script(w: int, h: int) -> str:
    """AppleScript setting the front window's size to ``(w, h)``."""
    return (
        'tell application "System Events" to set size of first window '
        "of (first application process whose frontmost is true) to "
        f"{{{int(w)}, {int(h)}}}"
    )


def build_screen_size_script() -> str:
    """AppleScript returning the primary display size as ``w, h``."""
    return (
        'tell application "Finder" to get bounds of window of desktop'
    )


def build_set_app_window_bounds_script(
    app_name: str, x: int, y: int, w: int, h: int
) -> str:
    """AppleScript positioning and sizing ``app_name``'s front window."""
    safe = app_name.replace('"', '\\"')
    return (
        'tell application "System Events" to tell '
        f'(first application process whose name is "{safe}") to '
        "set position of first window to "
        f"{{{int(x)}, {int(y)}}}\n"
        'tell application "System Events" to tell '
        f'(first application process whose name is "{safe}") to '
        "set size of first window to "
        f"{{{int(w)}, {int(h)}}}"
    )


def parse_coords(output: str) -> Tuple[int, int]:
    """Parse ``"x, y"`` (osascript list output) into a pair of ints.

    Raises ``ValueError`` if fewer than two numbers are present.
    """
    parts = [p.strip() for p in output.replace("{", "").replace("}", "").split(",")]
    nums = [int(round(float(p))) for p in parts if p]
    if len(nums) < 2:
        raise ValueError(f"expected two numbers, got: {output!r}")
    return nums[0], nums[1]


def parse_bounds(output: str) -> Tuple[int, int, int, int]:
    """Parse ``"l, t, r, b"`` desktop bounds into ``(left, top, right, bottom)``."""
    parts = [p.strip() for p in output.replace("{", "").replace("}", "").split(",")]
    nums = [int(round(float(p))) for p in parts if p]
    if len(nums) < 4:
        raise ValueError(f"expected four numbers, got: {output!r}")
    return nums[0], nums[1], nums[2], nums[3]


# ---------------------------------------------------------------------------
# Controller (executes the scripts)
# ---------------------------------------------------------------------------
class OSWindowController:
    """Drives real macOS windows via ``osascript``.

    All AppleScript text comes from the pure ``build_*`` functions above; this
    class only adds execution, caching of screen size, and the app-cycle state.
    Errors (permission denied, no window) are caught and returned as short
    strings rather than raised, so the camera loop never crashes.
    """

    def __init__(self, screen_w: int = DEFAULT_SCREEN_W,
                 screen_h: int = DEFAULT_SCREEN_H) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._cycle: List[str] = []
        self._cycle_index: int = -1
        self._prev_front: Optional[str] = None
        self._moving = False

    # ----- execution ----------------------------------------------------
    def _run(self, script: str) -> Tuple[bool, str]:
        """Run an AppleScript via osascript. Returns ``(ok, output_or_error)``."""
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=OSASCRIPT_TIMEOUT_S,
            )
        except FileNotFoundError:
            return False, "osascript not found (macOS only)"
        except subprocess.TimeoutExpired:
            return False, "osascript timed out"
        if proc.returncode != 0:
            return False, proc.stderr.strip() or "osascript error"
        return True, proc.stdout.strip()

    # ----- screen -------------------------------------------------------
    def refresh_screen_size(self) -> str:
        """Query the primary display size and cache it."""
        ok, out = self._run(build_screen_size_script())
        if not ok:
            return f"screen-size failed: {out}"
        try:
            left, top, right, bottom = parse_bounds(out)
            self.screen_w = right - left
            self.screen_h = bottom - top
        except ValueError as exc:
            return f"screen-size parse failed: {exc}"
        return f"screen={self.screen_w}x{self.screen_h}"

    # ----- app cycling --------------------------------------------------
    def _refresh_cycle(self) -> str:
        """Rebuild the ordered list of cyclable apps and locate the front one."""
        ok, out = self._run(build_list_apps_script())
        if not ok:
            return ""
        apps = [a.strip() for a in out.split(",") if a.strip()]
        self._cycle = apps
        ok2, front = self._run(build_frontmost_app_script())
        if ok2 and front in apps:
            self._cycle_index = apps.index(front)
        elif apps:
            self._cycle_index = 0
        return front if ok2 else ""

    def _cycle_app(self, step: Callable[[int, int], int], label: str) -> str:
        """Cycle to an adjacent visible app using ``step`` for the index math."""
        self._prev_front = self._refresh_cycle()
        if not self._cycle:
            return "no apps to cycle"
        self._cycle_index = step(self._cycle_index, len(self._cycle))
        name = self._cycle[self._cycle_index]
        ok, out = self._run(build_activate_app_script(name))
        return f"{label} -> {name}" if ok else f"{label} failed: {out}"

    def next_app(self) -> str:
        """Bring the next visible app to the front (wraps)."""
        return self._cycle_app(next_index, "next_app")

    def prev_app(self) -> str:
        """Bring the previous visible app to the front (wraps)."""
        return self._cycle_app(prev_index, "prev_app")

    # ----- move ---------------------------------------------------------
    def _query_coords(self, script: str) -> Optional[Tuple[int, int]]:
        """Run a coord-returning script and parse it, or None on failure."""
        ok, out = self._run(script)
        if not ok:
            return None
        try:
            return parse_coords(out)
        except ValueError:
            return None

    def _front_position(self) -> Optional[Tuple[int, int]]:
        return self._query_coords(build_get_position_script())

    def _front_size(self) -> Optional[Tuple[int, int]]:
        return self._query_coords(build_get_size_script())

    def set_window_position(self, x: int, y: int) -> str:
        """Set the front window's absolute position."""
        ok, out = self._run(build_set_position_script(int(x), int(y)))
        return f"move -> ({int(x)}, {int(y)})" if ok else f"move failed: {out}"

    def move_window(self, dx: int, dy: int) -> str:
        """Move the front window by a relative delta."""
        pos = self._front_position()
        if pos is None:
            return "move failed: no window"
        return self.set_window_position(pos[0] + int(dx), pos[1] + int(dy))

    # ----- resize -------------------------------------------------------
    def set_window_size(self, w: int, h: int) -> str:
        """Set the front window's size, clamped to sane bounds."""
        cw, ch = clamp_size(w, h, max_w=self.screen_w, max_h=self.screen_h)
        ok, out = self._run(build_set_size_script(cw, ch))
        return f"resize -> ({cw}, {ch})" if ok else f"resize failed: {out}"

    def resize_window(self, scale: float) -> str:
        """Multiply the front window's current size by ``scale`` (clamped)."""
        size = self._front_size()
        if size is None:
            return "resize failed: no window"
        return self.set_window_size(size[0] * scale, size[1] * scale)

    # ----- tiling / split ----------------------------------------------
    def _tile(self, label: str, x: int, w: int) -> str:
        """Move+resize the front window to a full-height pane at ``x`` of width ``w``."""
        ok, out = self._run(build_set_position_script(x, 0))
        if not ok:
            return f"{label} failed: {out}"
        ok2, out2 = self._run(build_set_size_script(w, self.screen_h))
        return label if ok2 else f"{label} failed: {out2}"

    def tile_left(self) -> str:
        """Tile the front window to the left half of the main screen."""
        return self._tile("tile_left", 0, self.screen_w // 2)

    def tile_right(self) -> str:
        """Tile the front window to the right half of the main screen."""
        half = self.screen_w // 2
        return self._tile("tile_right", half, self.screen_w - half)

    def toggle_split(self) -> str:
        """Tile the front window left and the previously-front app right."""
        ok, front = self._run(build_frontmost_app_script())
        left_msg = self.tile_left()
        prev = self._prev_front
        if prev and ok and prev != front:
            half = self.screen_w // 2
            self._run(
                build_set_app_window_bounds_script(
                    prev, half, 0, self.screen_w - half, self.screen_h
                )
            )
            return f"split: {front} | {prev}"
        self._prev_front = front if ok else self._prev_front
        return f"split (left only): {left_msg}"

    # ----- move-mode flags (mirrors canvas router semantics) -----------
    def begin_move(self) -> str:
        self._moving = True
        return "move mode on"

    def end_move(self) -> str:
        was = self._moving
        self._moving = False
        return "move mode off" if was else ""
