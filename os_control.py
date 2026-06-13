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

# Height (pixels) of the macOS menu bar at the top of the screen. Tiling reserves
# this so tiled windows sit *below* the menu bar instead of underneath it.
MENU_BAR_HEIGHT = 25

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


def visible_frame(
    screen_w: int, screen_h: int, menubar_h: int = MENU_BAR_HEIGHT
) -> Tuple[int, int, int, int]:
    """Usable screen rect ``(x, y, w, h)`` below the macOS menu bar.

    The menu bar occupies the top ``menubar_h`` pixels, so tiled windows should
    start at ``y = menubar_h`` and be ``menubar_h`` shorter than the full screen.
    ``menubar_h`` is clamped to ``[0, screen_h]`` so the returned height is never
    negative.
    """
    mb = max(0, min(menubar_h, screen_h))
    return (0, mb, screen_w, screen_h - mb)


def display_for_point(
    px: float, py: float, displays: List[Tuple[int, int, int, int]]
) -> Optional[Tuple[int, int, int, int]]:
    """Pick the display frame ``(x, y, w, h)`` that a point belongs to.

    Returns the first frame whose rectangle *contains* ``(px, py)``. If no frame
    contains the point (e.g. the window is off-screen), returns the frame whose
    center is nearest to the point so it still resolves to *some* display.
    Returns ``None`` only when ``displays`` is empty.
    """
    if not displays:
        return None
    for (x, y, w, h) in displays:
        if x <= px < x + w and y <= py < y + h:
            return (x, y, w, h)
    # Nothing contained the point: fall back to the nearest display center.
    def _dist2(frame: Tuple[int, int, int, int]) -> float:
        x, y, w, h = frame
        cx, cy = x + w / 2.0, y + h / 2.0
        return (cx - px) ** 2 + (cy - py) ** 2

    return min(displays, key=_dist2)


def clamp_rect_to_frame(
    x: float, y: float, w: float, h: float, frame: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Clamp a window rect ``(x, y, w, h)`` so it fits inside ``frame``.

    Size is preserved where possible; if the window is larger than ``frame`` in
    either dimension it is shrunk to fit. The position is then nudged so the
    (possibly shrunk) window lies fully within ``frame``. Returns integers.
    """
    fx, fy, fw, fh = frame
    cw = min(int(round(w)), fw)
    ch = min(int(round(h)), fh)
    cx = int(round(x))
    cy = int(round(y))
    # Keep the right/bottom edges inside the frame, then the left/top edges.
    cx = min(cx, fx + fw - cw)
    cy = min(cy, fy + fh - ch)
    cx = max(cx, fx)
    cy = max(cy, fy)
    return cx, cy, cw, ch


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


def build_list_displays_script() -> str:
    """AppleScript returning every display's size as ``w1, h1, w2, h2, ...``.

    Enumerating *per-display origins* via System Events is unreliable, so this
    only asks for desktop sizes. ``parse_display_frames`` turns the flat number
    list into frames; callers MUST fall back to single-display behavior when
    fewer than two displays are returned (see ``OSWindowController``).
    """
    return (
        'tell application "Finder" to get bounds of every window of desktop'
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


def parse_display_frames(output: str) -> List[Tuple[int, int, int, int]]:
    """Parse osascript bounds output into a list of display frames.

    Accepts either a flat ``l, t, r, b, l, t, r, b, ...`` stream (groups of
    four, the natural ``bounds`` output) and converts each ``(l, t, r, b)`` to a
    frame ``(x, y, w, h)``. Any trailing numbers that don't form a complete
    group of four are ignored. Returns an empty list when nothing parses.
    """
    parts = [p.strip() for p in output.replace("{", "").replace("}", "").split(",")]
    nums: List[int] = []
    for p in parts:
        if not p:
            continue
        try:
            nums.append(int(round(float(p))))
        except ValueError:
            continue
    frames: List[Tuple[int, int, int, int]] = []
    for i in range(0, len(nums) - 3, 4):
        left, top, right, bottom = nums[i : i + 4]
        frames.append((left, top, right - left, bottom - top))
    return frames


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
                 screen_h: int = DEFAULT_SCREEN_H,
                 menubar_h: int = MENU_BAR_HEIGHT) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.menubar_h = menubar_h
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

    # ----- display selection -------------------------------------------
    def _enumerate_displays(self) -> List[Tuple[int, int, int, int]]:
        """Query the connected displays' frames, or ``[]`` on any failure.

        Per-display origin enumeration via osascript is unreliable, so this may
        legitimately return fewer frames than the user has displays. Callers
        treat ``<2`` results as "single display" and fall back accordingly.
        """
        ok, out = self._run(build_list_displays_script())
        if not ok:
            return []
        try:
            return parse_display_frames(out)
        except ValueError:
            return []

    def _primary_visible_frame(self) -> Tuple[int, int, int, int]:
        """The cached primary display's usable rect (below the menu bar)."""
        return visible_frame(self.screen_w, self.screen_h, self.menubar_h)

    def _active_display_frame(self) -> Tuple[int, int, int, int]:
        """Visible frame of the display the front window is on.

        Falls back to the primary visible frame whenever display enumeration
        fails, returns <2 displays, or the front window position is unknown —
        so behavior on a single-display Mac matches the original primary-only
        logic. The menu bar is assumed to occupy the top of *every* display
        (a simplification; see ``visible_frame``).
        """
        displays = self._enumerate_displays()
        if len(displays) < 2:
            return self._primary_visible_frame()
        pos = self._front_position()
        if pos is None:
            return self._primary_visible_frame()
        frame = display_for_point(pos[0], pos[1], displays)
        if frame is None:
            return self._primary_visible_frame()
        fx, fy, fw, fh = frame
        # Reserve the menu bar at this display's top.
        _, _, vw, vh = visible_frame(fw, fh, self.menubar_h)
        mb = max(0, min(self.menubar_h, fh))
        return (fx, fy + mb, vw, vh)

    # ----- tiling / split ----------------------------------------------
    def _tile(self, label: str, left_half: bool) -> str:
        """Move+resize the front window to the left or right half of a frame.

        Tiling happens within the *active display's* visible frame (below the
        menu bar) rather than always the primary, so a window on a secondary
        display tiles on that display. ``left_half`` selects the left pane;
        otherwise the right pane (remaining width).
        """
        fx, fy, fw, fh = self._active_display_frame()
        half = fw // 2
        if left_half:
            x, w = fx, half
        else:
            x, w = fx + half, fw - half
        ok, out = self._run(build_set_position_script(x, fy))
        if not ok:
            return f"{label} failed: {out}"
        ok2, out2 = self._run(build_set_size_script(w, fh))
        return label if ok2 else f"{label} failed: {out2}"

    def tile_left(self) -> str:
        """Tile the front window to the left half of its display."""
        return self._tile("tile_left", True)

    def tile_right(self) -> str:
        """Tile the front window to the right half of its display."""
        return self._tile("tile_right", False)

    def toggle_split(self) -> str:
        """Tile the front window left and the previously-front app right.

        Both panes are placed within the active display's visible frame.
        """
        ok, front = self._run(build_frontmost_app_script())
        fx, fy, fw, fh = self._active_display_frame()
        left_msg = self._tile("tile_left", True)
        prev = self._prev_front
        if prev and ok and prev != front:
            half = fw // 2
            self._run(
                build_set_app_window_bounds_script(
                    prev, fx + half, fy, fw - half, fh
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
