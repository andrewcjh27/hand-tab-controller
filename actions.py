"""Route gesture events onto workspace actions, with debounce/cooldown.

Pure logic -- no camera imports. A monotonic clock is injectable so tests can
control time without sleeping.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List

from config import Config
from gestures import Gesture, GestureType
from workspace import Workspace

# Actions that fire on every frame they're active (continuous motion) and so
# bypass the per-action cooldown debounce; all other actions honor it.
CONTINUOUS_ACTIONS = frozenset(
    {"drag_move", "begin_move", "resize_two_hand", "release"}
)

# Max number of recent action entries kept in ``log`` (for the on-screen panel).
MAX_LOG_ENTRIES = 8

# Resize-scale bounds. ``MIN_SHRINK_SCALE`` floors a single shrink step so a
# large ``resize_step`` can't invert/collapse the window. ``TWO_HAND_*`` clamp
# the inter-palm distance once mapped to a scale around 1.0.
MIN_SHRINK_SCALE = 0.05
TWO_HAND_MIN_SCALE = 0.5
TWO_HAND_MAX_SCALE = 1.5


def two_hand_scale(magnitude: float) -> float:
    """Map an inter-palm distance to a clamped resize scale around 1.0."""
    return max(TWO_HAND_MIN_SCALE, min(TWO_HAND_MAX_SCALE, TWO_HAND_MIN_SCALE + magnitude))


class _BaseRouter:
    """Shared gesture dispatch: cooldown debounce, log, and ``handle`` loop.

    Subclasses provide the ``_do_<action>`` methods that actually mutate a
    target (a :class:`Workspace` or an OS window controller).
    """

    def __init__(self, config: Config,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self._clock = clock
        self._last_fire: Dict[str, float] = {}
        self._moving = False
        self.log: List[str] = []

    # ----- cooldown -----------------------------------------------------
    def _ready(self, action: str) -> bool:
        cooldown = self.config.threshold("cooldown_ms") / 1000.0
        now = self._clock()
        last = self._last_fire.get(action, -1e9)
        if now - last >= cooldown:
            self._last_fire[action] = now
            return True
        return False

    # ----- dispatch -----------------------------------------------------
    def handle(self, gesture: Gesture) -> bool:
        """Handle one gesture event. Returns True if an action fired."""
        action = self.config.action_for(gesture.type.value)
        if action is None:
            return False
        method = getattr(self, f"_do_{action}", None)
        if method is None:
            return False
        if action not in CONTINUOUS_ACTIONS and not self._ready(action):
            return False
        result = method(gesture)
        if result:
            self.log.append(f"{gesture.type.value} -> {action}")
            if len(self.log) > MAX_LOG_ENTRIES:
                self.log.pop(0)
        return result

    def handle_all(self, gestures: List[Gesture]) -> int:
        """Handle a batch of gestures; return the number that fired."""
        return sum(1 for g in gestures if self.handle(g))


class ActionRouter(_BaseRouter):
    """Maps :class:`Gesture` events to :class:`Workspace` method calls.

    Each action is rate-limited by ``cooldown_ms`` so a single physical gesture
    (which may span many frames) fires at most once per cooldown window.
    """

    def __init__(self, workspace: Workspace, config: Config,
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(config, clock)
        self.workspace = workspace

    # ----- action implementations --------------------------------------
    def _do_next_tab(self, gesture: Gesture) -> bool:
        self.workspace.next_tab()
        return True

    def _do_prev_tab(self, gesture: Gesture) -> bool:
        self.workspace.prev_tab()
        return True

    def _do_resize_grow(self, gesture: Gesture) -> bool:
        step = self.config.threshold("resize_step")
        self.workspace.resize_active_tab(1.0 + step)
        return True

    def _do_resize_shrink(self, gesture: Gesture) -> bool:
        step = self.config.threshold("resize_step")
        self.workspace.resize_active_tab(max(MIN_SHRINK_SCALE, 1.0 - step))
        return True

    def _do_resize_two_hand(self, gesture: Gesture) -> bool:
        self.workspace.resize_active_tab(two_hand_scale(gesture.magnitude))
        return True

    def _do_begin_move(self, gesture: Gesture) -> bool:
        self._moving = True
        self._move_anchor = gesture.position
        return True

    def _do_drag_move(self, gesture: Gesture) -> bool:
        if not self._moving:
            return False
        speed = self.config.threshold("move_speed")
        anchor = getattr(self, "_move_anchor", gesture.position)
        dx = (gesture.position[0] - anchor[0]) * speed
        dy = (gesture.position[1] - anchor[1]) * speed
        self._move_anchor = gesture.position
        self.workspace.move_active_tab(int(round(dx)), int(round(dy)))
        return True

    def _do_release(self, gesture: Gesture) -> bool:
        was = self._moving
        self._moving = False
        return was

    def _do_toggle_double_view(self, gesture: Gesture) -> bool:
        self.workspace.toggle_double_view()
        return True


class OSActionRouter(_BaseRouter):
    """Maps :class:`Gesture` events to :class:`OSWindowController` calls.

    Mirrors :class:`ActionRouter`'s debounce semantics and ``log`` so main.py
    can swap routers based on the configured ``backend`` without other changes.
    The gesture *events* are identical; only the targeted actions differ.

    Mapped actions: ``prev_app``, ``next_app``, ``resize_grow``,
    ``resize_shrink``, ``resize_two_hand``, ``begin_move``, ``drag_move``,
    ``release``, ``toggle_split``.
    """

    def __init__(self, controller, config: Config,
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(config, clock)
        self.controller = controller

    # ----- action implementations --------------------------------------
    def _do_next_app(self, gesture: Gesture) -> bool:
        self.controller.next_app()
        return True

    def _do_prev_app(self, gesture: Gesture) -> bool:
        self.controller.prev_app()
        return True

    def _do_resize_grow(self, gesture: Gesture) -> bool:
        step = self.config.threshold("resize_step")
        self.controller.resize_window(1.0 + step)
        return True

    def _do_resize_shrink(self, gesture: Gesture) -> bool:
        step = self.config.threshold("resize_step")
        self.controller.resize_window(max(MIN_SHRINK_SCALE, 1.0 - step))
        return True

    def _do_resize_two_hand(self, gesture: Gesture) -> bool:
        self.controller.resize_window(two_hand_scale(gesture.magnitude))
        return True

    def _do_begin_move(self, gesture: Gesture) -> bool:
        self._moving = True
        self._move_anchor = gesture.position
        self.controller.begin_move()
        return True

    def _do_drag_move(self, gesture: Gesture) -> bool:
        if not self._moving:
            return False
        # Map the normalized hand centroid to absolute screen coords so the
        # front window follows the hand.
        from os_control import map_normalized_to_screen

        x, y = map_normalized_to_screen(
            gesture.position[0], gesture.position[1],
            self.controller.screen_w, self.controller.screen_h,
        )
        self.controller.set_window_position(x, y)
        return True

    def _do_release(self, gesture: Gesture) -> bool:
        was = self._moving
        self._moving = False
        self.controller.end_move()
        return was

    def _do_toggle_split(self, gesture: Gesture) -> bool:
        self.controller.toggle_split()
        return True
