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


class ActionRouter:
    """Maps :class:`Gesture` events to :class:`Workspace` method calls.

    Each action is rate-limited by ``cooldown_ms`` so a single physical gesture
    (which may span many frames) fires at most once per cooldown window.
    """

    def __init__(self, workspace: Workspace, config: Config,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.workspace = workspace
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
        # Continuous actions (move/drag) bypass cooldown; discrete ones honor it.
        continuous = action in {"drag_move", "begin_move",
                                "resize_two_hand", "release"}
        if not continuous and not self._ready(action):
            return False
        result = method(gesture)
        if result:
            self.log.append(f"{gesture.type.value} -> {action}")
            if len(self.log) > 8:
                self.log.pop(0)
        return result

    def handle_all(self, gestures: List[Gesture]) -> int:
        """Handle a batch of gestures; return the number that fired."""
        return sum(1 for g in gestures if self.handle(g))

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
        self.workspace.resize_active_tab(max(0.05, 1.0 - step))
        return True

    def _do_resize_two_hand(self, gesture: Gesture) -> bool:
        # magnitude is the inter-palm distance; map to a scale around 1.0.
        scale = max(0.5, min(1.5, 0.5 + gesture.magnitude))
        self.workspace.resize_active_tab(scale)
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


class OSActionRouter:
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
        self.controller = controller
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
        continuous = action in {"drag_move", "begin_move",
                                "resize_two_hand", "release"}
        if not continuous and not self._ready(action):
            return False
        result = method(gesture)
        if result:
            self.log.append(f"{gesture.type.value} -> {action}")
            if len(self.log) > 8:
                self.log.pop(0)
        return result

    def handle_all(self, gestures: List[Gesture]) -> int:
        """Handle a batch of gestures; return the number that fired."""
        return sum(1 for g in gestures if self.handle(g))

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
        self.controller.resize_window(max(0.05, 1.0 - step))
        return True

    def _do_resize_two_hand(self, gesture: Gesture) -> bool:
        # magnitude is the inter-palm distance; map to a scale around 1.0.
        scale = max(0.5, min(1.5, 0.5 + gesture.magnitude))
        self.controller.resize_window(scale)
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
