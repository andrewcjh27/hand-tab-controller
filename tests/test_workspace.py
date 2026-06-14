"""Unit tests for the workspace model and action routing. No camera needed."""

import pytest

from actions import ActionRouter
from config import Config
from gestures import Gesture, GestureType
from workspace import Workspace


def make_workspace(n=3):
    ws = Workspace(canvas_w=1000, canvas_h=600)
    for i in range(n):
        ws.add_tab(f"Tab{i}")
    ws.set_active(0)
    return ws


def test_add_and_active():
    ws = make_workspace(2)
    assert len(ws.tabs) == 2
    assert ws.active_tab.title == "Tab0"


def test_next_prev_wrap():
    ws = make_workspace(3)
    assert ws.next_tab().title == "Tab1"
    assert ws.next_tab().title == "Tab2"
    assert ws.next_tab().title == "Tab0"  # wraps
    assert ws.prev_tab().title == "Tab2"  # wraps backward


def test_remove_keeps_active_valid():
    ws = make_workspace(3)
    ws.set_active(2)
    last_id = ws.active_tab.id
    assert ws.remove_tab(last_id)
    assert ws.active_index == len(ws.tabs) - 1
    assert ws.active_tab is not None


def test_move_clamped():
    ws = make_workspace(1)
    ws.move_active_tab(-9999, -9999)
    assert ws.active_tab.x == 0 and ws.active_tab.y == 0
    ws.move_active_tab(9999, 9999)
    assert ws.active_tab.x == ws.canvas_w - ws.active_tab.w


def test_resize_bounds():
    ws = make_workspace(1)
    ws.active_tab.w = ws.active_tab.h = 100
    ws.resize_active_tab(2.0)
    assert ws.active_tab.w == 200
    ws.resize_active_tab(0.001, min_size=80)
    assert ws.active_tab.w == 80


def test_double_view_layout():
    ws = make_workspace(2)
    ws.set_active(0)
    assert ws.toggle_double_view() is True
    left, right = ws.tabs[0], ws.tabs[1]
    assert left.x < right.x
    assert abs((left.w) - (right.w)) <= 1
    assert ws.toggle_double_view() is False


def test_split_view_idempotent():
    ws = make_workspace(2)
    assert ws.split_view() is True
    assert ws.split_view() is True


# ----- action router -------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_swipe_cooldown_debounce():
    ws = make_workspace(3)
    clock = FakeClock()
    cfg = Config()
    cfg.thresholds["cooldown_ms"] = 600
    router = ActionRouter(ws, cfg, clock=clock)
    g = Gesture(GestureType.SWIPE_RIGHT, position=(0.5, 0.5))

    assert router.handle(g) is True          # fires
    assert ws.active_tab.title == "Tab1"
    assert router.handle(g) is False         # within cooldown -> debounced
    assert ws.active_tab.title == "Tab1"
    clock.t = 1.0                            # advance past cooldown
    assert router.handle(g) is True
    assert ws.active_tab.title == "Tab2"


def test_resize_actions_via_router():
    ws = make_workspace(1)
    ws.active_tab.w = ws.active_tab.h = 100
    cfg = Config()
    cfg.thresholds["resize_step"] = 0.5
    router = ActionRouter(ws, cfg, clock=FakeClock())
    router.handle(Gesture(GestureType.SPREAD, magnitude=0.7))
    assert ws.active_tab.w == 150


def test_drag_move_requires_begin():
    ws = make_workspace(1)
    cfg = Config()
    router = ActionRouter(ws, cfg, clock=FakeClock())
    # drag without begin_move does nothing
    assert router.handle(Gesture(GestureType.GRAB, position=(0.5, 0.5))) is False
    router.handle(Gesture(GestureType.POINT, position=(0.5, 0.5)))  # begin_move
    moved = router.handle(Gesture(GestureType.GRAB, position=(0.7, 0.5)))
    assert moved is True


def test_v_sign_toggles_double_view_by_default():
    ws = make_workspace(2)
    cfg = Config()  # V_SIGN -> toggle_double_view is a default mapping
    router = ActionRouter(ws, cfg, clock=FakeClock())
    assert router.handle(Gesture(GestureType.V_SIGN)) is True
    assert ws.double_view is True


def test_toggle_double_view_remappable():
    ws = make_workspace(2)
    cfg = Config()
    cfg.mappings["OPEN_PALM"] = "toggle_double_view"
    router = ActionRouter(ws, cfg, clock=FakeClock())
    assert router.handle(Gesture(GestureType.OPEN_PALM)) is True
    assert ws.double_view is True


# ----- set_config (live reload) --------------------------------------------


def test_set_config_swaps_mapping():
    ws = make_workspace(3)
    cfg = Config()  # SWIPE_RIGHT -> next_tab by default
    router = ActionRouter(ws, cfg, clock=FakeClock())
    router.handle(Gesture(GestureType.SWIPE_RIGHT))
    assert ws.active_tab.title == "Tab1"

    # New config remaps SWIPE_RIGHT to prev_tab; subsequent events use it.
    new_cfg = Config()
    new_cfg.mappings["SWIPE_RIGHT"] = "prev_tab"
    router.set_config(new_cfg)
    assert router.config is new_cfg
    router.handle(Gesture(GestureType.SWIPE_RIGHT))
    assert ws.active_tab.title == "Tab0"  # went backward, not forward


def test_set_config_applies_new_cooldown():
    ws = make_workspace(3)
    clock = FakeClock()
    cfg = Config()
    cfg.thresholds["cooldown_ms"] = 10_000
    router = ActionRouter(ws, cfg, clock=clock)
    g = Gesture(GestureType.SWIPE_RIGHT)
    assert router.handle(g) is True
    assert router.handle(g) is False  # long cooldown blocks the repeat

    # Reload with a zero cooldown -> the next event fires immediately.
    new_cfg = Config()
    new_cfg.thresholds["cooldown_ms"] = 0
    router.set_config(new_cfg)
    assert router.handle(g) is True


def test_set_config_keeps_log():
    ws = make_workspace(3)
    router = ActionRouter(ws, Config(), clock=FakeClock())
    router.handle(Gesture(GestureType.SWIPE_RIGHT))
    assert router.log  # has an entry
    before = list(router.log)
    router.set_config(Config())
    assert router.log == before  # log preserved across reload
