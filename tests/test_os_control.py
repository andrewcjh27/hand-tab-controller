"""Unit tests for os_control pure logic and the OS action router.

No osascript execution and no permissions: tests target the pure AppleScript
string builders, the cycle-index math, coordinate mapping/parsing, and the
OSActionRouter wired to a fake controller.
"""

import pytest

import os_control as osc
from actions import (
    TWO_HAND_MAX_SCALE,
    TWO_HAND_MIN_SCALE,
    OSActionRouter,
    two_hand_scale,
)
from config import Config, DEFAULT_OS_MAPPINGS, load_config, resolve_camera_index
from gestures import Gesture, GestureType


# ----- cycle-index math ----------------------------------------------------
def test_next_prev_index_wrap():
    assert osc.next_index(0, 3) == 1
    assert osc.next_index(2, 3) == 0  # wraps
    assert osc.prev_index(0, 3) == 2  # wraps backward
    assert osc.prev_index(1, 3) == 0


def test_index_empty_cycle():
    assert osc.next_index(0, 0) == -1
    assert osc.prev_index(0, 0) == -1


# ----- clamp / mapping -----------------------------------------------------
def test_clamp_size_min_and_max():
    assert osc.clamp_size(10, 10) == (200, 150)
    assert osc.clamp_size(5000, 5000, max_w=1440, max_h=900) == (1440, 900)
    assert osc.clamp_size(800, 600) == (800, 600)


def test_map_normalized_mirrors_x_by_default():
    # nx=0 (hand at left of un-mirrored image) maps to right of screen.
    assert osc.map_normalized_to_screen(0.0, 0.0, 1000, 800) == (1000, 0)
    assert osc.map_normalized_to_screen(1.0, 1.0, 1000, 800) == (0, 800)
    assert osc.map_normalized_to_screen(0.5, 0.5, 1000, 800, mirror_x=False) == (500, 400)


# ----- AppleScript builders ------------------------------------------------
def test_build_set_position_script():
    s = osc.build_set_position_script(10, 20)
    assert "set position of first window" in s
    assert "{10, 20}" in s
    assert "frontmost is true" in s


def test_build_set_size_script():
    s = osc.build_set_size_script(640, 480)
    assert "set size of first window" in s
    assert "{640, 480}" in s


def test_build_activate_app_escapes_quotes():
    s = osc.build_activate_app_script('Weird"Name')
    assert '\\"' in s
    assert "set frontmost" in s


def test_build_list_and_front_scripts():
    assert "visible is true" in osc.build_list_apps_script()
    assert "background only is false" in osc.build_list_apps_script()
    assert "frontmost is true" in osc.build_frontmost_app_script()


def test_build_app_window_bounds_sets_both():
    s = osc.build_set_app_window_bounds_script("Notes", 100, 0, 200, 300)
    assert "set position of first window" in s
    assert "set size of first window" in s
    assert "{100, 0}" in s
    assert "{200, 300}" in s


# ----- parsers -------------------------------------------------------------
def test_parse_coords():
    assert osc.parse_coords("10, 20") == (10, 20)
    assert osc.parse_coords("{640, 480}") == (640, 480)


def test_parse_coords_bad():
    with pytest.raises(ValueError):
        osc.parse_coords("7")


def test_parse_bounds():
    assert osc.parse_bounds("0, 0, 1440, 900") == (0, 0, 1440, 900)


# ----- two-hand scale mapping ----------------------------------------------
def test_two_hand_scale_maps_and_clamps():
    assert two_hand_scale(0.5) == 1.0          # 0.5 + 0.5
    assert two_hand_scale(0.0) == TWO_HAND_MIN_SCALE
    assert two_hand_scale(-5.0) == TWO_HAND_MIN_SCALE   # clamped low
    assert two_hand_scale(5.0) == TWO_HAND_MAX_SCALE    # clamped high


# ----- fake controller + router -------------------------------------------
class FakeController:
    """Records calls instead of running osascript."""

    def __init__(self):
        self.screen_w = 1000
        self.screen_h = 800
        self.calls = []

    def next_app(self):
        self.calls.append(("next_app",))

    def prev_app(self):
        self.calls.append(("prev_app",))

    def resize_window(self, scale):
        self.calls.append(("resize_window", round(scale, 3)))

    def set_window_position(self, x, y):
        self.calls.append(("set_window_position", x, y))

    def begin_move(self):
        self.calls.append(("begin_move",))

    def end_move(self):
        self.calls.append(("end_move",))

    def toggle_split(self):
        self.calls.append(("toggle_split",))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_router():
    cfg = Config(mappings=dict(DEFAULT_OS_MAPPINGS), backend="os")
    ctrl = FakeController()
    return OSActionRouter(ctrl, cfg, clock=FakeClock()), ctrl


def test_swipe_routes_to_app_cycling():
    router, ctrl = make_router()
    assert router.handle(Gesture(GestureType.SWIPE_RIGHT)) is True
    assert ctrl.calls[-1] == ("next_app",)
    # within cooldown -> debounced
    assert router.handle(Gesture(GestureType.SWIPE_RIGHT)) is False


def test_swipe_left_prev_app():
    router, ctrl = make_router()
    assert router.handle(Gesture(GestureType.SWIPE_LEFT)) is True
    assert ctrl.calls[-1] == ("prev_app",)


def test_pinch_spread_resize():
    router, ctrl = make_router()
    router.config.thresholds["resize_step"] = 0.1
    router.handle(Gesture(GestureType.SPREAD))
    assert ctrl.calls[-1] == ("resize_window", 1.1)
    router._clock.t = 10.0
    router.handle(Gesture(GestureType.PINCH))
    assert ctrl.calls[-1] == ("resize_window", 0.9)


def test_point_then_grab_moves_window():
    router, ctrl = make_router()
    # grab before point does nothing
    assert router.handle(Gesture(GestureType.GRAB, position=(0.5, 0.5))) is False
    router.handle(Gesture(GestureType.POINT, position=(0.5, 0.5)))  # begin_move
    assert ("begin_move",) in ctrl.calls
    moved = router.handle(Gesture(GestureType.GRAB, position=(0.2, 0.3)))
    assert moved is True
    # nx=0.2 mirrored -> x=0.8*1000=800 ; ny=0.3 -> 240
    assert ctrl.calls[-1] == ("set_window_position", 800, 240)


def test_open_palm_releases():
    router, ctrl = make_router()
    router.handle(Gesture(GestureType.POINT, position=(0.5, 0.5)))
    assert router.handle(Gesture(GestureType.OPEN_PALM)) is True
    assert ("end_move",) in ctrl.calls


def test_v_sign_toggles_split():
    router, ctrl = make_router()
    assert router.handle(Gesture(GestureType.V_SIGN)) is True
    assert ctrl.calls[-1] == ("toggle_split",)


def test_two_hand_pinch_resize_scale():
    router, ctrl = make_router()
    router.handle(Gesture(GestureType.TWO_HAND_PINCH, magnitude=0.7))
    assert ctrl.calls[-1] == ("resize_window", 1.2)


def test_router_log_records_actions():
    router, ctrl = make_router()
    router.handle(Gesture(GestureType.SWIPE_RIGHT))
    assert router.log and "next_app" in router.log[-1]


# ----- config: backend + camera index -------------------------------------
def test_resolve_camera_index_cli_wins():
    assert resolve_camera_index(0, 2) == 2
    assert resolve_camera_index(3, None) == 3
    assert resolve_camera_index(3, -1) == 3  # negative = unset


def test_load_config_os_backend_uses_os_mappings(tmp_path):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"backend": "os", "camera_index": 1}))
    cfg = load_config(str(p))
    assert cfg.backend == "os"
    assert cfg.camera_index == 1
    assert cfg.action_for("SWIPE_RIGHT") == "next_app"


def test_load_config_canvas_backend_uses_tab_mappings(tmp_path):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"backend": "canvas"}))
    cfg = load_config(str(p))
    assert cfg.backend == "canvas"
    assert cfg.action_for("SWIPE_RIGHT") == "next_tab"


def test_load_config_mapping_override(tmp_path):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"backend": "os", "mappings": {"V_SIGN": "next_app"}}))
    cfg = load_config(str(p))
    assert cfg.action_for("V_SIGN") == "next_app"
    # other os defaults still present
    assert cfg.action_for("SWIPE_LEFT") == "prev_app"


def test_invalid_backend_falls_back_to_os(tmp_path):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"backend": "bogus"}))
    cfg = load_config(str(p))
    assert cfg.backend == "os"
