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


def test_visible_frame_reserves_menu_bar():
    # Default menu bar height (25) is reserved off the top.
    assert osc.visible_frame(1440, 900) == (0, 25, 1440, 875)
    assert osc.visible_frame(1000, 800, menubar_h=40) == (0, 40, 1000, 760)


def test_visible_frame_clamps_menu_bar_to_screen():
    # A menu bar taller than the screen can't yield a negative height.
    assert osc.visible_frame(1000, 800, menubar_h=2000) == (0, 800, 1000, 0)
    assert osc.visible_frame(1000, 800, menubar_h=0) == (0, 0, 1000, 800)


def test_visible_frame_negative_menu_bar_floored_to_zero():
    assert osc.visible_frame(1000, 800, menubar_h=-10) == (0, 0, 1000, 800)


def test_tile_uses_visible_frame_below_menu_bar(monkeypatch):
    # _tile should position at the visible-frame y (menu bar) and size to vh,
    # never y=0 / full screen_h.
    calls = []
    ctrl = osc.OSWindowController(screen_w=1440, screen_h=900, menubar_h=25)

    def fake_run(script):
        calls.append(script)
        return True, ""

    monkeypatch.setattr(ctrl, "_run", fake_run)
    assert ctrl.tile_left() == "tile_left"
    # calls[0] is the display-enumeration query (empty -> single-display
    # fallback); the position script (menu-bar y 25) and size script (reduced
    # height 875) follow it.
    assert "25" in calls[1]
    assert "875" in calls[2]


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


# ----- multi-display pure helpers ------------------------------------------
def test_display_for_point_containment():
    primary = (0, 0, 1440, 900)
    secondary = (1440, 0, 1920, 1080)
    displays = [primary, secondary]
    # point inside primary
    assert osc.display_for_point(100, 100, displays) == primary
    # point inside secondary
    assert osc.display_for_point(1500, 200, displays) == secondary
    # boundary: top-left is inclusive, right/bottom exclusive
    assert osc.display_for_point(0, 0, displays) == primary
    assert osc.display_for_point(1440, 0, displays) == secondary


def test_display_for_point_nearest_fallback_when_offscreen():
    primary = (0, 0, 1000, 800)        # center (500, 400)
    secondary = (1000, 0, 1000, 800)   # center (1500, 400)
    displays = [primary, secondary]
    # off-screen far to the right -> nearest is secondary
    assert osc.display_for_point(5000, 400, displays) == secondary
    # off-screen far to the left (negative) -> nearest is primary
    assert osc.display_for_point(-500, 400, displays) == primary


def test_display_for_point_empty_returns_none():
    assert osc.display_for_point(10, 10, []) is None


def test_clamp_rect_to_frame_already_inside_unchanged():
    frame = (0, 25, 1000, 775)
    assert osc.clamp_rect_to_frame(100, 100, 400, 300, frame) == (100, 100, 400, 300)


def test_clamp_rect_to_frame_nudges_position_inside():
    frame = (0, 25, 1000, 775)
    # window pushed past right/bottom edges -> slid back in, size preserved
    assert osc.clamp_rect_to_frame(900, 700, 400, 300, frame) == (600, 500, 400, 300)
    # window above/left of frame -> snapped to frame origin
    assert osc.clamp_rect_to_frame(-50, 0, 400, 300, frame) == (0, 25, 400, 300)


def test_clamp_rect_to_frame_shrinks_when_larger_than_frame():
    frame = (100, 50, 800, 600)
    x, y, w, h = osc.clamp_rect_to_frame(0, 0, 5000, 5000, frame)
    assert (w, h) == (800, 600)        # shrunk to frame size
    assert (x, y) == (100, 50)         # and positioned at frame origin


def test_clamp_rect_to_frame_offset_origin():
    frame = (1440, 0, 1920, 1080)      # a secondary display
    assert osc.clamp_rect_to_frame(2000, 100, 500, 400, frame) == (2000, 100, 500, 400)
    # off the right edge of the secondary display -> slid in
    assert osc.clamp_rect_to_frame(3500, 100, 500, 400, frame) == (2860, 100, 500, 400)


def test_parse_display_frames_multiple():
    # two displays as flat l,t,r,b groups
    out = "0, 0, 1440, 900, 1440, 0, 3360, 1080"
    assert osc.parse_display_frames(out) == [
        (0, 0, 1440, 900),
        (1440, 0, 1920, 1080),
    ]


def test_parse_display_frames_braces_and_partial():
    # braces stripped; trailing partial group (<4) ignored
    out = "{0, 0, 1000, 800, 1000}"
    assert osc.parse_display_frames(out) == [(0, 0, 1000, 800)]


def test_parse_display_frames_empty():
    assert osc.parse_display_frames("") == []


# ----- multi-display controller wiring -------------------------------------
def test_active_display_frame_single_display_fallback():
    # enumeration returns one frame -> fall back to primary visible frame.
    ctrl = _stub_controller([(True, "0, 0, 1000, 800")])
    assert ctrl._active_display_frame() == (0, 25, 1000, 775)


def test_active_display_frame_enumeration_failure_fallback():
    ctrl = _stub_controller([(False, "denied")])
    assert ctrl._active_display_frame() == (0, 25, 1000, 775)


def test_active_display_frame_picks_display_under_front_window():
    # two displays; front window at x=1500 lives on the secondary display.
    # responses: enumerate displays, then front-window position query.
    ctrl = _stub_controller([
        (True, "0, 0, 1000, 800, 1000, 0, 3000, 800"),
        (True, "1500, 300"),
    ])
    # secondary frame (1000, 0, 2000, 800), menu bar reserved at its top.
    assert ctrl._active_display_frame() == (1000, 25, 2000, 775)


def test_tile_left_on_secondary_display():
    # enumerate (2 displays), front position (secondary), then pos + size.
    ctrl = _stub_controller([
        (True, "0, 0, 1000, 800, 1000, 0, 3000, 800"),
        (True, "1500, 300"),
        (True, ""),
        (True, ""),
    ])
    assert ctrl.tile_left() == "tile_left"
    # left half of the secondary display: x=1000, y=25 (below menu bar)
    assert "{1000, 25}" in ctrl.scripts[2]
    assert "{1000, 775}" in ctrl.scripts[3]  # half of 2000 wide, 775 tall


# ----- controller helpers with a stubbed _run -----------------------------
def _stub_controller(responses):
    """Build a controller whose ``_run`` returns queued ``(ok, out)`` tuples
    and records the scripts it was asked to run."""
    ctrl = osc.OSWindowController(screen_w=1000, screen_h=800)
    queue = list(responses)
    ctrl.scripts = []

    def fake_run(script):
        ctrl.scripts.append(script)
        return queue.pop(0) if queue else (True, "")

    ctrl._run = fake_run
    return ctrl


def test_tile_left_sets_position_and_size():
    # scripts[0] is the (empty) display-enumeration query -> single-display
    # fallback, so the position/size scripts follow it unchanged.
    ctrl = _stub_controller([(True, ""), (True, ""), (True, "")])
    assert ctrl.tile_left() == "tile_left"
    assert "{0, 25}" in ctrl.scripts[1]  # below the 25px menu bar
    assert "{500, 775}" in ctrl.scripts[2]  # left half, visible-frame height


def test_tile_right_uses_remaining_width():
    ctrl = _stub_controller([(True, ""), (True, ""), (True, "")])
    assert ctrl.tile_right() == "tile_right"
    assert "{500, 25}" in ctrl.scripts[1]
    assert "{500, 775}" in ctrl.scripts[2]


def test_tile_reports_failure_from_run():
    # enumeration returns nothing (fallback), then the position script fails.
    ctrl = _stub_controller([(True, ""), (False, "denied")])
    assert ctrl.tile_left() == "tile_left failed: denied"


def test_next_app_cycles_via_helper():
    # _refresh_cycle: list apps, then frontmost; then activate.
    ctrl = _stub_controller([(True, "A, B, C"), (True, "A"), (True, "")])
    assert ctrl.next_app() == "next_app -> B"


def test_prev_app_wraps_via_helper():
    ctrl = _stub_controller([(True, "A, B, C"), (True, "A"), (True, "")])
    assert ctrl.prev_app() == "prev_app -> C"


def test_next_app_no_apps():
    ctrl = _stub_controller([(False, "err"), (False, "err")])
    assert ctrl.next_app() == "no apps to cycle"


def test_query_coords_parses_and_handles_failure():
    ctrl = _stub_controller([(True, "10, 20")])
    assert ctrl._front_position() == (10, 20)
    ctrl = _stub_controller([(False, "no window")])
    assert ctrl._front_size() is None


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
