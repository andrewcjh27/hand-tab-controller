"""Tests for main.py glue helpers: _help_lines, _make_render, camera index.

These cover the thin "wiring" layer in main.py without needing a real camera,
display, or the heavy cv2/mediapipe stack. The render path is exercised by
monkeypatching the ``ui`` functions so we can assert *which* backend path runs.
"""

import argparse
import types

import pytest

import main
from config import Config, resolve_camera_index


# ----- _help_lines ----------------------------------------------------------


def test_help_lines_header_mentions_backend():
    config = Config(backend="canvas")
    lines = main._help_lines(config)
    assert lines[0].startswith("Backend: canvas")
    assert "q to quit" in lines[0]


def test_help_lines_has_one_line_per_mapping():
    config = Config(mappings={"SWIPE_LEFT": "prev_tab", "SWIPE_RIGHT": "next_tab"},
                    backend="canvas")
    lines = main._help_lines(config)
    # header + one line per mapping
    assert len(lines) == 1 + len(config.mappings)
    body = "\n".join(lines[1:])
    assert "SWIPE_LEFT" in body and "prev_tab" in body
    assert "SWIPE_RIGHT" in body and "next_tab" in body


def test_help_lines_reflects_os_backend_actions():
    config = Config(mappings={"SWIPE_LEFT": "prev_app"}, backend="os")
    lines = main._help_lines(config)
    assert lines[0].startswith("Backend: os")
    assert any("prev_app" in line for line in lines[1:])


# ----- _make_render selection ----------------------------------------------


class _DummyRouter:
    def __init__(self):
        self.log = ["LOG1", "LOG2"]


def _patch_ui(monkeypatch):
    """Replace the ui drawing fns with recorders; return a calls dict."""
    calls = {"render_workspace": 0, "overlay_camera": 0,
             "overlay_minimal": 0, "draw_landmarks": 0}

    def render_workspace(workspace, lines):
        calls["render_workspace"] += 1
        return "canvas"

    def overlay_camera(canvas, frame):
        calls["overlay_camera"] += 1
        return "canvas+cam"

    def overlay_minimal(frame, status=None, action=None, hint="h: help",
                        show_help=False, help_lines=None, max_chars=0):
        calls["overlay_minimal"] += 1
        return "minimal"

    def draw_landmarks(frame, points):
        calls["draw_landmarks"] += 1
        return frame

    monkeypatch.setattr(main.ui, "render_workspace", render_workspace)
    monkeypatch.setattr(main.ui, "overlay_camera", overlay_camera)
    monkeypatch.setattr(main.ui, "overlay_minimal", overlay_minimal)
    monkeypatch.setattr(main.ui, "draw_landmarks", draw_landmarks)
    return calls


def test_make_render_canvas_path(monkeypatch):
    calls = _patch_ui(monkeypatch)
    config = Config(backend="canvas")
    router = _DummyRouter()
    render = main._make_render(config, workspace=object(), router=router,
                               help_lines=["H"])
    out = render("frame", [], "Tracking...", "next_app", False)
    # Canvas path renders the workspace + overlays the camera thumbnail.
    assert out == "canvas+cam"
    assert calls["render_workspace"] == 1
    assert calls["overlay_camera"] == 1
    assert calls["overlay_minimal"] == 0


def test_make_render_os_path(monkeypatch):
    calls = _patch_ui(monkeypatch)
    config = Config(backend="os")
    router = _DummyRouter()
    render = main._make_render(config, workspace=None, router=router,
                               help_lines=["H"])
    out = render("frame", [], "Swipe left", "prev_app", False)
    # OS path draws landmarks on the frame + the minimal HUD; no workspace canvas.
    assert out == "minimal"
    assert calls["overlay_minimal"] == 1
    assert calls["render_workspace"] == 0
    assert calls["overlay_camera"] == 0
    assert calls["draw_landmarks"] == 1


# ----- camera-index resolution ---------------------------------------------


def test_resolve_camera_index_cli_override_wins():
    assert resolve_camera_index(0, 3) == 3


def test_resolve_camera_index_none_falls_back_to_config():
    assert resolve_camera_index(2, None) == 2


def test_resolve_camera_index_negative_cli_falls_back():
    assert resolve_camera_index(1, -1) == 1


def test_resolve_camera_index_zero_cli_is_valid_override():
    # 0 is a real camera index and must win over a non-zero config value.
    assert resolve_camera_index(4, 0) == 0


# ----- build_workspace (small glue helper) ---------------------------------


def test_build_workspace_creates_four_active_tabs():
    ws = main.build_workspace()
    assert len(ws.tabs) == 4
    assert [t.title for t in ws.tabs] == ["Editor", "Browser", "Terminal", "Docs"]
    assert ws.tabs[0].active is True


# ----- _status_text ---------------------------------------------------------


def test_status_text_uses_friendly_gesture_label():
    from gestures import Gesture, GestureType
    events = [Gesture(GestureType.SWIPE_LEFT)]
    assert main._status_text(events, hands_present=True) == "Swipe left"


def test_status_text_tracking_and_no_hand():
    assert main._status_text([], hands_present=True) == "Tracking..."
    assert main._status_text([], hands_present=False) == "No hand"
