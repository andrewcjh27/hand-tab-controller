"""Tests for ui.py text-panel line wrapping and panel drawing.

``wrap_lines`` is a pure helper (no cv2/numpy) and is always tested. The
drawing path (``overlay_text_panel``) is only exercised when cv2 + numpy are
importable, otherwise those tests are skipped.
"""

import pytest

import importlib.util

import ui


# ----- wrap_lines: pure word-wrap logic ------------------------------------


def test_wrap_lines_short_line_passes_through():
    assert ui.wrap_lines(["hello world"], 40) == ["hello world"]


def test_wrap_lines_wraps_long_line_on_word_boundaries():
    out = ui.wrap_lines(["the quick brown fox jumps"], 10)
    # Every produced line stays within the limit.
    assert all(len(line) <= 10 for line in out)
    # No words are lost or split (they fit within the limit individually).
    assert " ".join(out).split() == "the quick brown fox jumps".split()


def test_wrap_lines_hard_breaks_overlong_word():
    out = ui.wrap_lines(["supercalifragilistic"], 5)
    assert all(len(line) <= 5 for line in out)
    assert "".join(out) == "supercalifragilistic"


def test_wrap_lines_hard_break_with_leading_word():
    out = ui.wrap_lines(["ab hugewordthatislong"], 4)
    assert all(len(line) <= 4 for line in out)
    # The short leading word survives and the long token is fully preserved.
    assert "ab" in out
    assert "".join(out).replace("ab", "", 1) == "hugewordthatislong"


def test_wrap_lines_preserves_empty_lines():
    assert ui.wrap_lines(["", "x"], 5) == ["", "x"]


def test_wrap_lines_empty_input():
    assert ui.wrap_lines([], 5) == []


def test_wrap_lines_nonpositive_max_disables_wrapping():
    lines = ["a very long line that would otherwise wrap"]
    assert ui.wrap_lines(lines, 0) == lines
    assert ui.wrap_lines(lines, -1) == lines


# ----- overlay_text_panel: drawing path (needs cv2 + numpy) ----------------

_HAS_CV2 = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("numpy") is not None
)
cv2_required = pytest.mark.skipif(
    not _HAS_CV2, reason="cv2/numpy not available for image drawing"
)


@cv2_required
def test_overlay_text_panel_returns_frame_with_lines():
    import numpy as _np

    frame = _np.zeros((120, 200, 3), dtype=_np.uint8)
    out = ui.overlay_text_panel(frame, ["line one", "line two"])
    assert out is not None
    assert out.shape == frame.shape


@cv2_required
def test_overlay_text_panel_empty_lines_returns_mirrored_frame():
    import numpy as _np

    frame = _np.zeros((60, 80, 3), dtype=_np.uint8)
    out = ui.overlay_text_panel(frame, [])
    assert out is not None
    assert out.shape == frame.shape


@cv2_required
def test_overlay_text_panel_wraps_when_max_chars_set():
    import numpy as _np

    frame = _np.zeros((300, 200, 3), dtype=_np.uint8)
    # Should not raise even with a very long line forced to wrap.
    out = ui.overlay_text_panel(frame, ["x" * 200], max_chars=10)
    assert out is not None
    assert out.shape == frame.shape


def test_overlay_text_panel_none_frame_returns_none():
    # No frame and/or no cv2 -> graceful passthrough of the input.
    assert ui.overlay_text_panel(None, ["a"]) is None


# ----- overlay_minimal / overlay_prompt: minimal HUD ------------------------


@cv2_required
def test_overlay_minimal_preserves_frame_shape():
    import numpy as _np

    frame = _np.zeros((240, 320, 3), dtype=_np.uint8)
    out = ui.overlay_minimal(frame, status="Swipe left", action="prev_app")
    assert out is not None and out.shape == frame.shape


@cv2_required
def test_overlay_minimal_help_panel_and_no_status():
    import numpy as _np

    frame = _np.zeros((240, 320, 3), dtype=_np.uint8)
    # show_help draws the mapping list; status=None should still render fine.
    out = ui.overlay_minimal(frame, status=None, show_help=True,
                             help_lines=["A -> b", "C -> d"], max_chars=20)
    assert out is not None and out.shape == frame.shape


def test_overlay_minimal_none_frame_returns_none():
    assert ui.overlay_minimal(None, status="x") is None


@cv2_required
def test_overlay_prompt_preserves_frame_shape():
    import numpy as _np

    frame = _np.zeros((240, 320, 3), dtype=_np.uint8)
    out = ui.overlay_prompt(frame, "Make a FIST and hold", "4s")
    assert out is not None and out.shape == frame.shape


def test_overlay_prompt_none_frame_returns_none():
    assert ui.overlay_prompt(None, "title") is None


@cv2_required
def test_overlay_training_banner_and_complete():
    import numpy as _np

    canvas = _np.zeros((300, 420, 3), dtype=_np.uint8)
    out = ui.overlay_training(canvas, "Swipe right", "Move right", "next tab",
                              reps_done=2, reps_total=5, step_index=1, step_total=9)
    assert out is not None and out.shape == canvas.shape
    done = ui.overlay_training(canvas, "", "", "", 0, 0, 0, 0, complete=True)
    assert done is not None and done.shape == canvas.shape


def test_overlay_training_none_canvas_returns_none():
    assert ui.overlay_training(None, "x", "y", "z", 0, 1, 1, 1) is None
