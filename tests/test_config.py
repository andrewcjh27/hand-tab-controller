"""Tests for config validation and load_config warning behavior."""

import json
import logging

import pytest

from config import (
    DEFAULT_MAPPINGS,
    DEFAULT_OS_MAPPINGS,
    DEFAULT_THRESHOLDS,
    VALID_ACTIONS_CANVAS,
    VALID_ACTIONS_OS,
    VALID_GESTURES,
    Config,
    load_config,
    validate_config,
)
from gestures import GestureType


def _clean_os_data():
    return {
        "backend": "os",
        "camera_index": 0,
        "mappings": dict(DEFAULT_OS_MAPPINGS),
        "thresholds": dict(DEFAULT_THRESHOLDS),
    }


# ----- validate_config: clean configs --------------------------------------


def test_validate_clean_os_config_returns_empty():
    assert validate_config(_clean_os_data(), "os") == []


def test_validate_clean_canvas_config_returns_empty():
    data = {
        "backend": "canvas",
        "camera_index": 0,
        "mappings": dict(DEFAULT_MAPPINGS),
        "thresholds": dict(DEFAULT_THRESHOLDS),
    }
    assert validate_config(data, "canvas") == []


def test_real_gestures_json_is_clean():
    """The shipped gestures.json must not produce warnings."""
    import config as config_mod

    with open(config_mod.DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    backend = str(data.get("backend", "os")).lower()
    assert validate_config(data, backend) == []


def test_valid_gesture_set_matches_enum():
    assert VALID_GESTURES == frozenset(g.value for g in GestureType)


def test_valid_action_sets_derive_from_default_maps():
    assert VALID_ACTIONS_OS == frozenset(DEFAULT_OS_MAPPINGS.values())
    assert VALID_ACTIONS_CANVAS == frozenset(DEFAULT_MAPPINGS.values())


# ----- validate_config: each error category --------------------------------


def test_unknown_gesture_name():
    data = _clean_os_data()
    data["mappings"]["SWIPE_UP"] = "next_app"
    problems = validate_config(data, "os")
    assert any("unknown gesture" in p and "SWIPE_UP" in p for p in problems)


def test_unknown_action_name():
    data = _clean_os_data()
    data["mappings"]["SWIPE_LEFT"] = "do_a_barrel_roll"
    problems = validate_config(data, "os")
    assert any(
        "unknown action" in p and "do_a_barrel_roll" in p for p in problems
    )


def test_action_valid_for_one_backend_invalid_for_other():
    # 'prev_app' is an OS action; under canvas backend it is unknown.
    data = {"backend": "canvas", "mappings": {"SWIPE_LEFT": "prev_app"}}
    problems = validate_config(data, "canvas")
    assert any("unknown action" in p and "prev_app" in p for p in problems)
    # ...but valid under the os backend.
    assert validate_config({"mappings": {"SWIPE_LEFT": "prev_app"}}, "os") == []


def test_threshold_out_of_range_high():
    data = _clean_os_data()
    data["thresholds"]["swipe_velocity"] = 2.0
    problems = validate_config(data, "os")
    assert any(
        "swipe_velocity" in p and "out of range" in p for p in problems
    )


def test_threshold_out_of_range_low_exclusive():
    data = _clean_os_data()
    data["thresholds"]["resize_step"] = 0.0  # (0, 1] excludes 0
    problems = validate_config(data, "os")
    assert any("resize_step" in p and "out of range" in p for p in problems)


def test_threshold_wrong_type():
    data = _clean_os_data()
    data["thresholds"]["pinch_threshold"] = "high"
    problems = validate_config(data, "os")
    assert any(
        "pinch_threshold" in p and "not a number" in p for p in problems
    )


def test_threshold_int_required():
    data = _clean_os_data()
    data["thresholds"]["smoothing_window"] = 2.5
    problems = validate_config(data, "os")
    assert any(
        "smoothing_window" in p and "integer" in p for p in problems
    )


def test_smoothing_window_below_min():
    data = _clean_os_data()
    data["thresholds"]["smoothing_window"] = 0
    problems = validate_config(data, "os")
    assert any(
        "smoothing_window" in p and "out of range" in p for p in problems
    )


def test_cooldown_ms_zero_is_valid():
    data = _clean_os_data()
    data["thresholds"]["cooldown_ms"] = 0
    assert validate_config(data, "os") == []


def test_unknown_threshold_key():
    data = _clean_os_data()
    data["thresholds"]["wiggle_factor"] = 0.5
    problems = validate_config(data, "os")
    assert any("unknown threshold" in p and "wiggle_factor" in p for p in problems)


def test_invalid_backend():
    data = {"backend": "rocket", "mappings": {}, "thresholds": {}}
    problems = validate_config(data, "os")
    assert any("invalid backend" in p and "rocket" in p for p in problems)


def test_invalid_camera_index_negative():
    data = _clean_os_data()
    data["camera_index"] = -3
    problems = validate_config(data, "os")
    assert any("camera_index" in p and "negative" in p for p in problems)


def test_invalid_camera_index_non_int():
    data = _clean_os_data()
    data["camera_index"] = "front"
    problems = validate_config(data, "os")
    assert any("camera_index" in p and "not an integer" in p for p in problems)


def test_bool_is_not_a_valid_number():
    data = _clean_os_data()
    data["thresholds"]["swipe_velocity"] = True
    problems = validate_config(data, "os")
    assert any("swipe_velocity" in p and "not a number" in p for p in problems)


def test_multiple_problems_collected():
    data = {
        "backend": "rocket",
        "camera_index": -1,
        "mappings": {"BOGUS": "nope"},
        "thresholds": {"swipe_velocity": 99},
    }
    problems = validate_config(data, "os")
    assert len(problems) >= 4


# ----- load_config: non-fatal warn + usable Config -------------------------


def test_load_config_warns_but_returns_usable_config(tmp_path, caplog):
    bad = {
        "backend": "os",
        "camera_index": 0,
        "mappings": {"SWIPE_UP": "warp_speed"},
        "thresholds": {"swipe_velocity": 5.0, "mystery": 1},
    }
    p = tmp_path / "gestures.json"
    p.write_text(json.dumps(bad))

    with caplog.at_level(logging.WARNING, logger="config"):
        cfg = load_config(str(p))

    # Non-fatal: a usable Config is returned with valid defaults intact.
    assert isinstance(cfg, Config)
    assert cfg.backend == "os"
    assert cfg.action_for("SWIPE_LEFT") == "prev_app"

    # Warnings were emitted for the problems.
    messages = [r.getMessage() for r in caplog.records]
    text = " ".join(messages)
    assert "SWIPE_UP" in text
    assert "warp_speed" in text
    assert "swipe_velocity" in text
    assert "mystery" in text


def test_load_config_clean_emits_no_warnings(tmp_path, caplog):
    p = tmp_path / "gestures.json"
    p.write_text(json.dumps(_clean_os_data()))

    with caplog.at_level(logging.WARNING, logger="config"):
        cfg = load_config(str(p))

    assert isinstance(cfg, Config)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_load_config_bad_camera_index_falls_back(tmp_path, caplog):
    bad = {"backend": "os", "camera_index": -5}
    p = tmp_path / "gestures.json"
    p.write_text(json.dumps(bad))
    with caplog.at_level(logging.WARNING, logger="config"):
        cfg = load_config(str(p))
    assert cfg.camera_index == 0  # fell back to default
