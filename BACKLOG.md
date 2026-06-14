# hand-tab-controller backlog

The quality/improvement loop works through this list **top to bottom**: each cycle
it picks the **first unchecked item**, implements it on a worktree branch, opens a
PR, and checks the item off **in that same PR**. One item per PR. Keep changes
camera-free-testable (verify via pytest; no live camera/osascript in CI).

If every item is checked, make no changes and say so.

## Items

## Done

- [x] **Live gesture-mapping reload.** Pressing `r` in the OpenCV window reloads
  `gestures.json` without restarting via a testable `main.reload_config(recognizer,
  router) -> (config, help_lines)` helper that calls `load_config()`, pushes fresh
  thresholds onto the recognizer through a new `GestureRecognizer.apply_thresholds`
  (resetting per-hand `_trackers`/`_pinch_wins` when `smoothing_window` changes),
  swaps the router's config via a new `_BaseRouter.set_config` (refreshes mappings +
  cooldown, clears `_last_fire`, keeps `log`), and returns rebuilt `help_lines`. The
  `run()` loop rebinds `config`/`help_lines`/`render` on `r` and the controls hint
  now mentions `r reload`. Unit-tested in `tests/test_gestures.py` (apply_thresholds
  updates values, missing keys keep current, window change clears trackers),
  `tests/test_workspace.py` (set_config swaps mapping/cooldown, keeps log), and
  `tests/test_main.py` (reload_config with fake and real recognizer+router via a
  monkeypatched `load_config`). 9 tests added; all 163 pass.

- [x] **Expand test coverage of glue code.** Added `tests/test_main.py` (covers
  `main._help_lines` header/backend + one-line-per-mapping, `main._make_render`
  canvas-vs-os path selection via monkeypatched `ui` drawing fns, `build_workspace`,
  and `config.resolve_camera_index` override/fallback cases) and `tests/test_ui.py`
  (extracted a pure `ui.wrap_lines(lines, max_chars)` word-wrap helper — short
  passthrough, word-boundary wrap, hard-break of overlong tokens, empty-line/empty-
  input handling, wrap-disabled — plus `overlay_text_panel` drawing tests guarded by
  cv2/numpy availability). `overlay_text_panel` gained an optional `max_chars`
  parameter (default 0 = unchanged behavior) that wires in the new helper. 21 tests
  added; all 119 pass.

- [x] **Config validation with clear errors.** Added a pure
  `validate_config(data, backend) -> list[str]` to `config.py` that returns
  human-readable problems for unknown gesture names (vs the `GestureType` enum),
  unknown action names (vs the active backend's default mapping, exposed as
  `VALID_ACTIONS_OS`/`VALID_ACTIONS_CANVAS`), out-of-range / wrong-type / unknown
  thresholds (module-level `THRESHOLD_RANGES`), invalid backend, and invalid
  camera_index. `load_config` now logs each problem as a `WARNING` via
  `logging.getLogger("config")` and stays non-fatal — bad entries fall back to
  defaults exactly as before. `config.py` remains free of cv2/mediapipe. Unit-tested
  in `tests/test_config.py` (clean config → `[]`, each error category, and a
  `caplog` test that `load_config` warns yet returns a usable `Config`).

- [x] **Multi-display awareness.** Added pure helpers `display_for_point` (containment with
  nearest-center fallback for off-screen windows; `None` on empty list), `clamp_rect_to_frame`
  (fit/shrink a window rect into a display frame), `parse_display_frames`, and a thin
  `build_list_displays_script` builder. The controller's new `_active_display_frame` resolves the
  display under the front window and `_tile`/`toggle_split` now tile within that display's visible
  frame, falling back safely to the primary display when enumeration fails or returns <2 displays
  (single-display results unchanged). The menu bar is assumed to apply to every display's top.
- [x] **CI: run pytest on every PR.** Added `.github/workflows/ci.yml` (GitHub Actions on
  Ubuntu / Python 3.11) that installs system OpenGL libs + `requirements.txt` + pytest and
  runs the suite on every PR and push to main.
- [x] **Menu-bar / Dock-aware tiling.** Added pure `visible_frame(screen_w, screen_h,
  menubar_h)` helper; `tile_left`/`tile_right`/`toggle_split` now place panes within the
  visible frame (below the 25px menu bar) instead of full-screen-height under it.
