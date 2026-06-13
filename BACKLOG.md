# hand-tab-controller backlog

The quality/improvement loop works through this list **top to bottom**: each cycle
it picks the **first unchecked item**, implements it on a worktree branch, opens a
PR, and checks the item off **in that same PR**. One item per PR. Keep changes
camera-free-testable (verify via pytest; no live camera/osascript in CI).

If every item is checked, make no changes and say so.

## Items

- [ ] **Multi-display awareness.** Detect which display the frontmost window is on and
  tile/clamp within that display's visible frame rather than always the primary.
  Add pure helpers + tests (no hardware).
- [ ] **Config validation with clear errors.** Validate `gestures.json` on load:
  unknown gesture names, unknown action names, out-of-range thresholds → raise/log a
  clear message instead of silently misbehaving. Unit-test the validator.
- [ ] **Expand test coverage of glue code.** Add tests for `main.py` helpers
  (`_help_lines`, `_make_render` selection, camera-index resolution) and `ui.py`
  text-panel line wrapping — the currently-thin-on-coverage modules.
- [ ] **Live gesture-mapping reload.** A keypress in the OpenCV window (e.g. `r`)
  reloads `gestures.json` without restarting, so remapping is instant. Factor the
  reload path to be unit-testable.

## Done

- [x] **CI: run pytest on every PR.** Added `.github/workflows/ci.yml` (GitHub Actions on
  Ubuntu / Python 3.11) that installs system OpenGL libs + `requirements.txt` + pytest and
  runs the suite on every PR and push to main.
- [x] **Menu-bar / Dock-aware tiling.** Added pure `visible_frame(screen_w, screen_h,
  menubar_h)` helper; `tile_left`/`tile_right`/`toggle_split` now place panes within the
  visible frame (below the 25px menu bar) instead of full-screen-height under it.
