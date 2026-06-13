# hand-tab-controller backlog

The quality/improvement loop works through this list **top to bottom**: each cycle
it picks the **first unchecked item**, implements it on a worktree branch, opens a
PR, and checks the item off **in that same PR**. One item per PR. Keep changes
camera-free-testable (verify via pytest; no live camera/osascript in CI).

If every item is checked, make no changes and say so.

## Items

- [ ] **Menu-bar / Dock-aware tiling.** `tile_left`/`tile_right`/`toggle_split` use the
  full Finder desktop bounds, so tiled windows sit under the menu bar. Use the
  *visible* frame instead. Add a pure geometry helper (e.g. `visible_frame(bounds,
  menubar_h, dock)`) with unit tests; keep the AppleScript execution thin.
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
