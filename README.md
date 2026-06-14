# Hand Gesture Controller

A webcam-based hand-motion gesture controller. Wave, pinch, point, and grab to
**switch apps, move/resize windows, and tile them side by side** — controlling
**real macOS windows** (Notes, Terminal, VS Code, Figma, any app). An optional
in-app "canvas" demo backend is kept for testing without permissions.

It uses [MediaPipe Hands](https://developers.google.com/mediapipe) (the new
**Tasks API** `HandLandmarker`, VIDEO mode) for hand tracking and OpenCV for the
camera feed. Window control is driven by `osascript` (AppleScript + System
Events) — no extra Python dependencies. The gesture math, config, and routing
are pure, dependency-light modules so they can be unit-tested without a camera.

## Requirements & install (Apple Silicon)

MediaPipe must run under a **native arm64** Python. The bundled venv is arm64
Python 3.11. The system `python3` may be x86_64/Rosetta and will crash on
`import mediapipe` (no AVX). Always use the venv:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

### Tasks API model file

The app needs the `hand_landmarker.task` model in the project root. If missing,
download it:

```bash
curl -sSL -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Permissions (macOS)

OS window-control mode needs **both**:

1. **Camera** — System Settings → Privacy & Security → Camera → enable your
   terminal/IDE.
2. **Accessibility** — System Settings → Privacy & Security → **Accessibility**
   → enable **Terminal** (or your IDE). **Quit and reopen** the terminal after
   granting, or the permission won't take effect.

`osascript` drives other apps' windows via System Events, which is gated behind
Accessibility. Without it, window actions fail gracefully (an error string is
logged on screen) instead of crashing.

## Choosing a camera

On macOS, the built-in **FaceTime HD Camera** is usually index `0`, but an
**iPhone Continuity Camera** can claim a slot. List what's available:

```bash
.venv/bin/python main.py --list-cameras
```

This probes indices 0–5 and prints which open and their frame size. Then pick
one either on the command line or in `gestures.json`:

```bash
.venv/bin/python main.py --camera 0          # force built-in webcam
```

```json
{ "camera_index": 0 }
```

A valid `--camera` flag overrides `camera_index` in the config.

## Run

```bash
.venv/bin/python main.py                # OS window control (default backend)
.venv/bin/python main.py --backend canvas   # in-app demo, no Accessibility needed
```

The OpenCV window shows the mirrored camera feed with subtle hand-landmark dots
and a **minimal HUD**: a thin bottom bar with the current detected gesture on the
left and the most recent action on the right. It stays out of the way.

Controls:

- **`q`** — quit
- **`h`** — toggle the full gesture→action mapping list (hidden by default)

## Calibrate to your hand (recommended)

Default sensitivity is intentionally low so gestures don't fire on incidental
motion. For the best fit, calibrate to your own movements:

```bash
.venv/bin/python main.py --calibrate
```

It walks you through a few prompts — swipe left/right, pinch & spread, make a
fist, open your hand — measures your actual motion, and writes tuned
`swipe_velocity`, `pinch_sensitivity`, and `pinch_threshold` values into
`gestures.json` (other settings are preserved). Press `q` during calibration to
abort without saving. Re-run anytime it feels too sensitive or not responsive
enough.

## Gestures → window actions (OS mode)

| Gesture            | Type             | Action          | What it does                                         |
|--------------------|------------------|-----------------|------------------------------------------------------|
| Open-hand swipe ←  | `SWIPE_LEFT`     | `prev_app`      | Bring the previous visible app to the front          |
| Open-hand swipe →  | `SWIPE_RIGHT`    | `next_app`      | Bring the next visible app to the front              |
| Pinch in           | `PINCH`          | `resize_shrink` | Shrink the front window (×0.9, clamped)              |
| Pinch out (spread) | `SPREAD`         | `resize_grow`   | Grow the front window (×1.1, clamped to screen)      |
| Point (index)      | `POINT`          | `begin_move`    | Enter move mode for the front window                 |
| Grab / fist        | `GRAB`           | `drag_move`     | Drag the front window to follow the hand             |
| Open palm          | `OPEN_PALM`      | `release`       | Release / commit the move                            |
| Two-hand pinch     | `TWO_HAND_PINCH` | `resize_two_hand` | Resize the front window by the distance between hands |
| V / peace sign     | `V_SIGN`         | `toggle_split`  | Tile the front window left, previous app right       |

App cycling covers visible, non-background application processes. The hand
centroid (normalized) is mapped to absolute screen coordinates (mirrored in x)
so the window follows your hand during a drag.

### Swipe vs. zoom

Only one gesture fires per frame, decided by how the whole hand moves:

- **Swipe** — *move your whole hand* across the frame (open palm). A translating
  hand is always read as a swipe, never a zoom.
- **Zoom (pinch/spread)** — *hold your hand roughly still* and change the
  thumb-index distance. A stationary hand with moving fingers is a zoom.

Motion that's neither clearly moving nor clearly still does nothing, on purpose.
If swipes and zooms still get confused, run `--calibrate` — it now measures the
finger jitter during your swipes and sets the zoom threshold just above it.

### Canvas (demo) mode

`--backend canvas` (or `"backend": "canvas"` in `gestures.json`) keeps the
original in-app tab workspace. It needs no Accessibility permission and is handy
for testing the pipeline. In that mode the same gestures map to `next_tab`,
`prev_tab`, `begin_move`/`drag_move`/`release`, `resize_*`, and
`toggle_double_view`.

## Customizing (`gestures.json`)

```json
{
  "backend": "os",
  "camera_index": 0,
  "mappings": { "V_SIGN": "toggle_split" },
  "thresholds": { "cooldown_ms": 600 }
}
```

- **`backend`** — `"os"` (real windows, default) or `"canvas"` (in-app demo).
- **`camera_index`** — webcam index (see `--list-cameras`).
- **`mappings`** — gesture type → action name. Overlaid on the active backend's
  defaults, so you can override individual gestures.
  - OS actions: `next_app`, `prev_app`, `resize_grow`, `resize_shrink`,
    `resize_two_hand`, `begin_move`, `drag_move`, `release`, `toggle_split`.
  - Canvas actions: `next_tab`, `prev_tab`, `resize_grow`, `resize_shrink`,
    `resize_two_hand`, `begin_move`, `drag_move`, `release`, `toggle_double_view`.
- **`thresholds`** — tunable numbers:

| Field               | Meaning                                                          |
|---------------------|------------------------------------------------------------------|
| `swipe_velocity`    | Min horizontal palm velocity (norm. units/frame) for a swipe     |
| `pinch_sensitivity` | Min per-frame change in thumb-index distance for PINCH/SPREAD    |
| `pinch_threshold`   | Thumb-index distance below which a hand counts as pinched (GRAB) |
| `smoothing_window`  | Frames used for velocity estimation                              |
| `cooldown_ms`       | Debounce between repeated firings of a discrete action           |
| `move_speed`        | Pixels moved per drag step (canvas mode)                         |
| `resize_step`       | Fractional size change per PINCH/SPREAD                          |

Regenerate the default config any time with:

```bash
.venv/bin/python config.py
```

## Tests

Pure-logic tests (gesture math, workspace, routing, AppleScript builders, camera
index resolution, app-cycle math) run without a camera, MediaPipe, or osascript:

```bash
.venv/bin/python -m pytest -q
```

## Troubleshooting

- **Window actions do nothing / "not authorized"**: grant Accessibility to your
  terminal and **reopen it**.
- **Camera won't open**: grant Camera permission; run `--list-cameras` to find a
  working index; close other apps using the camera.
- **`mediapipe` crashes on import**: you're likely on x86_64/Rosetta Python. Use
  the arm64 venv (`.venv/bin/python`).
- **`Missing required packages`**: `.venv/bin/pip install -r requirements.txt`.
