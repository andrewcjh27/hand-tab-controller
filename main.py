"""Live entry point for the hand-gesture controller.

Opens the webcam, runs MediaPipe Hands (Tasks API HandLandmarker, VIDEO mode),
converts landmarks to gesture events, and routes them either to **real macOS
windows** (``backend: "os"``) via :mod:`os_control`, or to the in-app demo
:class:`workspace.Workspace` (``backend: "canvas"``). Rendering uses OpenCV.

Heavy dependencies (cv2, mediapipe) are import-guarded with a clear install
hint so running without them fails gracefully instead of crashing on import.

Run:
    python main.py                 # uses gestures.json backend (default "os")
    python main.py --list-cameras  # probe camera indices 0..5 and exit
    python main.py --camera 0      # force a specific camera index

OS mode drives windows via ``osascript`` and needs Accessibility permission
(System Settings -> Privacy & Security -> Accessibility -> enable your Terminal,
then quit & reopen it). Camera permission is also required. Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from config import load_config, resolve_camera_index, save_thresholds
from gestures import GestureRecognizer, HandLandmarks, VelocityTracker
from actions import ActionRouter, OSActionRouter
from os_control import OSWindowController
from workspace import Workspace
import calibration
import ui

_MISSING = []
try:
    import cv2
except Exception:
    cv2 = None
    _MISSING.append("opencv-python")
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp = None
    mp_python = None
    mp_vision = None
    _MISSING.append("mediapipe")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def _help_lines(config) -> list[str]:
    lines = [f"Backend: {config.backend}  (q to quit)"]
    for gesture, action in config.mappings.items():
        lines.append(f"  {gesture:<16} -> {action}")
    return lines


def build_workspace() -> Workspace:
    ws = Workspace()
    for title in ("Editor", "Browser", "Terminal", "Docs"):
        ws.add_tab(title)
    ws.set_active(0)
    return ws


def list_cameras(max_index: int = 5) -> int:
    """Probe camera indices 0..max_index and print which open + frame size."""
    if cv2 is None:
        print("opencv-python is required for --list-cameras.", file=sys.stderr)
        return 1
    print("Probing camera indices 0..%d ..." % max_index)
    print(
        "Note: on macOS index 0 is usually the built-in FaceTime HD Camera, but\n"
        "an iPhone Continuity Camera can claim a slot. Pick the built-in one and\n"
        "set it via `--camera N` or `camera_index` in gestures.json.\n"
    )
    found = 0
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print(f"  [{idx}] OPEN   frame={w}x{h}")
                found += 1
            else:
                print(f"  [{idx}] opened but no frame")
        else:
            print(f"  [{idx}] (none)")
        cap.release()
    if found == 0:
        print("\nNo cameras opened. Check camera permission for your terminal.")
    return 0


def _make_render(config, workspace, router, help_lines):
    """Return ``render(frame, points, status, action, show_help) -> canvas``.

    ``status`` is the current detected gesture (or a tracking hint), ``action``
    is the most recent action that fired, and ``show_help`` toggles the full
    mapping list. The OS backend uses the minimal HUD; canvas keeps the demo.
    """
    if config.backend == "canvas":
        def render(frame, points, status, action, show_help):
            log = (help_lines if show_help else []) + router.log[-4:]
            canvas = ui.render_workspace(workspace, log)
            ui.draw_landmarks(frame, points)
            return ui.overlay_camera(canvas, frame)
        return render

    # OS mode: minimal HUD over the mirrored camera feed.
    def render(frame, points, status, action, show_help):
        ui.draw_landmarks(frame, points)
        return ui.overlay_minimal(
            frame, status=status, action=action, show_help=show_help,
            help_lines=help_lines, max_chars=42,
        )
    return render


# Human-friendly labels for the status readout.
_GESTURE_LABELS = {
    "SWIPE_LEFT": "Swipe left", "SWIPE_RIGHT": "Swipe right",
    "PINCH": "Pinch", "SPREAD": "Spread", "POINT": "Point",
    "GRAB": "Grab", "OPEN_PALM": "Open palm",
    "TWO_HAND_PINCH": "Two-hand", "V_SIGN": "Peace sign",
}


def _status_text(events, hands_present: bool) -> str:
    """Pick the status-bar label from this frame's events / tracking state."""
    if events:
        return _GESTURE_LABELS.get(events[-1].type.value, events[-1].type.value)
    if hands_present:
        return "Tracking..."
    return "No hand"


def _landmarker_options():
    """Build HandLandmarker options (VIDEO mode). Requires mediapipe present."""
    return mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )


def _detect_hands(result) -> tuple[list, list]:
    """Convert a HandLandmarker result into (HandLandmarks list, overlay points)."""
    detected: list[HandLandmarks] = []
    points: list = []
    if result.hand_landmarks:
        handedness = result.handedness or []
        for i, hand_lms in enumerate(result.hand_landmarks):
            label = "Right"
            if i < len(handedness) and handedness[i]:
                label = handedness[i][0].category_name
            hl = HandLandmarks.from_task_landmarks(hand_lms, label=label)
            detected.append(hl)
            points.append(hl.points)
    return detected, points


def _preflight(cam_index: int):
    """Open the camera + verify the model file. Returns (cap, error_code).

    On success returns ``(cap, 0)``; on failure ``(None, code)`` with a message
    already printed to stderr.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(
            f"Could not open webcam (VideoCapture({cam_index})). Check camera "
            "permissions, or run with --list-cameras to find a working index.",
            file=sys.stderr,
        )
        return None, 2
    if not os.path.exists(MODEL_PATH):
        print(
            f"Missing hand landmark model: {MODEL_PATH}\n"
            "Download it with:\n    curl -sSL -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            file=sys.stderr,
        )
        cap.release()
        return None, 3
    return cap, 0


def _capture(cap, landmarker, title: str, duration: float, on_frame) -> bool:
    """Show ``title`` + a countdown for ``duration`` seconds, calling
    ``on_frame(hands, dt)`` each frame. Returns False if the user pressed q.
    """
    start = time.monotonic()
    prev = start
    while True:
        ok, frame = cap.read()
        if not ok:
            return True
        now = time.monotonic()
        remaining = max(0.0, duration - (now - start))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                       data=rgb), int(now * 1000))
        detected, points = _detect_hands(result)
        on_frame(detected, now - prev)
        prev = now
        ui.draw_landmarks(frame, points)
        view = ui.overlay_prompt(frame, title, f"{remaining:0.0f}s  (q to abort)")
        if view is not None:
            cv2.imshow("Hand Gesture Controller - Calibration", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        if remaining <= 0:
            return True


def _collect_swipe_peaks(window: int):
    """Return (on_frame, peaks) collecting peak |horizontal velocity| per swipe."""
    tracker = VelocityTracker(window=window)
    peaks: list[float] = []
    state = {"in_burst": False, "peak": 0.0}
    floor = 0.02

    def on_frame(hands, dt):
        if not hands:
            return
        tracker.update(hands[0].centroid)
        vel = abs(tracker.horizontal_velocity())
        if vel > floor:
            state["in_burst"] = True
            state["peak"] = max(state["peak"], vel)
        elif state["in_burst"]:
            peaks.append(state["peak"])
            state["in_burst"], state["peak"] = False, 0.0

    return on_frame, peaks


def _collect_pinch_deltas():
    """Return (on_frame, deltas) collecting abs frame-to-frame pinch changes."""
    deltas: list[float] = []
    prev = {"d": None}

    def on_frame(hands, dt):
        if not hands:
            return
        d = hands[0].pinch()
        if prev["d"] is not None:
            deltas.append(abs(d - prev["d"]))
        prev["d"] = d

    return on_frame, deltas


def _collect_pinch_distances():
    """Return (on_frame, dists) collecting raw pinch distances while a hand shows."""
    dists: list[float] = []

    def on_frame(hands, dt):
        if hands:
            dists.append(hands[0].pinch())

    return on_frame, dists


def calibrate(cli_camera: int | None = None) -> int:
    """Guided per-gesture calibration; writes tuned thresholds to gestures.json."""
    if _MISSING:
        print("Missing required packages: " + ", ".join(_MISSING)
              + "\nInstall them with:\n    pip install -r requirements.txt",
              file=sys.stderr)
        return 1
    config = load_config()
    cam_index = resolve_camera_index(config.camera_index, cli_camera)
    cap, err = _preflight(cam_index)
    if cap is None:
        return err

    print("Calibration: follow the on-screen prompts. Press q to abort.")
    measured: dict[str, float | None] = {}
    window = int(config.threshold("smoothing_window"))
    aborted = False
    with mp_vision.HandLandmarker.create_from_options(_landmarker_options()) as lm:
        # During the swipe step, also record the pinch jitter so the zoom
        # threshold can be floored above it (swipe -> false zoom fix).
        swipe_cb, peaks = _collect_swipe_peaks(window)
        noise_cb, swipe_noise = _collect_pinch_deltas()

        def swipe_frame(hands, dt):
            swipe_cb(hands, dt)
            noise_cb(hands, dt)

        if not _capture(cap, lm, "Swipe LEFT and RIGHT a few times", 6.0, swipe_frame):
            aborted = True
        if not aborted:
            measured["swipe_velocity"] = calibration.recommend_swipe_velocity(peaks)

            pinch_cb, deltas = _collect_pinch_deltas()
            if not _capture(cap, lm, "Hold your hand STILL; pinch and spread", 6.0, pinch_cb):
                aborted = True
        if not aborted:
            measured["pinch_sensitivity"] = calibration.recommend_pinch_sensitivity(
                deltas, noise_deltas=swipe_noise
            )

            fist_cb, closed = _collect_pinch_distances()
            if not _capture(cap, lm, "Make a FIST and hold", 4.0, fist_cb):
                aborted = True
        if not aborted:
            open_cb, opened = _collect_pinch_distances()
            if not _capture(cap, lm, "OPEN your hand and hold", 4.0, open_cb):
                aborted = True
        if not aborted:
            measured["pinch_threshold"] = calibration.recommend_pinch_threshold(closed, opened)

    cap.release()
    cv2.destroyAllWindows()
    if aborted:
        print("Calibration aborted; gestures.json unchanged.")
        return 0

    new_thresholds = calibration.merge_thresholds(config.thresholds, measured)
    path = save_thresholds(new_thresholds)
    print("Calibration complete. Saved to", path)
    for key in ("swipe_velocity", "pinch_sensitivity", "pinch_threshold"):
        got = measured.get(key)
        note = f"{got}" if got is not None else "(unchanged — not measured)"
        print(f"  {key:<18} {note}")
    return 0


def run(argv: list[str] | None = None) -> int:
    """Run the live pipeline. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Hand-gesture window controller")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Probe camera indices 0..5 and exit.")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use (overrides config).")
    parser.add_argument("--backend", choices=["os", "canvas"], default=None,
                        help="Override the config backend.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run guided gesture calibration and save tuned "
                             "thresholds to gestures.json, then exit.")
    args = parser.parse_args(argv)

    if args.list_cameras:
        return list_cameras()

    if args.calibrate:
        return calibrate(args.camera)

    if _MISSING:
        print(
            "Missing required packages: "
            + ", ".join(_MISSING)
            + "\nInstall them with:\n    pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    config = load_config()
    if args.backend:
        config.backend = args.backend
    cam_index = resolve_camera_index(config.camera_index, args.camera)

    # Build the backend + router.
    if config.backend == "os":
        controller = OSWindowController()
        msg = controller.refresh_screen_size()
        router = OSActionRouter(controller, config)
        workspace = None
        print(f"OS window control mode. {msg}")
        print(
            "Requires Accessibility permission (System Settings -> Privacy & "
            "Security -> Accessibility -> enable your Terminal, then reopen it)."
        )
    else:
        workspace = build_workspace()
        router = ActionRouter(workspace, config)

    recognizer = GestureRecognizer(
        {
            "swipe_velocity": config.threshold("swipe_velocity"),
            "pinch_sensitivity": config.threshold("pinch_sensitivity"),
            "pinch_threshold": config.threshold("pinch_threshold"),
            "smoothing_window": int(config.threshold("smoothing_window")),
        }
    )
    help_lines = _help_lines(config)
    render = _make_render(config, workspace, router, help_lines)

    cap, err = _preflight(cam_index)
    if cap is None:
        return err

    print("Controls:  q quit   h toggle help")
    show_help = False
    with mp_vision.HandLandmarker.create_from_options(_landmarker_options()) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(time.monotonic() * 1000))

            detected, points_for_overlay = _detect_hands(result)
            events = recognizer.update(detected)
            router.handle_all(events)

            status = _status_text(events, bool(detected))
            action = router.log[-1] if router.log else None
            canvas = render(frame, points_for_overlay, status, action, show_help)
            if canvas is not None:
                cv2.imshow("Hand Gesture Controller", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
