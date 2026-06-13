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

from config import load_config, resolve_camera_index
from gestures import GestureRecognizer, HandLandmarks
from actions import ActionRouter, OSActionRouter
from os_control import OSWindowController
from workspace import Workspace
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
    """Return a function ``render(frame, points) -> canvas`` for the backend."""
    if config.backend == "canvas":
        def render(frame, points):
            canvas = ui.render_workspace(workspace, help_lines + router.log[-4:])
            ui.draw_landmarks(frame, points)
            return ui.overlay_camera(canvas, frame)
        return render

    # OS mode: show the camera feed full-frame with landmark dots and a text
    # panel describing detected gestures / fired window actions.
    def render(frame, points):
        ui.draw_landmarks(frame, points)
        return ui.overlay_text_panel(frame, help_lines + router.log[-5:])
    return render


def run(argv: list[str] | None = None) -> int:
    """Run the live pipeline. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Hand-gesture window controller")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Probe camera indices 0..5 and exit.")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use (overrides config).")
    parser.add_argument("--backend", choices=["os", "canvas"], default=None,
                        help="Override the config backend.")
    args = parser.parse_args(argv)

    if args.list_cameras:
        return list_cameras()

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

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(
            f"Could not open webcam (VideoCapture({cam_index})). Check camera "
            "permissions, or run with --list-cameras to find a working index.",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(MODEL_PATH):
        print(
            f"Missing hand landmark model: {MODEL_PATH}\n"
            "Download it with:\n    curl -sSL -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            file=sys.stderr,
        )
        cap.release()
        return 3

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(time.monotonic() * 1000))

            detected: list[HandLandmarks] = []
            points_for_overlay: list = []
            if result.hand_landmarks:
                handedness = result.handedness or []
                for i, hand_lms in enumerate(result.hand_landmarks):
                    label = "Right"
                    if i < len(handedness) and handedness[i]:
                        label = handedness[i][0].category_name
                    hl = HandLandmarks.from_task_landmarks(hand_lms, label=label)
                    detected.append(hl)
                    points_for_overlay.append(hl.points)

            events = recognizer.update(detected)
            router.handle_all(events)

            canvas = render(frame, points_for_overlay)
            if canvas is not None:
                cv2.imshow("Hand Gesture Controller", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
