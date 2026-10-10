"""Annotated camera video of a control run (for demonstrations), written off the control-loop thread."""

from __future__ import annotations

import queue
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

BALL = (214, 120, 42)        # BGR of #2a78d6
REF = (52, 104, 235)         # BGR of #eb6834
TRAIL_FRAMES = 90


class VideoRecorder:
    def __init__(self, path: Path, pixel_per_cm: float, fps: float = 30.0, width: int = 1280) -> None:
        self.path = path
        self.fps = fps
        self.width = width
        self.writer: cv2.VideoWriter | None = None
        self.queue: queue.Queue = queue.Queue(maxsize=90)
        self.dropped = 0
        self.written = 0
        self.overlay = Overlay(pixel_per_cm)
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def add(self, frame: np.ndarray, state: dict) -> None:
        """Queue a frame; dropped (and counted) rather than ever blocking the control loop."""
        try:
            self.queue.put_nowait((frame.copy(), state))
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=30)
        if self.writer is not None:
            self.writer.release()
        print(f"[Video] {self.path} ({self.written} frames, {self.dropped} dropped)")

    def _work(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            frame, state = item
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self.overlay.draw(frame, state)
            if frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, round(frame.shape[0] * self.width / frame.shape[1])),
                                   interpolation=cv2.INTER_AREA)
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.writer.write(frame)
            self.written += 1


class Overlay:
    """The annotation drawn on every recorded frame: reference path, ball trail and a two-line header.

    Kept apart from the recorder so the identical overlay can be drawn on the frames a run saved to
    disk (logs/frames_<stamp>/) when that run has no recorded video.
    """

    def __init__(self, pixel_per_cm: float) -> None:
        self.ppc = pixel_per_cm
        self.ball_trail: deque = deque(maxlen=TRAIL_FRAMES)
        self.ref_trail: deque = deque(maxlen=TRAIL_FRAMES * 20)

    def _to_px(self, shape: tuple, x_m: float, y_m: float) -> tuple[int, int]:
        h, w = shape[:2]
        return int(round(w / 2 + x_m * 100 * self.ppc)), int(round(h / 2 - y_m * 100 * self.ppc))

    def track(self, shape: tuple, s: dict) -> None:
        """Add one control step's ball and reference to the trails (shape = the frame the trails are drawn on)."""
        if s["tracking"]:
            self.ref_trail.append(self._to_px(shape, s["x_ref"], s["y_ref"]))
        det = s["det_px"]
        if det is not None:
            self.ball_trail.append((int(det[0]), int(det[1])))

    def draw(self, frame: np.ndarray, s: dict) -> None:
        """Add this frame's ball and reference to the trails, then draw everything onto the frame (in place)."""
        self.track(frame.shape, s)
        self.render(frame, s)

    def render(self, frame: np.ndarray, s: dict) -> None:
        """Draw the trails collected so far plus this step's markers and header, without adding to the trails."""
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
        if s["tracking"] and len(self.ref_trail) > 1:
            cv2.polylines(frame, [np.array(self.ref_trail, np.int32)], False, REF, 2, cv2.LINE_AA)
        cv2.drawMarker(frame, self._to_px(frame.shape, s["x_ref"], s["y_ref"]), REF, cv2.MARKER_CROSS, 28, 3)

        det = s["det_px"]
        if len(self.ball_trail) > 1:
            cv2.polylines(frame, [np.array(self.ball_trail, np.int32)], False, BALL, 3, cv2.LINE_AA)
        if det is not None:
            cv2.circle(frame, (int(det[0]), int(det[1])), int(det[2]), BALL, 3, cv2.LINE_AA)

        err_cm = float(np.hypot(s["x_m"] - s["x_ref"], s["y_m"] - s["y_ref"]) * 100)
        lines = [
            f"t = {s['t']:5.1f} s   {'PATH TRACKING' if s['tracking'] else 'CENTERING'}"
            f"   ball {'tracked' if det is not None else 'NOT SEEN'}",
            f"ball ({s['x_m'] * 100:+5.1f}, {s['y_m'] * 100:+5.1f}) cm   error {err_cm:4.1f} cm"
            f"   tilt cmd ({s['tx_deg']:+4.1f}, {s['ty_deg']:+4.1f}) deg",
        ]
        cv2.rectangle(frame, (0, 0), (w, 12 + 34 * len(lines)), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (14, 36 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
