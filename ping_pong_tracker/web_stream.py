from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http import server
from pathlib import Path
from socketserver import ThreadingMixIn

import cv2
import numpy as np

from importlib.util import module_from_spec, spec_from_file_location

MODULE_PATH = Path(__file__).resolve().parent / "03_ball_detector.py"
SPEC = spec_from_file_location("ball_detector", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load detector module: {MODULE_PATH}")
DETECTOR = module_from_spec(SPEC)
SPEC.loader.exec_module(DETECTOR)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_integrated_system"))
from ball_tracker import BallTracker  # noqa: E402

# Radius of the circle drawn around (0,0): the area where tracking works.
WORK_RADIUS_CM = 10.0


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg = None
        self.mask_jpeg = None
        self.status = {
            "detected": False,
            "confidence": 0.0,
            "x_px": None,
            "y_px": None,
            "radius_px": None,
            "fps": 0.0,
            "roi": None,
            "updated_at": None,
        }
        self.running = True

    def update(self, jpeg: bytes, mask_jpeg: bytes, status: dict) -> None:
        with self.lock:
            self.jpeg = jpeg
            self.mask_jpeg = mask_jpeg
            self.status = status

    def snapshot(self) -> tuple[bytes | None, bytes | None, dict]:
        with self.lock:
            return self.jpeg, self.mask_jpeg, dict(self.status)


class DetectorWorker(threading.Thread):
    """Runs the same BallTracker the balance controller uses, so what the page shows is what main.py sees."""

    def __init__(self, state: SharedState, mirror: bool = True, detector: str = "old") -> None:
        super().__init__(daemon=True)
        self.state = state
        self.tracker = BallTracker(mirror=mirror, detector_mode=detector)
        self.pixel_per_cm = self.tracker.pixel_per_cm
        self.last_time = time.perf_counter()
        self.fps = 0.0

    def run(self) -> None:
        try:
            while self.state.running:
                ok, x_m, y_m, frame, mask, dt, det = self.tracker.read()
                if frame.size <= 1:
                    time.sleep(0.01)
                    continue
                instant = min(60.0, 1.0 / dt)
                self.fps = instant if self.fps == 0 else 0.1 * instant + 0.9 * self.fps

                output = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                height, width = output.shape[:2]
                # (0,0) is the frame centre, the same origin BallTracker._px_to_m uses.
                origin = (width // 2, height // 2)
                work_radius = int(round(WORK_RADIUS_CM * self.pixel_per_cm))
                cv2.circle(output, origin, work_radius, (0, 255, 255), 2)
                cv2.drawMarker(output, origin, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(output, f"{WORK_RADIUS_CM:g} cm", (origin[0] + 8, origin[1] - work_radius + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                detected = ok and det is not None
                confidence = 0.0
                x_px = y_px = radius_px = None
                if detected:
                    x_px, y_px, radius_px = int(round(det[0])), int(round(det[1])), int(round(det[2]))
                    confidence = float(det[3])
                    DETECTOR.draw_detection(output, x_px, y_px, radius_px)
                DETECTOR.draw_status(output, detected, x_px or 0, y_px or 0, self.fps, confidence)
                if detected:
                    cv2.rectangle(output, (0, 122), (370, 156), (0, 0, 0), -1)
                    cv2.putText(output, f"x={x_m * 100:+.1f} cm  y={y_m * 100:+.1f} cm", (10, 146),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                mask_view = cv2.resize(DETECTOR.draw_mask_preview(mask), (320, 200), interpolation=cv2.INTER_NEAREST)
                output[0:200, width - 320:width] = mask_view
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                mask_bgr = cv2.cvtColor(mask_view, cv2.COLOR_RGB2BGR)
                ok_jpeg, encoded = cv2.imencode(".jpg", output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                ok_mask, encoded_mask = cv2.imencode(".jpg", mask_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok_jpeg and ok_mask:
                    self.state.update(
                        encoded.tobytes(),
                        encoded_mask.tobytes(),
                        {
                            "detected": detected,
                            "confidence": round(confidence, 3),
                            "x_cm": round(x_m * 100, 2) if detected else None,
                            "y_cm": round(y_m * 100, 2) if detected else None,
                            "distance_from_origin_cm": round(float(np.hypot(x_m, y_m)) * 100, 2) if detected else None,
                            "x_px": x_px,
                            "y_px": y_px,
                            "radius_px": radius_px,
                            "fps": round(self.fps, 2),
                            "work_radius_cm": WORK_RADIUS_CM,
                            "updated_at": time.time(),
                        },
                    )
        finally:
            self.tracker.release()


class ThreadingHTTPServer(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True


class Handler(server.BaseHTTPRequestHandler):
    state: SharedState = None

    def do_GET(self) -> None:
        if self.path == "/":
            body = HTML.replace("__PORT__", str(self.server.server_port)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/status":
            _, _, status = self.state.snapshot()
            body = json.dumps(status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg, _, _ = self.state.snapshot()
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                return
        if self.path == "/mask.jpg":
            _, jpeg, _ = self.state.snapshot()
            if jpeg is None:
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return
        self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        return


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Ping Pong Tracker</title>
<style>body{font-family:Arial,sans-serif;background:#111;color:#eee;margin:20px}main{max-width:1300px;margin:auto}img{max-width:100%;border:2px solid #555}.grid{display:grid;grid-template-columns:1fr 320px;gap:16px}.panel{background:#222;padding:12px;border-radius:8px}pre{white-space:pre-wrap;color:#8fdaff}</style></head>
<body><main><h2>Ping Pong Tracker</h2><div class="grid"><div><img src="/stream.mjpg"><h3>Mask</h3><img src="/mask.jpg"></div><div class="panel"><h3>Status</h3><pre id="status">loading...</pre></div></div></main>
<script>async function update(){try{const r=await fetch('/status');document.getElementById('status').textContent=JSON.stringify(await r.json(),null,2)}catch(e){}}setInterval(update,500);update();</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-mirror", action="store_true", help="match main.py --no-mirror")
    parser.add_argument("--detector", choices=("old", "bgsub"), default="old")
    args = parser.parse_args()
    state = SharedState()
    worker = DetectorWorker(state, mirror=not args.no_mirror, detector=args.detector)
    worker.start()
    Handler.state = state
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Web GUI: http://<raspberry-pi-ip>:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        httpd.server_close()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
