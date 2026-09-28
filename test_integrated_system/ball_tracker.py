from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.append("/usr/lib/python3/dist-packages")
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


# Detection is done on a downscaled copy: 4x less pixel work, and this is the width
# detection_config.json was originally tuned at.
PROC_WIDTH = int(os.environ.get("BALL_PROC_WIDTH", "640"))
# Frames averaged into the initial background. The platform must be EMPTY during these.
BG_INIT_FRAMES = int(os.environ.get("BALL_BG_INIT_FRAMES", "20"))
# Per-frame blend rate for the slow background update (pixels near the ball are skipped).
BG_UPDATE_ALPHA = float(os.environ.get("BALL_BG_ALPHA", "0.02"))
# Long exposure smears a moving ball; capping it and paying with analogue gain instead
# keeps the image just as bright (measured) while cutting motion blur proportionally.
MAX_EXPOSURE_US = int(os.environ.get("CAMERA_MAX_EXPOSURE_US", "8000"))
# Flat-field gain map from calibrate_flat_field.py: evens out vignetting and room lighting so the
# brightness thresholds work at the plate edge too. Missing file or BALL_FLAT_FIELD=0 -> no correction.
FLAT_FIELD_PATH = Path(__file__).resolve().parents[1] / "ping_pong_tracker" / "flat_field.npz"
USE_FLAT_FIELD = os.environ.get("BALL_FLAT_FIELD", "1") != "0"
# The ball is white and lit, so its brightness is well above the frame median at every spot
# on the platform (measured: >= 1.47x), while structures that move in the image when the
# plate tilts -- the black connector seen through the plate, servos, wires -- are dark
# (measured: <= 0.76x). Peaks on anything dimmer than this fraction are not the ball.
MIN_BALL_BRIGHTNESS_RATIO = float(os.environ.get("BALL_MIN_BRIGHTNESS_RATIO", "1.15"))
# A changed region bigger than this many ball-areas is a hand/arm, not the ball.
# A bright lamp sits just outside the plate at about (-14.3, +2.1) cm in the image. The tracker locked onto it
# whenever the ball was not visible (12 of the 30 trials of 24 Sep were contaminated by it). Nothing that matters
# is beyond ~11.5 cm from the centre (the ball is lost past ~10 cm anyway), so candidates outside this radius
# are ignored.
MAX_PLATE_RADIUS_CM = float(os.environ.get("BALL_MAX_PLATE_RADIUS_CM", "11.5"))
MAX_BLOB_AREA_FACTOR = float(os.environ.get("BALL_MAX_BLOB_AREA_FACTOR", "4.0"))


def _frame_quality_score(frame: np.ndarray) -> float:
    # Heuristic: richer textures and edges usually indicate a real scene vs static virtual feed.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 60, 120)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
    return std + 0.02 * lap_var + 100.0 * edge_ratio


def _open_with_backends(index: int, width: int, height: int) -> tuple[Optional[cv2.VideoCapture], str]:
    backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None, "NONE"

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, frame = cap.read()
    if ok and frame is not None and frame.size > 0:
        return cap, "DSHOW"

    cap.release()
    return None, "NONE"


def probe_cameras(max_index: int = 8) -> list[dict]:
    found = []
    for idx in range(max_index):
        cap, backend_name = _open_with_backends(idx, 1280, 720)
        if cap is None:
            continue
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            cap.release()
            continue
        h, w = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        score = _frame_quality_score(frame)
        found.append(
            {
                "index": idx,
                "width": int(w),
                "height": int(h),
                "fps": fps,
                "score": score,
                "backend": backend_name,
            }
        )
        cap.release()
    return found


class BallTracker:
    def __init__(
        self,
        camera_index: int = -1,
        mirror: bool = True,
        width: int = 1280,
        height: int = 800,
        hsv_config_path: Optional[Path] = None,
        calib_config_path: Optional[Path] = None,
        detector_mode: str = "old",
        save_frames_every: int = 0,
    ) -> None:
        if detector_mode not in ("old", "bgsub"):
            raise ValueError("detector_mode must be 'old' or 'bgsub'")
        self.detector_mode = detector_mode
        self.debug_every = int(save_frames_every)
        self.debug_dir: Optional[Path] = None
        self.debug_count = 0
        root = Path(__file__).resolve().parents[1]
        default_hsv = root / "ping_pong_tracker" / "hsv_config.json"
        default_calib = root / "ping_pong_tracker" / "calib_config.json"

        default_detection = root / "ping_pong_tracker" / "detection_config.json"
        self.detection_config_path = Path(hsv_config_path) if hsv_config_path else default_detection
        self.calib_config_path = Path(calib_config_path) if calib_config_path else default_calib

        self.detection_cfg = self._load_json(self.detection_config_path, {"gray_min": 160, "gray_max": 255, "blur_kernel": 7, "morph_kernel": 5})
        # Detection runs on a PROC_WIDTH-wide copy of the frame, which is also the width
        # detection_config.json was tuned at, so its radius/kernel values are used as-is.
        # Results are scaled back to full-frame pixels before leaving this class.
        self.proc_width = PROC_WIDTH
        self.proc_scale = self.proc_width / float(width)
        self.proc_height = int(round(height * self.proc_scale))

        def _odd(v: float) -> int:
            v = int(round(v))
            return v if v % 2 == 1 else v + 1

        self.hough_min_radius = int(self.detection_cfg.get("hough_min_radius", 32))
        self.hough_max_radius = int(self.detection_cfg.get("hough_max_radius", 48))
        self.expected_radius = float(self.detection_cfg.get("expected_radius", 40))
        self.blur_kernel = _odd(self.detection_cfg.get("blur_kernel", 7))
        self.morph_kernel = _odd(self.detection_cfg.get("morph_kernel", 5))
        self.hough_blur_kernel = _odd(9)
        self.min_confidence = float(self.detection_cfg.get("min_confidence", 0.45))
        # The contour fallback used to accept blobs down to 10 px, and the loop of servo cable in the
        # middle of the plate (r ~15 px) passed as a ball -- the tracker then stayed on it and ignored
        # the real ball (r ~40 px) at the edge. Anything well under the ball's size is rejected.
        self.min_contour_radius = float(self.detection_cfg.get("min_contour_radius", 0.6 * self.expected_radius))

        # Background-difference detector: works from "how different is this from the
        # remembered background", so it finds the ball whether it is brighter or darker
        # than its surroundings -- unlike an absolute-brightness threshold.
        self.bg_model: Optional[np.ndarray] = None
        self.bg_reference: Optional[np.ndarray] = None
        self.bg_init_frames = 0
        self.bg_suppress_until = 0.0
        self.bg_kernel = _odd(self.expected_radius)
        print(
            f"[BallTracker] detector={self.detector_mode}; detection at {self.proc_width}x{self.proc_height} "
            f"(frame {width}x{height}); expected_radius={self.expected_radius:.0f}px, "
            f"bg_kernel={self.bg_kernel}, hough band {self.hough_min_radius}-{self.hough_max_radius}"
        )
        if self.debug_every > 0:
            self.debug_dir = Path(__file__).resolve().parent / "logs" / f"frames_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"[BallTracker] saving every {self.debug_every}th frame (frame | |frame-bg| | background) to {self.debug_dir}")
        self.calib_cfg = self._load_json(self.calib_config_path, {})
        if "pixel_per_cm" not in self.calib_cfg:
            raise RuntimeError(f"Missing calibration: {self.calib_config_path}")
        self.pixel_per_cm = float(self.calib_cfg["pixel_per_cm"])
        if self.pixel_per_cm <= 0:
            raise ValueError("pixel_per_cm must be positive")

        self.width = width
        self.height = height
        self.mirror = mirror
        self.flat_gain: Optional[np.ndarray] = None
        self.flat_field_info: Optional[dict] = None
        if USE_FLAT_FIELD and FLAT_FIELD_PATH.exists():
            data = np.load(FLAT_FIELD_PATH)
            gain = data["gain"].astype(np.float32)
            info = json.loads(str(data["info"]))
            if gain.shape != (self.proc_height, self.proc_width):
                print(f"[BallTracker] flat field ignored: size {gain.shape[::-1]} != detection size {(self.proc_width, self.proc_height)}")
            else:
                if bool(info.get("mirror", True)) != bool(mirror):
                    gain = cv2.flip(gain, 1)
                self.flat_gain = gain
                self.flat_field_info = info
                print(f"[BallTracker] flat-field correction ON ({FLAT_FIELD_PATH.name}, made {info.get('created_utc')}, "
                      f"gain {info.get('gain_range')})")
        elif USE_FLAT_FIELD:
            print("[BallTracker] flat-field correction OFF (no calibration file; run calibrate_flat_field.py)")
        else:
            print("[BallTracker] flat-field correction OFF (BALL_FLAT_FIELD=0)")
        self.camera_index = self._resolve_camera_index(camera_index)
        self.cap = self._open_camera(self.camera_index)
        self.last_detection: Optional[Tuple[float, float, float, float]] = None
        self.last_detection_time = None
        self.detection_velocity = np.zeros(2, dtype=float)
        self.pending_detection: Optional[Tuple[float, float, float, float]] = None
        self.pending_frames = 0
        self.missed_frames = 0

        self.last_frame_time = time.perf_counter()

        self.last_frame_age_s = float('nan')

    @staticmethod
    def _resolve_camera_index(camera_index: int) -> int:
        return camera_index if camera_index >= 0 else 0

    def _open_camera(self, index: int):
        if Picamera2 is not None:
            camera = Picamera2()
            config = camera.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": 30},
            )
            camera.configure(config)
            camera.start()
            time.sleep(0.5)
            print("[Camera] Opened using Picamera2/libcamera")
            self._lock_ae_awb(camera)
            return camera
        cap, backend_name = _open_with_backends(index, self.width, self.height)
        if cap is None:
            raise RuntimeError(f"Cannot open camera index {index}")
        print(f"[Camera] Opened index={index} using backend={backend_name}")
        return cap

    @staticmethod
    def _lock_ae_awb(camera) -> None:
        try:
            for _ in range(20):
                camera.capture_metadata()
                time.sleep(0.06)
            meta = camera.capture_metadata()
            exposure = int(meta["ExposureTime"])
            gain = float(meta["AnalogueGain"])

            if exposure > MAX_EXPOSURE_US:
                try:
                    gain_max = float(camera.camera_controls["AnalogueGain"][1])
                except Exception:
                    gain_max = 16.0
                gain = min(gain * exposure / MAX_EXPOSURE_US, gain_max)
                exposure = MAX_EXPOSURE_US

            controls = {"AeEnable": False, "ExposureTime": exposure, "AnalogueGain": gain}
            colour_gains = meta.get("ColourGains")
            if colour_gains is not None:
                # Absent on monochrome sensors (e.g. OV9281 mono) -- nothing to white-balance.
                controls["AwbEnable"] = False
                controls["ColourGains"] = colour_gains
            camera.set_controls(controls)
            print(
                f"[Camera] Locked exposure={exposure}us gain={gain:.2f} "
                f"(auto-converged {meta['ExposureTime']}us @ {meta['AnalogueGain']:.2f}; "
                f"capped at {MAX_EXPOSURE_US}us to limit motion blur)"
            )
        except Exception as exc:
            print(f"[Camera] Could not lock AE/AWB, leaving auto-exposure enabled: {exc}")

    def switch_camera(self, new_index: int) -> None:
        old_cap = self.cap
        self.cap = self._open_camera(new_index)
        self.camera_index = new_index
        self.bg_model = None
        self.bg_reference = None
        self.bg_init_frames = 0
        self.last_frame_time = time.perf_counter()
        if Picamera2 is not None and isinstance(old_cap, Picamera2):
            old_cap.stop()
            old_cap.close()
        else:
            old_cap.release()
        print(f"[Camera] Switched to index={new_index}")

    @staticmethod
    def _load_json(path: Path, default: dict) -> dict:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _accumulate_background(self, proc_gray: np.ndarray) -> None:
        """Build the initial background from the first frames; platform must be empty."""
        sample = proc_gray.astype(np.float32)
        if self.bg_model is None:
            print(f"[BallTracker] Capturing background from {BG_INIT_FRAMES} frames -- keep the ball OFF the platform")
            self.bg_model = sample.copy()
        else:
            self.bg_model += (sample - self.bg_model) / float(self.bg_init_frames + 1)
        self.bg_init_frames += 1
        if self.bg_init_frames == BG_INIT_FRAMES:
            # Kept untouched: the empty platform at the pose it was captured in (neutral).
            self.bg_reference = self.bg_model.copy()
            print(f"[BallTracker] Background captured from {BG_INIT_FRAMES} frames; place the ball now")

    def _update_background(self, proc_gray: np.ndarray, freeze_at: list[Tuple[float, float]]) -> None:
        """Track slow scene drift, but never absorb the ball itself into the background."""
        blend = BG_UPDATE_ALPHA * (proc_gray.astype(np.float32) - self.bg_model)
        for x, y in freeze_at:
            cv2.circle(blend, (int(x), int(y)), int(self.expected_radius * 1.8), 0.0, -1)
        self.bg_model += blend

    def _save_debug(self, proc_gray: np.ndarray, det_proc: Optional[Tuple[float, float, float, float]]) -> None:
        """Save what the detector saw: frame with its pick | |frame - background| | background."""
        left = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2BGR)
        if det_proc is not None:
            cv2.circle(left, (int(det_proc[0]), int(det_proc[1])), int(self.expected_radius), (0, 255, 255), 2)
            cv2.putText(left, f"conf {det_proc[3]:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(left, "no candidate", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self.bg_model is not None:
            bg_u8 = cv2.convertScaleAbs(self.bg_model)
            diff = cv2.absdiff(proc_gray, bg_u8)
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            middle = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            right = cv2.cvtColor(bg_u8, cv2.COLOR_GRAY2BGR)
        else:
            middle = np.zeros_like(left)
            right = np.zeros_like(left)
        name = f"{self.debug_count:06d}.jpg"
        cv2.imwrite(str(self.debug_dir / name), np.hstack([left, middle, right]), [cv2.IMWRITE_JPEG_QUALITY, 80])

    def reset_background(self, settle_s: float = 0.4) -> None:
        """Return to the empty-platform reference after the platform is put back to neutral.

        Call this when the controller neutralizes the platform: the adapted background
        belongs to whatever tilted pose the platform had while tracking, but the reference
        was captured at neutral, which is where the platform is headed now. Detection
        pauses for settle_s so the platform's own motion is not mistaken for the ball.
        """
        if self.bg_reference is None:
            return
        self.bg_model = self.bg_reference.copy()
        self.bg_suppress_until = time.perf_counter() + settle_s
        self.last_detection = None
        self.last_detection_time = None
        self.detection_velocity[:] = 0.0
        self.pending_detection = None
        self.pending_frames = 0

    def _detect_ball_bgsub(
        self,
        proc_gray: np.ndarray,
        search: Optional[tuple[tuple[float, float], float]] = None,
    ) -> tuple[Optional[Tuple[float, float, float, float]], np.ndarray]:
        """Find the ball as the ball-sized region that differs most from the background.

        Uses the absolute difference, so it works whether the ball reads brighter or
        darker than whatever it happens to be sitting on. While a ball is being tracked,
        `search` ((x, y), radius) restricts the peak search to around its predicted spot,
        so background artifacts elsewhere (e.g. from the platform tilting) are ignored.
        """
        bg_u8 = cv2.convertScaleAbs(self.bg_model)
        diff = cv2.absdiff(proc_gray, bg_u8)
        diff = self._apply_roi(diff)
        # Blurring with a ball-sized kernel is a matched filter: ball-sized blobs survive,
        # thin edges and speckle (e.g. background shifting as the platform tilts) do not.
        response = cv2.GaussianBlur(diff, (self.bg_kernel, self.bg_kernel), 0)
        search_response = response
        # Absolute-brightness gate: a background difference only counts if the spot itself
        # is bright. Without this the detector locks onto dark hardware that shifts in the
        # image as the plate tilts, since |frame - background| is large for it too.
        brightness = cv2.GaussianBlur(proc_gray, (self.bg_kernel, self.bg_kernel), 0)
        bright_enough = brightness >= MIN_BALL_BRIGHTNESS_RATIO * float(np.median(proc_gray))
        search_response = np.where(bright_enough, search_response, 0).astype(response.dtype)
        if search is not None:
            (sx, sy), radius = search
            window = np.zeros_like(response)
            cv2.circle(window, (int(sx), int(sy)), max(1, int(radius)), 255, -1)
            search_response = cv2.bitwise_and(response, window)
        _, peak, _, loc = cv2.minMaxLoc(search_response)
        floor = float(np.percentile(response, 95))
        snr = float(peak) - floor
        confidence = float(np.clip((snr - 8.0) / 45.0, 0.0, 1.0))

        mask = cv2.inRange(diff, max(8, int(peak * 0.5)), 255)
        if confidence < self.min_confidence:
            return None, mask

        # A hand or arm reaching in differs from the background far more widely than a
        # ball does; reject a changed region much larger than the ball.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blob_area = int(stats[labels[loc[1], loc[0]], cv2.CC_STAT_AREA]) if labels[loc[1], loc[0]] > 0 else 0
        if blob_area > MAX_BLOB_AREA_FACTOR * np.pi * self.expected_radius ** 2:
            return None, mask

        # The matched-filter peak is the center estimate. Centroid refinement was measured
        # to be worse here: the ball's shadow drags the centroid off the ball.
        return (float(loc[0]), float(loc[1]), self.expected_radius, confidence), mask

    def _get_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        mask = cv2.inRange(blur, int(self.detection_cfg.get("gray_min", 160)), int(self.detection_cfg.get("gray_max", 255)))
        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = self._apply_roi(mask)
        return mask

    def _roi_bounds(self, shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
        roi = self.detection_cfg.get("roi", {})
        if not roi.get("enabled", False):
            return None
        height, width = shape[:2]
        x = max(0, min(width - 1, int(float(roi.get("x", 0.0)) * width)))
        y = max(0, min(height - 1, int(float(roi.get("y", 0.0)) * height)))
        right = max(x + 1, min(width, int(float(roi.get("x", 0.0)) * width + float(roi.get("width", 1.0)) * width)))
        bottom = max(y + 1, min(height, int(float(roi.get("y", 0.0)) * height + float(roi.get("height", 1.0)) * height)))
        return x, y, right, bottom

    def _apply_roi(self, image: np.ndarray) -> np.ndarray:
        bounds = self._roi_bounds(image.shape)
        if bounds is None:
            return image
        x, y, right, bottom = bounds
        result = np.zeros_like(image)
        result[y:bottom, x:right] = image[y:bottom, x:right]
        return result

    def _inside_plate(self, cx: float, cy: float, frame_shape) -> bool:
        """True if a candidate centre (in the coordinates of the frame being searched) is within the plate area."""
        height, width = frame_shape[:2]
        limit_px = MAX_PLATE_RADIUS_CM * self.pixel_per_cm * (width / float(self.width))
        return math.hypot(cx - width / 2.0, cy - height / 2.0) <= limit_px

    def _detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        roi_bounds = self._roi_bounds(gray.shape)
        offset_x = offset_y = 0
        if roi_bounds is not None:
            left, top, right, bottom = roi_bounds
            gray = gray[top:bottom, left:right]
            offset_x, offset_y = left, top
        blurred = cv2.GaussianBlur(gray, (self.hough_blur_kernel, self.hough_blur_kernel), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=80,
            param2=22,
            minRadius=self.hough_min_radius,
            maxRadius=self.hough_max_radius,
        )
        if circles is not None:
            height, width = gray.shape
            yy, xx = np.ogrid[:height, :width]
            bright_mask = cv2.threshold(blurred, 145, 255, cv2.THRESH_BINARY)[1]
            candidates = []
            for cx, cy, radius in np.round(circles[0]).astype(int):
                distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                inner = gray[distance <= radius * 0.65]
                ring = gray[(distance >= radius * 1.15) & (distance <= radius * 1.55)]
                if inner.size == 0 or ring.size == 0:
                    continue
                contrast = float(np.mean(inner) - np.mean(ring))
                inner_fill = float(np.mean(bright_mask[distance <= radius * 0.65]) / 255.0)
                ring_fill = float(np.mean(bright_mask[(distance >= radius * 1.15) & (distance <= radius * 1.55)]) / 255.0)
                if contrast <= 8 or inner_fill < 0.45 or inner_fill - ring_fill < 0.12:
                    continue
                fill_score = min(1.0, max(0.0, (inner_fill - ring_fill) / 0.7))
                confidence = min(1.0, max(0.0, (contrast - 8) / 80)) * (0.5 + 0.5 * max(0.0, 1.0 - abs(radius - self.expected_radius) / self.expected_radius)) * (0.5 + 0.5 * fill_score)
                if not self._inside_plate(cx + offset_x, cy + offset_y, frame.shape):
                    continue
                candidates.append((float(cx + offset_x), float(cy + offset_y), float(radius), confidence))
            if candidates:
                # Nearest-to-last only among circles that would pass min_confidence: otherwise a weak
                # circle on the servo cable next to the ball (conf ~0.07) won over the ball itself
                # (conf ~0.95) and the frame was then thrown away as a miss.
                strong = [c for c in candidates if c[3] >= self.min_confidence]
                if self.last_detection is not None and strong:
                    candidates = strong
                    candidates.sort(key=lambda item: np.hypot(item[0] - self.last_detection[0], item[1] - self.last_detection[1]))
                else:
                    candidates.sort(key=lambda item: -item[3])
                return candidates[0]

        mask = self._get_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > 35000:
                continue
            per = cv2.arcLength(cnt, True)
            if per <= 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (per * per)
            if circularity < 0.6:
                continue
            _, _, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            if not (0.65 <= aspect_ratio <= 1.5):
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1e-5)
            if solidity < 0.75:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.min_contour_radius or radius > 90:
                continue
            confidence = min(1.0, circularity * solidity)
            if not self._inside_plate(cx, cy, frame.shape):
                continue
            candidates.append((float(cx), float(cy), float(radius), confidence, area))
        if not candidates:
            return None

        if self.last_detection is not None:
            previous_x, previous_y = self.last_detection[:2]
            candidates.sort(key=lambda item: np.hypot(item[0] - previous_x, item[1] - previous_y))
        else:
            frame_center = (gray.shape[1] / 2.0, gray.shape[0] / 2.0)
            candidates.sort(key=lambda item: np.hypot(item[0] - frame_center[0], item[1] - frame_center[1]))
        cx, cy, radius, confidence, _ = candidates[0]
        return cx, cy, radius, confidence

    @staticmethod
    def _frame_age(metadata: dict) -> float:
        """Seconds between the sensor timestamp of a frame and now (nan if the timestamp is missing)."""
        stamp = metadata.get("SensorTimestamp")
        if stamp is None:
            return float("nan")
        now_ns = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        age = (now_ns - int(stamp)) / 1e9
        return age if 0.0 <= age < 5.0 else float("nan")   # a clock mismatch would show as an absurd age

    def _px_to_m(self, px: float, py: float, frame_w: int, frame_h: int) -> Tuple[float, float]:
        x_cm = (px - frame_w / 2.0) / self.pixel_per_cm
        y_cm = (frame_h / 2.0 - py) / self.pixel_per_cm
        return x_cm / 100.0, y_cm / 100.0

    def read(self) -> Tuple[bool, Optional[float], Optional[float], np.ndarray, np.ndarray, float, Optional[Tuple[float, float, float, float]]]:
        if Picamera2 is not None and isinstance(self.cap, Picamera2):
            try:
                # Same as capture_array("main"), but keeps the metadata so the age of the frame can be measured:
                # time from the start of the sensor read-out to the moment this loop got the frame. That is the
                # vision half of the loop delay (exposure + ISP + queueing behind earlier frames).
                request = self.cap.capture_request()
                try:
                    frame = request.make_array("main")
                    self.last_frame_age_s = self._frame_age(request.get_metadata())
                finally:
                    request.release()
                ok = True
            except Exception:
                ok, frame = False, None
        else:
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
        now = time.perf_counter()
        dt = max(1e-3, now - self.last_frame_time)
        self.last_frame_time = now
        if not ok:
            return False, None, None, np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), dt, None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        gray_full = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        proc_gray = cv2.resize(gray_full, (self.proc_width, self.proc_height), interpolation=cv2.INTER_AREA)
        if self.flat_gain is not None:
            proc_gray = np.clip(proc_gray * self.flat_gain, 0, 255).astype(np.uint8)

        if self.detector_mode == "old":
            # Brightness threshold + Hough circle. No background needed, so no empty-platform
            # step; it cannot be captured by dark hardware, but loses the ball where glare
            # makes the plate brighter than the ball.
            mask = self._get_mask(proc_gray)
            det_proc = self._detect_ball(proc_gray)
        else:
            if self.bg_init_frames < BG_INIT_FRAMES:
                self._accumulate_background(proc_gray)
                return False, None, None, frame, np.zeros_like(proc_gray), dt, None
            if now < self.bg_suppress_until:
                return False, None, None, frame, np.zeros_like(proc_gray), dt, None

            search = None
            if self.last_detection is not None:
                elapsed = max(1e-3, now - self.last_detection_time)
                predicted = np.array(self.last_detection[:2]) + self.detection_velocity * elapsed
                gate = min(260.0, max(80.0, float(np.linalg.norm(self.detection_velocity) * elapsed * 2.0 + 60.0)))
                search = ((predicted[0] * self.proc_scale, predicted[1] * self.proc_scale), gate * self.proc_scale)
            det_proc, mask = self._detect_ball_bgsub(proc_gray, search)
            if det_proc is None:
                # Fall back to the original brightness/Hough detector, which still does well
                # wherever the ball is clearly brighter than what it sits on.
                det_proc = self._detect_ball(proc_gray)

            # Only adapt the background while a ball is being tracked. With no confirmed track
            # the frame may contain a hand placing the ball, or a ball not yet confirmed --
            # blending those in would erase the ball and leave a "ghost" of the hand behind.
            if self.last_detection is not None:
                freeze_at = [(self.last_detection[0] * self.proc_scale, self.last_detection[1] * self.proc_scale)]
                if det_proc is not None:
                    freeze_at.append((det_proc[0], det_proc[1]))
                self._update_background(proc_gray, freeze_at)

        if self.debug_dir is not None:
            self.debug_count += 1
            if self.debug_count % self.debug_every == 0:
                self._save_debug(proc_gray, det_proc)

        # Detection runs at proc scale; everything downstream expects full-frame pixels.
        det = None
        if det_proc is not None:
            inv = 1.0 / self.proc_scale
            det = (det_proc[0] * inv, det_proc[1] * inv, det_proc[2] * inv, det_proc[3])

        if det is not None and det[3] >= self.min_confidence:
            if self.last_detection is not None:
                elapsed = max(1e-3, now - self.last_detection_time)
                predicted = np.array(self.last_detection[:2]) + self.detection_velocity * elapsed
                distance = float(np.linalg.norm(np.array(det[:2]) - predicted))
                adaptive_gate = min(260.0, max(80.0, float(np.linalg.norm(self.detection_velocity) * elapsed * 2.0 + 60.0)))
                if distance > adaptive_gate:
                    det = None
                    self.missed_frames += 1
                else:
                    measured_velocity = (np.array(det[:2]) - np.array(self.last_detection[:2])) / elapsed
                    self.detection_velocity = 0.35 * measured_velocity + 0.65 * self.detection_velocity
                    self.last_detection = det
                    self.last_detection_time = now
                    self.missed_frames = 0
            elif self.last_detection is None:
                if self.pending_detection is not None and np.hypot(det[0] - self.pending_detection[0], det[1] - self.pending_detection[1]) <= 120:
                    self.pending_frames += 1
                else:
                    self.pending_detection = det
                    self.pending_frames = 1
                if self.pending_frames >= 2:
                    self.last_detection = det
                    self.last_detection_time = now
                    self.detection_velocity[:] = 0.0
                    self.pending_detection = None
                    self.pending_frames = 0
                    self.missed_frames = 0
                else:
                    det = None
        else:
            det = None
            self.missed_frames += 1
            if self.last_detection is None:
                # Confirmation needs two consecutive sightings, so a blank frame restarts it.
                self.pending_detection = None
                self.pending_frames = 0
        # Drop the track once it has been lost for a while. Only when a track exists: missed_frames
        # stays above 5 until a new track is confirmed, and clearing the pending sighting on every
        # frame afterwards meant it could never reach 2 -- one 0.2 s dropout ended tracking for good.
        if self.missed_frames > 5 and self.last_detection is not None:
            self.last_detection = None
            self.last_detection_time = None
            self.detection_velocity[:] = 0.0
            self.pending_detection = None
            self.pending_frames = 0
        if det is None:
            return False, None, None, frame, mask, dt, None

        cx, cy = det[:2]
        h, w = frame.shape[:2]
        x_m, y_m = self._px_to_m(cx, cy, w, h)
        return True, x_m, y_m, frame, mask, dt, det

    def release(self) -> None:
        if self.cap is not None:
            if Picamera2 is not None and isinstance(self.cap, Picamera2):
                self.cap.stop()
                self.cap.close()
            else:
                self.cap.release()
