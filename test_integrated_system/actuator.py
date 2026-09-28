from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from servo_calibration.hx35h import HX35HBus


@dataclass
class ServoSafety:
    neutral: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    max_delta: int = 20

    def clamp(self, target: np.ndarray, previous: np.ndarray) -> np.ndarray:
        target = np.clip(np.rint(target), self.minimum, self.maximum)
        target = np.clip(target, previous - self.max_delta, previous + self.max_delta)
        return target.astype(int)


class ServoGateway:
    """Safe actuator boundary. Default mode is dry-run; serial needs explicit enable."""

    def __init__(self, safety: ServoSafety, port: str | None = None, move_ms: int | None = None, min_confidence: float = 0.45):
        self.safety = safety
        self.min_confidence = min_confidence
        self.port = port or os.environ.get("SERVO_PORT")
        self.enabled = os.environ.get("SERVO_OUTPUT", "0") == "1"
        self.readback_enabled = os.environ.get("SERVO_READBACK", "0") == "1"
        self.command_period = float(os.environ.get("SERVO_COMMAND_PERIOD", "0.10"))
        self.safety.max_delta = int(os.environ.get("SERVO_MAX_DELTA", str(self.safety.max_delta)))
        if move_ms is None:
            move_ms = int(os.environ.get("SERVO_MOVE_MS", "150"))
        self.last_command_time = 0.0
        self.move_ms = move_ms
        self.previous = safety.neutral.copy().astype(int)
        self.last_feedback: np.ndarray | None = None
        self.bus = None
        # One serial bus, half duplex: commands and position reads must not interleave.
        self.bus_lock = threading.Lock()
        # Diagnostic: measured servo positions, read in the background one servo at a time
        # (~8 ms each, so each servo is refreshed about every 25 ms) and logged next to the
        # commanded ones. A command can wait up to one read (~8 ms) for the bus.
        self.feedback_log = os.environ.get("SERVO_FEEDBACK_LOG", "0") == "1"
        self.feedback = np.full(len(safety.neutral), -1, dtype=int)
        self._feedback_stop = threading.Event()
        self._feedback_thread: threading.Thread | None = None
        if self.enabled:
            if not self.port:
                raise RuntimeError("SERVO_OUTPUT=1 requires SERVO_PORT")
            self.bus = HX35HBus(self.port, timeout=0.08)
            if self.feedback_log:
                # Short timeout so a missed reply holds the bus for 30 ms at most, not 80.
                self.bus.timeout = 0.03
                self.bus.serial.timeout = 0.03
                self._feedback_thread = threading.Thread(target=self._feedback_loop, daemon=True)
                self._feedback_thread.start()

    def _feedback_loop(self) -> None:
        servo_ids = range(1, len(self.feedback) + 1)
        while not self._feedback_stop.is_set():
            for servo_id in servo_ids:
                with self.bus_lock:
                    try:
                        self.feedback[servo_id - 1] = self.bus.read_position(servo_id, retries=1)
                    except (TimeoutError, OSError):
                        pass
                time.sleep(0.001)   # let a waiting command take the bus

    def send_positions(self, positions: np.ndarray, confidence: float) -> np.ndarray:
        if confidence < self.min_confidence:
            return self.previous.copy()
        safe = self.safety.clamp(np.asarray(positions), self.previous)
        if self.enabled and time.monotonic() - self.last_command_time < self.command_period:
            return self.previous.copy()
        if self.enabled and self.bus is not None:
            with self.bus_lock:
                for servo_id, position in enumerate(safe, start=1):
                    self.bus.move(servo_id, int(position), self.move_ms)
            self.last_command_time = time.monotonic()
            if self.readback_enabled:
                time.sleep(max(0.10, self.move_ms / 1000.0 * 0.75))
                try:
                    with self.bus_lock:
                        feedback = [self.bus.read_position(servo_id, retries=5) for servo_id in range(1, len(safe) + 1)]
                    self.last_feedback = np.array(feedback, dtype=int)
                except (TimeoutError, OSError) as exc:
                    print(f"[Actuator] HX-35H readback failed after command (ignored): {exc}")
        self.previous = safe
        return safe.copy()

    def neutralize(self) -> None:
        safe = self.safety.neutral.astype(int)
        if self.enabled and self.bus is not None:
            with self.bus_lock:
                for servo_id, position in enumerate(safe, start=1):
                    self.bus.move(servo_id, int(position), self.move_ms)
        self.previous = safe.copy()

    def close(self) -> None:
        self._feedback_stop.set()
        if self._feedback_thread is not None:
            self._feedback_thread.join(timeout=0.5)
        self.neutralize()
        if self.bus is not None:
            self.bus.close()