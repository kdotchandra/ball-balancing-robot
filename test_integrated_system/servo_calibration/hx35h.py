from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import serial


@dataclass
class HX35HBus:
    port: str
    baudrate: int = 115200
    timeout: float = 0.08

    def __post_init__(self) -> None:
        if not Path(self.port).exists():
            ports = sorted(
                str(path)
                for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/serial*")
                for path in Path("/dev").glob(Path(pattern).name)
            )
            found = ", ".join(ports) if ports else "none"
            raise FileNotFoundError(
                f"Serial port {self.port} does not exist. Found: {found}. "
                "Connect a suitable UART/half-duplex interface first."
            )
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=self.timeout)
            time.sleep(0.15)
            self.serial.reset_input_buffer()
        except Exception as exc:
            if getattr(self, "serial", None) is not None and self.serial.is_open:
                self.serial.close()
            raise RuntimeError(f"Cannot open {self.port}: {exc}") from exc

    @staticmethod
    def checksum(values: list[int]) -> int:
        return (~sum(values)) & 0xFF

    @classmethod
    def move_packet(cls, servo_id: int, position: int, move_ms: int) -> bytes:
        if not 0 <= servo_id <= 253:
            raise ValueError("servo_id must be 0..253")
        if not 0 <= position <= 1000:
            raise ValueError("position must be 0..1000")
        values = [servo_id, 7, 1, position & 0xFF, position >> 8, move_ms & 0xFF, move_ms >> 8]
        return bytes([0x55, 0x55, *values, cls.checksum(values)])

    @classmethod
    def read_packet(cls, servo_id: int) -> bytes:
        values = [servo_id, 3, 28]
        return bytes([0x55, 0x55, *values, cls.checksum(values)])

    def move(self, servo_id: int, position: int, move_ms: int = 500) -> None:
        self.serial.write(self.move_packet(servo_id, position, move_ms))
        self.serial.flush()

    def read_position(self, servo_id: int, retries: int = 3) -> int:
        for _ in range(retries):
            self.serial.reset_input_buffer()
            self.serial.write(self.read_packet(servo_id))
            self.serial.flush()
            header = bytearray()
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                byte = self.serial.read(1)
                if not byte:
                    continue
                header.append(byte[0])
                if len(header) >= 2 and header[-2:] == b"\x55\x55":
                    break
            if len(header) < 2 or header[-2:] != b"\x55\x55":
                continue
            body = self.serial.read(2)
            if len(body) != 2:
                continue
            length = body[1]
            payload = self.serial.read(length - 1)
            if len(payload) != length - 1:
                continue
            packet = bytes(header[-2:] + body + payload)
            if len(packet) < 7 or packet[2] != servo_id or packet[4] != 28:
                continue
            if self.checksum(list(packet[2:-1])) != packet[-1]:
                continue
            position = packet[5] | (packet[6] << 8)
            if 0 <= position <= 1000:
                return position
            time.sleep(0.03)
        raise TimeoutError(f"No valid position response from servo {servo_id}")

    def close(self) -> None:
        self.serial.close()
