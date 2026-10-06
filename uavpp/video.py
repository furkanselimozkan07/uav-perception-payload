"""Video capture, frame/telemetry synchronisation and stream output.

Two problems matter more on a UAV than raw FPS:

1. **Latency must stay bounded.** If inference is slower than the camera, a
   naive ``cap.read()`` loop drains an ever-growing buffer and the pipeline
   ends up working on frames that are seconds old. ``LatestFrameSource`` reads
   in a background thread and keeps only the newest frame, so the consumer
   always sees fresh data and drops are counted rather than hidden.

2. **Every frame needs the attitude at its own capture time.** At 30 deg/s of
   yaw a 100 ms mismatch moves a target ~5 % of the image width, which at
   30 m AGL is several metres on the ground. ``TelemetryBuffer`` stores
   timestamped autopilot samples and interpolates position and attitude to
   the frame timestamp (with a configurable camera latency offset).
"""
from __future__ import annotations

import bisect
import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np

from .types import Attitude, FramePacket, GeoPoint, Telemetry


# --------------------------------------------------------------------------- GStreamer helpers
def jetson_csi_pipeline(sensor_id: int = 0, width: int = 1920, height: int = 1080, fps: int = 30,
                        out_width: int = 1280, out_height: int = 720) -> str:
    """nvarguscamerasrc pipeline for a CSI camera on Jetson (hardware ISP + scaling)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw,width={out_width},height={out_height},format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )


def rtsp_pipeline(url: str, latency_ms: int = 50, hw_decode: bool = True) -> str:
    """Low-latency RTSP H.264 receive pipeline (hardware decode on Jetson)."""
    dec = "nvv4l2decoder ! nvvidconv" if hw_decode else "avdec_h264"
    return (
        f"rtspsrc location={url} latency={latency_ms} ! rtph264depay ! h264parse ! {dec} ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )


def udp_h264_sender_pipeline(host: str, port: int, bitrate_kbps: int = 2000, hw_encode: bool = False) -> str:
    """Pipeline string for cv2.VideoWriter that streams annotated frames to the GCS over RTP/UDP."""
    enc = ("nvvidconv ! nvv4l2h264enc insert-sps-pps=true bitrate=%d" % (bitrate_kbps * 1000)) if hw_encode \
        else "x264enc tune=zerolatency speed-preset=ultrafast bitrate=%d key-int-max=30" % bitrate_kbps
    return (f"appsrc ! videoconvert ! {enc} ! h264parse ! rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={host} port={port} sync=false")


# --------------------------------------------------------------------------- telemetry buffer
def _lerp_angle(a: float, b: float, w: float) -> float:
    d = (b - a + np.pi) % (2 * np.pi) - np.pi
    return a + w * d


class TelemetryBuffer:
    """Thread-safe ring buffer of telemetry with time interpolation."""

    def __init__(self, maxlen: int = 2000):
        self._t: list[float] = []
        self._s: list[Telemetry] = []
        self._maxlen = maxlen
        self._lock = threading.Lock()

    def push(self, tel: Telemetry) -> None:
        with self._lock:
            if self._t and tel.t <= self._t[-1]:
                return  # out-of-order or duplicate sample
            self._t.append(tel.t)
            self._s.append(tel)
            if len(self._t) > self._maxlen:
                del self._t[0], self._s[0]

    def __len__(self) -> int:
        return len(self._t)

    def at(self, t: float, max_gap: float = 0.5) -> Optional[Telemetry]:
        """Telemetry interpolated to time ``t``; None if t is outside the buffer or the gap is too large."""
        with self._lock:
            if not self._t or t < self._t[0] or t > self._t[-1]:
                return None
            i = bisect.bisect_left(self._t, t)
            if self._t[i] == t:
                return self._s[i]
            a, b = self._s[i - 1], self._s[i]
        if b.t - a.t > max_gap:
            return None
        w = (t - a.t) / (b.t - a.t)
        pos = GeoPoint(a.position.lat + w * (b.position.lat - a.position.lat),
                       a.position.lon + w * (b.position.lon - a.position.lon),
                       a.position.alt + w * (b.position.alt - a.position.alt))
        att = Attitude(_lerp_angle(a.attitude.roll, b.attitude.roll, w),
                       _lerp_angle(a.attitude.pitch, b.attitude.pitch, w),
                       _lerp_angle(a.attitude.yaw, b.attitude.yaw, w) % (2 * np.pi))
        vel = tuple(float(x + w * (y - x)) for x, y in zip(a.vel_ned, b.vel_ned))
        return Telemetry(t, pos, att, vel)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- frame sources
class LatestFrameSource:
    """Background reader that always exposes only the newest frame.

    ``source`` is anything ``cv2.VideoCapture`` accepts: a device index, a file,
    an RTSP URL or a GStreamer pipeline string (``use_gstreamer=True``).
    """

    def __init__(self, source, use_gstreamer: bool = False, clock: Callable[[], float] = time.monotonic,
                 latency_s: float = 0.0):
        import cv2

        api = cv2.CAP_GSTREAMER if use_gstreamer else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(source, api)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video source: {source!r}")
        self._clock = clock
        self._latency = latency_s  # sensor-to-host delay, subtracted from the timestamp
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._t = 0.0
        self._seq = 0
        self._read_seq = 0
        self.dropped = 0
        self.captured = 0
        self._running = True
        self._eof = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self._eof = True
                break
            t = self._clock() - self._latency
            with self._lock:
                if self._seq > self._read_seq:
                    self.dropped += 1  # previous frame was never consumed
                self._frame, self._t = frame, t
                self._seq += 1
                self.captured += 1

    def read(self, timeout: float = 1.0) -> Optional[tuple[np.ndarray, float]]:
        """Block until a frame newer than the last one read is available."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._seq > self._read_seq:
                    self._read_seq = self._seq
                    return self._frame, self._t
            if self._eof:
                return None
            time.sleep(0.001)
        return None

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def synced_packets(frames: Iterator[tuple[np.ndarray, float]], telemetry: TelemetryBuffer,
                   max_gap: float = 0.5) -> Iterator[FramePacket]:
    """Pair each (frame, t) with telemetry interpolated to t; frames without valid telemetry are skipped."""
    for frame, t in frames:
        tel = telemetry.at(t, max_gap=max_gap)
        if tel is not None:
            yield FramePacket(frame, t, tel)


class FpsMeter:
    """Exponential moving average of frames per second."""

    def __init__(self, alpha: float = 0.1):
        self.alpha, self.fps, self._last = alpha, 0.0, None

    def tick(self, t: Optional[float] = None) -> float:
        t = time.monotonic() if t is None else t
        if self._last is not None and t > self._last:
            inst = 1.0 / (t - self._last)
            self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = t
        return self.fps
