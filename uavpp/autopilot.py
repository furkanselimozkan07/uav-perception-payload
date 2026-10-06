"""Autopilot abstraction: a MAVLink (ArduPilot) link and a kinematic simulator.

The mission logic only talks to ``Autopilot``; swapping the simulator for the
real aircraft is a one-line change in ``scripts/run_onboard.py``.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

import numpy as np

from .geo import ne_offset, offset_geopoint
from .types import Attitude, GeoPoint, Telemetry


class Autopilot(ABC):
    @abstractmethod
    def telemetry(self) -> Optional[Telemetry]: ...

    @abstractmethod
    def goto(self, target: GeoPoint) -> None:
        """Fly to ``target`` (alt is AGL) in guided mode and hold there."""

    @abstractmethod
    def resume_mission(self) -> None:
        """Return control to the autopilot's own waypoint mission (AUTO)."""

    @abstractmethod
    def set_servo(self, channel: int, pwm: int) -> None: ...

    @property
    @abstractmethod
    def laps_completed(self) -> int: ...

    def add_listener(self, fn: Callable[[Telemetry], None]) -> None:
        self._listeners = getattr(self, "_listeners", [])
        self._listeners.append(fn)

    def _notify(self, tel: Telemetry) -> None:
        for fn in getattr(self, "_listeners", []):
            fn(tel)


# --------------------------------------------------------------------------- MAVLink
class MavlinkAutopilot(Autopilot):
    """ArduPilot over MAVLink (serial ``/dev/ttyTHS1`` on Jetson, or UDP for SITL).

    Telemetry is assembled from ``ATTITUDE`` and ``GLOBAL_POSITION_INT`` and
    timestamped with the companion clock at reception, the same clock the
    camera thread uses, so ``TelemetryBuffer`` can interpolate between them.
    A waypoint lap is counted each time the autopilot reports reaching the
    last waypoint of the uploaded lap (``lap_last_seq``).
    """

    def __init__(self, url: str = "udpin:0.0.0.0:14550", baud: int = 921600, lap_last_seq: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic, stream_hz: int = 25):
        from pymavlink import mavutil

        self._mavutil = mavutil
        self.m = mavutil.mavlink_connection(url, baud=baud, source_system=255, source_component=191)
        self.m.wait_heartbeat(timeout=30)
        self.m.mav.request_data_stream_send(self.m.target_system, self.m.target_component,
                                            mavutil.mavlink.MAV_DATA_STREAM_ALL, stream_hz, 1)
        self._clock = clock
        self._att: Optional[Attitude] = None
        self._tel: Optional[Telemetry] = None
        self._laps = 0
        self.lap_last_seq = lap_last_seq
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._rx, daemon=True).start()

    def _rx(self) -> None:
        while self._running:
            msg = self.m.recv_match(type=["ATTITUDE", "GLOBAL_POSITION_INT", "MISSION_ITEM_REACHED"],
                                    blocking=True, timeout=1.0)
            if msg is None:
                continue
            kind = msg.get_type()
            if kind == "ATTITUDE":
                self._att = Attitude(msg.roll, msg.pitch, msg.yaw % (2 * np.pi))
            elif kind == "GLOBAL_POSITION_INT" and self._att is not None:
                tel = Telemetry(self._clock(), GeoPoint(msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0),
                                self._att, (msg.vx / 100.0, msg.vy / 100.0, msg.vz / 100.0))
                with self._lock:
                    self._tel = tel
                self._notify(tel)
            elif kind == "MISSION_ITEM_REACHED" and self.lap_last_seq is not None and msg.seq == self.lap_last_seq:
                self._laps += 1

    def telemetry(self) -> Optional[Telemetry]:
        with self._lock:
            return self._tel

    def _set_mode(self, mode: str) -> None:
        self.m.set_mode(self.m.mode_mapping()[mode])

    def goto(self, target: GeoPoint) -> None:
        mav = self._mavutil.mavlink
        self._set_mode("GUIDED")
        type_mask = 0b0000111111111000  # use position only
        self.m.mav.set_position_target_global_int_send(
            0, self.m.target_system, self.m.target_component, mav.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, type_mask,
            int(target.lat * 1e7), int(target.lon * 1e7), float(target.alt), 0, 0, 0, 0, 0, 0, 0, 0)

    def resume_mission(self) -> None:
        self._set_mode("AUTO")

    def set_servo(self, channel: int, pwm: int) -> None:
        mav = self._mavutil.mavlink
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, mav.MAV_CMD_DO_SET_SERVO,
                                     0, channel, pwm, 0, 0, 0, 0, 0)

    @property
    def laps_completed(self) -> int:
        return self._laps

    def close(self) -> None:
        self._running = False


# --------------------------------------------------------------------------- simulator
class SimAutopilot(Autopilot):
    """Kinematic multirotor for offline testing.

    Flies the waypoint lap in AUTO, flies to a guided target otherwise, and
    produces attitude from acceleration (a multirotor tilts to accelerate)
    plus configurable sensor noise and a constant attitude bias.
    """

    def __init__(self, origin: GeoPoint, lap_ne: Sequence[tuple[float, float]],
                 search_ne: Sequence[tuple[float, float]] = (), alt_agl: float = 30.0,
                 speed: float = 8.0, accel: float = 3.0, rng: Optional[np.random.Generator] = None,
                 att_noise_rad: float = np.radians(0.5), att_bias_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 pos_noise_m: float = 0.5, wind_ne: tuple[float, float] = (0.0, 0.0)):
        self.origin, self.alt, self.speed, self.accel = origin, alt_agl, speed, accel
        self.lap = [np.array(p, dtype=float) for p in lap_ne]
        # AUTO route: the waypoint lap once, then the search pattern on repeat
        self.route = self.lap + [np.array(p, dtype=float) for p in search_ne]
        self.rng = rng or np.random.default_rng(0)
        self.att_noise, self.att_bias, self.pos_noise = att_noise_rad, att_bias_rad, pos_noise_m
        self.wind = np.array(wind_ne, dtype=float)
        self.p = self.lap[0].copy() if self.lap else np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0.0
        self.mode = "AUTO"
        self._wp = 1 % max(1, len(self.route))
        self._laps = 0
        self.search_passes = 0  # completed passes over the search pattern
        self._guided: Optional[np.ndarray] = None
        self.yaw = 0.0
        self.servo_log: list[tuple[float, int, int]] = []
        self._true_att = Attitude(0, 0, 0)

    # truth (used by the simulator / renderer, never by the pipeline)
    @property
    def true_position(self) -> GeoPoint:
        return offset_geopoint(self.origin, float(self.p[0]), float(self.p[1]), self.alt)

    @property
    def true_attitude(self) -> Attitude:
        return self._true_att

    def step(self, dt: float) -> None:
        if self.mode == "AUTO" and self.route:
            goal = self.route[self._wp]
            if np.linalg.norm(goal - self.p) < 2.0:
                if self._wp == len(self.lap) - 1:
                    self._laps += 1
                nxt = self._wp + 1
                if nxt >= len(self.route):  # loop: search pattern if there is one, else the lap
                    nxt = len(self.lap) if len(self.route) > len(self.lap) else 0
                    self.search_passes += 1
                self._wp = nxt
                goal = self.route[self._wp]
        else:
            goal = self._guided if self._guided is not None else self.p
        to = goal - self.p
        dist = float(np.linalg.norm(to))
        v_des = np.zeros(2) if dist < 0.05 else to / dist * min(self.speed, np.sqrt(2 * self.accel * dist))
        dv = v_des - self.v
        a = dv / dt
        n = np.linalg.norm(a)
        if n > self.accel:
            a = a / n * self.accel
        self.v = self.v + a * dt
        self.p = self.p + self.v * dt
        self.t += dt
        if np.linalg.norm(self.v) > 0.5:
            self.yaw = float(np.arctan2(self.v[1], self.v[0]) % (2 * np.pi))
        # small-angle tilt needed for the commanded acceleration, in body axes
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        a_fwd, a_right = c * a[0] + s * a[1], -s * a[0] + c * a[1]
        self._true_att = Attitude(float(np.arctan2(a_right, 9.81)), float(-np.arctan2(a_fwd, 9.81)), self.yaw)
        self._notify(self.telemetry())

    def telemetry(self) -> Telemetry:
        b, n = self.att_bias, self.att_noise
        att = Attitude(self._true_att.roll + b[0] + self.rng.normal(0, n),
                       self._true_att.pitch + b[1] + self.rng.normal(0, n),
                       (self._true_att.yaw + b[2] + self.rng.normal(0, n)) % (2 * np.pi))
        pn = self.p + self.rng.normal(0, self.pos_noise, 2)
        pos = offset_geopoint(self.origin, float(pn[0]), float(pn[1]), self.alt + self.rng.normal(0, 0.3))
        return Telemetry(self.t, pos, att, (float(self.v[0]), float(self.v[1]), 0.0))

    def goto(self, target: GeoPoint) -> None:
        self.mode = "GUIDED"
        self._guided = np.array(ne_offset(self.origin, target))

    def resume_mission(self) -> None:
        self.mode = "AUTO"
        self._guided = None

    def set_servo(self, channel: int, pwm: int) -> None:
        self.servo_log.append((self.t, channel, pwm))

    @property
    def laps_completed(self) -> int:
        return self._laps
