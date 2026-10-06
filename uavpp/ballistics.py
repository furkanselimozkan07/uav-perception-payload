"""Free-fall model of a released payload and release-point planning.

The payload is a point mass with quadratic air drag relative to the wind::

    m dv/dt = m g - 1/2 rho Cd A |v - w| (v - w)

integrated with RK4 from the release state until it reaches the ground. The
result is the ground displacement from the release point and the fall time.
Release planning inverts this: the aircraft must release at
``target - displacement(v_aircraft, wind)``.

For a multirotor that slows to a hover over the release point the
displacement is almost pure wind drift, which is why a hover release is the
default and a moving release is used only when time is short.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665
RHO_SEA_LEVEL = 1.225


@dataclass(frozen=True)
class Payload:
    name: str
    mass_kg: float
    cd: float  # drag coefficient (assumed, tumbling body)
    area_m2: float  # reference area
    target_cls: str  # which target this payload must reach

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> "Payload":
        return cls(name, float(cfg["mass_kg"]), float(cfg["cd"]), float(cfg["area_m2"]), str(cfg["target"]))

    @property
    def terminal_velocity(self) -> float:
        return float(np.sqrt(2 * self.mass_kg * G / (RHO_SEA_LEVEL * self.cd * self.area_m2)))


@dataclass(frozen=True)
class FallResult:
    d_north: float
    d_east: float
    t_fall: float
    impact_speed: float


def simulate_fall(payload: Payload, height_m: float, vel_ned: tuple[float, float, float] = (0.0, 0.0, 0.0),
                  wind_ne: tuple[float, float] = (0.0, 0.0), rho: float = RHO_SEA_LEVEL, dt: float = 0.005,
                  drag: bool = True) -> FallResult:
    """Integrate the fall from ``height_m`` with initial NED velocity; returns ground displacement."""
    if height_m <= 0:
        return FallResult(0.0, 0.0, 0.0, float(np.linalg.norm(vel_ned)))
    k = 0.5 * rho * payload.cd * payload.area_m2 / payload.mass_kg if drag else 0.0
    w = np.array([wind_ne[0], wind_ne[1], 0.0])
    g = np.array([0.0, 0.0, G])

    def acc(v: np.ndarray) -> np.ndarray:
        vr = v - w
        return g - k * np.linalg.norm(vr) * vr

    p = np.zeros(3)
    v = np.array(vel_ned, dtype=float)
    t = 0.0
    while p[2] < height_m:
        k1v = acc(v); k1p = v
        k2v = acc(v + 0.5 * dt * k1v); k2p = v + 0.5 * dt * k1v
        k3v = acc(v + 0.5 * dt * k2v); k3p = v + 0.5 * dt * k2v
        k4v = acc(v + dt * k3v); k4p = v + dt * k3v
        p_new = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if p_new[2] >= height_m:  # interpolate the exact ground crossing
            a = (height_m - p[2]) / (p_new[2] - p[2])
            p = p + a * (p_new - p)
            v = v + a * (v_new - v)
            t += a * dt
            break
        p, v, t = p_new, v_new, t + dt
        if t > 120:
            raise RuntimeError("fall integration did not converge")
    return FallResult(float(p[0]), float(p[1]), float(t), float(np.linalg.norm(v)))


def release_offset(payload: Payload, height_m: float, vel_ned: tuple[float, float, float],
                   wind_ne: tuple[float, float]) -> np.ndarray:
    """NE vector from the release point to the impact point."""
    r = simulate_fall(payload, height_m, vel_ned, wind_ne)
    return np.array([r.d_north, r.d_east])


def plan_release_point(target_ne: np.ndarray, payload: Payload, height_m: float,
                       vel_ned: tuple[float, float, float] = (0.0, 0.0, 0.0),
                       wind_ne: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """NE point at which to release so the payload lands on ``target_ne``."""
    return np.asarray(target_ne, dtype=float) - release_offset(payload, height_m, vel_ned, wind_ne)


def should_release(aircraft_ne: np.ndarray, release_ne: np.ndarray, groundspeed: float,
                   tol_m: float = 1.5, max_speed_hover: float = 1.0, servo_latency_s: float = 0.15,
                   vel_ne: tuple[float, float] = (0.0, 0.0)) -> bool:
    """Release trigger.

    Hover release: inside ``tol_m`` and slower than ``max_speed_hover``.
    Moving release: fire when the release point is reached after compensating
    the servo/actuation latency (look-ahead along the velocity vector).
    """
    aircraft_ne = np.asarray(aircraft_ne, dtype=float)
    if groundspeed <= max_speed_hover:
        return float(np.linalg.norm(aircraft_ne - release_ne)) <= tol_m
    ahead = aircraft_ne + np.asarray(vel_ne) * servo_latency_s
    to_rel = np.asarray(release_ne) - ahead
    along = float(to_rel @ np.asarray(vel_ne)) / max(groundspeed, 1e-6)
    cross = float(np.linalg.norm(to_rel - along * np.asarray(vel_ne) / groundspeed))
    return along <= 0.0 and cross <= tol_m
