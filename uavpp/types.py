"""Shared data types for the perception / payload pipeline.

All internal quantities are SI: metres, seconds, radians. Angles are only
converted from degrees at the configuration boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GeoPoint:
    """WGS-84 position. ``alt`` is height above the local ground (AGL)."""

    lat: float  # degrees
    lon: float  # degrees
    alt: float = 0.0  # metres AGL


@dataclass(frozen=True)
class Attitude:
    """Aircraft attitude, aerospace ZYX convention (yaw, then pitch, then roll)."""

    roll: float  # rad, right wing down positive
    pitch: float  # rad, nose up positive
    yaw: float  # rad, clockwise from true north


@dataclass(frozen=True)
class Telemetry:
    """One autopilot state sample, timestamped on the companion computer clock."""

    t: float
    position: GeoPoint
    attitude: Attitude
    vel_ned: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m/s

    @property
    def groundspeed(self) -> float:
        return float(np.hypot(self.vel_ned[0], self.vel_ned[1]))


@dataclass(frozen=True)
class Detection:
    """One object detection in pixel coordinates."""

    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class GeoFix:
    """A detection projected onto the ground, with a 2x2 NE covariance (m^2)."""

    cls: str
    conf: float
    point: GeoPoint
    cov_ne: np.ndarray
    t: float


@dataclass
class TargetEstimate:
    """Fused estimate of a static ground target built from many GeoFix samples."""

    track_id: int
    cls: str
    point: GeoPoint
    cov_ne: np.ndarray
    hits: int = 0
    mean_conf: float = 0.0
    confirmed: bool = False
    delivered: bool = False
    history: list = field(default_factory=list)

    @property
    def sigma_m(self) -> float:
        """1-sigma horizontal radius (square root of the larger eigenvalue)."""
        return float(np.sqrt(np.max(np.linalg.eigvalsh(self.cov_ne))))


@dataclass(frozen=True)
class FramePacket:
    """A video frame paired with the telemetry interpolated to its capture time."""

    frame: np.ndarray
    t: float
    telemetry: Optional[Telemetry] = None
