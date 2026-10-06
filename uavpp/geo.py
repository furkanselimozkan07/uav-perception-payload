"""Camera model and pixel -> ground geolocation.

Frames
------
* NED  : local tangent plane, x north, y east, z down.
* Body : FRD, x forward (nose), y right wing, z down.
* Cam  : OpenCV optical frame, x right, y down, z along the optical axis.

The default mount is a nadir camera whose image top points towards the nose,
so ``x_cam = y_body``, ``y_cam = -x_body`` and ``z_cam = z_body``. A gimbal or
tilted mount is modelled with an extra fixed rotation (``mount_rpy``).

A ray through pixel (u, v) is rotated into NED and intersected with a flat
ground plane at the aircraft's height above ground. The flat-earth
north/east offset is then turned into latitude / longitude with the WGS-84
meridian and prime-vertical radii, which is accurate to centimetres over the
few hundred metres a SUAS search area spans.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .types import Attitude, GeoPoint

# WGS-84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)

# Nadir mount: columns are the camera axes expressed in the body frame.
R_BODY_CAM_NADIR = np.array([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- rotations
def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def body_to_ned(att: Attitude) -> np.ndarray:
    """Rotation taking body-frame vectors to NED (ZYX / yaw-pitch-roll)."""
    return rot_z(att.yaw) @ rot_y(att.pitch) @ rot_x(att.roll)


# --------------------------------------------------------------------------- WGS-84
def earth_radii(lat_deg: float) -> tuple[float, float]:
    """Return (meridian radius M, prime-vertical radius N) at a latitude."""
    s = np.sin(np.radians(lat_deg))
    w = np.sqrt(1.0 - _E2 * s * s)
    return _A * (1.0 - _E2) / w ** 3, _A / w


def offset_geopoint(origin: GeoPoint, d_north: float, d_east: float, alt: float = 0.0) -> GeoPoint:
    """Move ``origin`` by a north/east offset in metres (local flat-earth)."""
    m, n = earth_radii(origin.lat)
    lat = origin.lat + np.degrees(d_north / m)
    lon = origin.lon + np.degrees(d_east / (n * np.cos(np.radians(origin.lat))))
    return GeoPoint(float(lat), float(lon), alt)


def ne_offset(origin: GeoPoint, p: GeoPoint) -> tuple[float, float]:
    """North/east metres from ``origin`` to ``p`` (inverse of offset_geopoint)."""
    m, n = earth_radii(origin.lat)
    d_n = np.radians(p.lat - origin.lat) * m
    d_e = np.radians(p.lon - origin.lon) * n * np.cos(np.radians(origin.lat))
    return float(d_n), float(d_e)


# --------------------------------------------------------------------------- camera
@dataclass
class CameraModel:
    """Pinhole camera with optional OpenCV distortion and a fixed mount rotation."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: Optional[np.ndarray] = None
    mount_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad, extra rotation in body frame
    stabilized: bool = False  # True: gimbal holds nadir, ignore aircraft roll/pitch

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)

    @property
    def R_body_cam(self) -> np.ndarray:
        r, p, y = self.mount_rpy
        return rot_z(y) @ rot_y(p) @ rot_x(r) @ R_BODY_CAM_NADIR

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float, **kw) -> "CameraModel":
        f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, width=width, height=height, **kw)

    @classmethod
    def from_config(cls, cfg: dict) -> "CameraModel":
        mount = tuple(np.radians(cfg.get("mount_rpy_deg", [0, 0, 0])))
        common = dict(mount_rpy=mount, stabilized=bool(cfg.get("stabilized", False)))
        if "hfov_deg" in cfg:
            return cls.from_hfov(cfg["width"], cfg["height"], cfg["hfov_deg"], **common)
        dist = np.array(cfg["dist"], dtype=float) if cfg.get("dist") else None
        return cls(cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"], cfg["width"], cfg["height"], dist=dist, **common)

    def undistort_pixel(self, u: float, v: float) -> tuple[float, float]:
        if self.dist is None or not np.any(self.dist):
            return u, v
        import cv2

        pts = np.array([[[u, v]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def R_ned_cam(self, att: Attitude) -> np.ndarray:
        if self.stabilized:
            att = Attitude(0.0, 0.0, att.yaw)
        return body_to_ned(att) @ self.R_body_cam

    def ground_sample_distance(self, alt_agl: float) -> float:
        """Metres per pixel at the image centre for a nadir view."""
        return alt_agl / self.fx


# --------------------------------------------------------------------------- geolocation
def pixel_to_ned_offset(cam: CameraModel, u: float, v: float, att: Attitude, alt_agl: float) -> Optional[np.ndarray]:
    """North/east offset (m) from the aircraft to the ground point seen at pixel (u, v).

    Returns ``None`` if the ray does not hit the ground (points at/above horizon).
    """
    u, v = cam.undistort_pixel(u, v)
    ray_cam = np.linalg.solve(cam.K, np.array([u, v, 1.0]))
    ray_ned = cam.R_ned_cam(att) @ ray_cam
    if ray_ned[2] <= 1e-6:
        return None
    t = alt_agl / ray_ned[2]
    return ray_ned[:2] * t


def geolocate_pixel(cam: CameraModel, u: float, v: float, position: GeoPoint, att: Attitude) -> Optional[GeoPoint]:
    off = pixel_to_ned_offset(cam, u, v, att, position.alt)
    if off is None:
        return None
    return offset_geopoint(position, off[0], off[1])


def geolocate_with_covariance(
    cam: CameraModel,
    u: float,
    v: float,
    position: GeoPoint,
    att: Attitude,
    sigma_px: float = 2.0,
    sigma_att_rad: float = np.radians(1.5),
    sigma_alt_m: float = 1.0,
    sigma_pos_m: float = 1.5,
) -> Optional[tuple[GeoPoint, np.ndarray]]:
    """Geolocate a pixel and propagate input noise to a 2x2 NE covariance.

    Uses a central-difference Jacobian of the ground offset with respect to
    (u, v, roll, pitch, yaw, altitude); GNSS horizontal error is added as an
    isotropic term. Cheap enough to run for every detection.
    """
    base = pixel_to_ned_offset(cam, u, v, att, position.alt)
    if base is None:
        return None

    def f(x: Sequence[float]) -> np.ndarray:
        out = pixel_to_ned_offset(cam, x[0], x[1], Attitude(x[2], x[3], x[4]), x[5])
        return base if out is None else out

    x0 = np.array([u, v, att.roll, att.pitch, att.yaw, position.alt])
    steps = np.array([0.5, 0.5, 1e-4, 1e-4, 1e-4, 0.05])
    J = np.zeros((2, 6))
    for i, h in enumerate(steps):
        dx = np.zeros(6)
        dx[i] = h
        J[:, i] = (f(x0 + dx) - f(x0 - dx)) / (2 * h)
    S = np.diag([sigma_px, sigma_px, sigma_att_rad, sigma_att_rad, sigma_att_rad, sigma_alt_m]) ** 2
    cov = J @ S @ J.T + np.eye(2) * sigma_pos_m ** 2
    return offset_geopoint(position, base[0], base[1]), cov


def ground_to_pixel(cam: CameraModel, d_north: float, d_east: float, att: Attitude, alt_agl: float) -> Optional[tuple[float, float]]:
    """Project a ground point (NE offset from the aircraft) into the image (no distortion)."""
    p_ned = np.array([d_north, d_east, alt_agl])  # vector from camera to ground point
    p_cam = cam.R_ned_cam(att).T @ p_ned
    if p_cam[2] <= 1e-6:
        return None
    uvw = cam.K @ (p_cam / p_cam[2])
    return float(uvw[0]), float(uvw[1])
