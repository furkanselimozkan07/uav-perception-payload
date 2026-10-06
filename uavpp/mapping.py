"""Georeferenced mosaic for the Risk Mapping task.

Feature-based stitching (``cv2.Stitcher``) looks good on texture-rich scenes
but drifts or fails over grass and repetitive debris. Because every frame
already carries a pose, each keyframe can instead be projected straight onto
a north-up ground grid: for a flat ground the image->ground mapping is an
exact homography, computed from the four image corners geolocated with
``geo.pixel_to_ned_offset``. Overlaps are blended with feather weights
(distance to the image border) so seams fade instead of cutting.

Pose-only placement inherits attitude bias: a 1 deg roll bias shifts the
footprint ~0.5 m at 30 m AGL, in *opposite* directions on alternate
lawnmower lanes, which shows up as ghosting. So each new keyframe is then
refined against the mosaic built so far with phase correlation on the
overlap (translation only, bounded by ``max_shift_m``): the pose gives the
coarse placement, the image gives the last half metre.

The output is a north-up PNG/JPEG with a known metres-per-pixel scale and
origin, which is also what makes the map useful to a first responder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geo import CameraModel, ne_offset, pixel_to_ned_offset
from .types import FramePacket, GeoPoint


@dataclass
class MosaicGrid:
    origin: GeoPoint  # NE (0, 0)
    north_min: float
    north_max: float
    east_min: float
    east_max: float
    res_m: float = 0.05  # metres per pixel

    @property
    def shape(self) -> tuple[int, int]:
        return (int(np.ceil((self.north_max - self.north_min) / self.res_m)),
                int(np.ceil((self.east_max - self.east_min) / self.res_m)))

    def ne_to_px(self, n: float, e: float) -> tuple[float, float]:
        """Canvas pixel (x right = east, y down = south)."""
        return (e - self.east_min) / self.res_m, (self.north_max - n) / self.res_m


class GeoMosaic:
    def __init__(self, cam: CameraModel, grid: MosaicGrid, min_spacing_m: float = 8.0, border_px: int = 8,
                 refine: bool = True, max_shift_m: float = 3.0, min_response: float = 0.08):
        self.cam, self.grid = cam, grid
        h, w = grid.shape
        self._acc = np.zeros((h, w, 3), np.float32)
        self._wsum = np.zeros((h, w), np.float32)
        self.min_spacing = min_spacing_m
        self._last_ne: Optional[np.ndarray] = None
        self.n_frames = 0
        self.refine, self.max_shift_px, self.min_response = refine, max_shift_m / grid.res_m, min_response
        self.corrections_m: list[float] = []
        # feather weight: distance to the nearest image border, normalised
        yy, xx = np.mgrid[0:cam.height, 0:cam.width].astype(np.float32)
        d = np.minimum.reduce([xx, yy, cam.width - 1 - xx, cam.height - 1 - yy])
        self._feather = np.clip((d - border_px) / (0.25 * min(cam.width, cam.height)), 0, 1)

    def _footprint_homography(self, pkt: FramePacket) -> Optional[np.ndarray]:
        import cv2

        tel = pkt.telemetry
        a_ne = np.array(ne_offset(self.grid.origin, tel.position))
        w, h = self.cam.width, self.cam.height
        src, dst = [], []
        for u, v in [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]:
            off = pixel_to_ned_offset(self.cam, u, v, tel.attitude, tel.position.alt)
            if off is None:
                return None
            n, e = a_ne + off
            src.append((u, v))
            dst.append(self.grid.ne_to_px(n, e))
        return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))

    def add(self, pkt: FramePacket, force: bool = False) -> bool:
        """Add a frame if it is far enough from the previous keyframe. Returns True if used."""
        import cv2

        tel = pkt.telemetry
        if tel is None:
            return False
        ne = np.array(ne_offset(self.grid.origin, tel.position))
        if not force and self._last_ne is not None and np.linalg.norm(ne - self._last_ne) < self.min_spacing:
            return False
        H = self._footprint_homography(pkt)
        if H is None:
            return False
        hh, ww = self.grid.shape
        warped = cv2.warpPerspective(pkt.frame.astype(np.float32), H, (ww, hh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)
        wmask = cv2.warpPerspective(self._feather, H, (ww, hh), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        if self.refine and self.n_frames > 0:
            shift = self._register(warped, wmask)
            if shift is not None:
                T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], dtype=np.float64)
                H = T @ H
                warped = cv2.warpPerspective(pkt.frame.astype(np.float32), H, (ww, hh), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT)
                wmask = cv2.warpPerspective(self._feather, H, (ww, hh), flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)
                self.corrections_m.append(float(np.hypot(*shift)) * self.grid.res_m)
        self._acc += warped * wmask[..., None]
        self._wsum += wmask
        self._last_ne = ne
        self.n_frames += 1
        return True

    def _register(self, warped: np.ndarray, wmask: np.ndarray) -> Optional[tuple[float, float]]:
        """Translation (canvas px) that aligns ``warped`` to the current mosaic, or None."""
        import cv2

        overlap = (wmask > 0.2) & (self._wsum > 0.2)
        if overlap.sum() < 0.15 * max(1, (wmask > 0.2).sum()):
            return None
        ys, xs = np.nonzero(overlap)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if (y1 - y0) < 64 or (x1 - x0) < 64:
            return None
        mos = self._acc[y0:y1, x0:x1] / np.maximum(self._wsum[y0:y1, x0:x1, None], 1e-6)
        a = cv2.cvtColor(mos.astype(np.float32), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(warped[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        m = overlap[y0:y1, x0:x1].astype(np.float32)
        a = (a - a[m > 0].mean()) * m
        b = (b - b[m > 0].mean()) * m
        win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        if resp < self.min_response or np.hypot(dx, dy) > self.max_shift_px:
            return None
        return -dx, -dy

    def coverage(self) -> float:
        return float(np.mean(self._wsum > 1e-3))

    def render(self, background: int = 0) -> np.ndarray:
        out = np.full(self._acc.shape, background, np.float32)
        m = self._wsum > 1e-3
        out[m] = self._acc[m] / self._wsum[m, None]
        return np.clip(out, 0, 255).astype(np.uint8)
