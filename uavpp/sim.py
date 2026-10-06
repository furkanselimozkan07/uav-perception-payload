"""Synthetic search area and camera renderer for closed-loop testing.

The ground is a procedurally textured orthophoto (grass, bushes, debris)
with the two SUAS targets placed at known positions: a mannequin (orange,
elongated, random heading) and an open pop-up tent (blue). The camera image
for any pose is rendered *exactly* by a ground->image homography, so the
geometry the pipeline sees is consistent with ``geo.py`` while the pose it is
given comes from noisy, biased telemetry - which is the error that matters.

This is a test harness, not a photorealistic simulator: it exercises
geometry, timing, tracking and release logic, not detector generalisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geo import CameraModel, ne_offset, pixel_to_ned_offset
from .mapping import MosaicGrid
from .types import Attitude, GeoPoint

# BGR colours, chosen so ColorDetector can separate them from grass/debris
MANNEQUIN_BGR = (40, 120, 235)  # orange
TENT_BGR = (200, 110, 30)  # blue
COLOR_RANGES = {  # HSV ranges for ColorDetector, matching the colours above
    "mannequin": [(5, 120, 120), (22, 255, 255)],
    "tent": [(100, 120, 90), (125, 255, 255)],
}


@dataclass
class SimTarget:
    cls: str
    north: float
    east: float
    length_m: float
    width_m: float
    heading_rad: float


class GroundScene:
    def __init__(self, grid: MosaicGrid, rng: np.random.Generator, n_debris: int = 60, n_bushes: int = 40):
        import cv2

        self.grid, self.rng = grid, rng
        h, w = grid.shape
        # grass: low-frequency + high-frequency noise in green
        base = rng.normal(0, 1, (h // 16 + 1, w // 16 + 1)).astype(np.float32)
        base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        fine = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2)
        tex = 0.6 * base + 0.4 * fine
        img = np.zeros((h, w, 3), np.float32)
        img[..., 0] = 45 + 10 * tex
        img[..., 1] = 110 + 22 * tex
        img[..., 2] = 70 + 12 * tex
        self.image = img
        for _ in range(n_bushes):
            self._blob((25, 70, 30), rng.uniform(1.0, 2.5))
        for _ in range(n_debris):
            g = rng.uniform(80, 170)
            self._rect(self._rand_ne(), rng.uniform(0.4, 3.0), rng.uniform(0.3, 1.5), rng.uniform(0, np.pi),
                       (g * 0.9, g, g * 1.05))
        self.targets: list[SimTarget] = []

    def _rand_ne(self, margin: float = 5.0) -> tuple[float, float]:
        g = self.grid
        return (float(self.rng.uniform(g.north_min + margin, g.north_max - margin)),
                float(self.rng.uniform(g.east_min + margin, g.east_max - margin)))

    def _rect(self, ne, length, width, heading, color) -> None:
        import cv2

        n, e = ne
        c, s = np.cos(heading), np.sin(heading)
        corners = []
        for dl, dw in [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]:
            cn = n + dl * length * c - dw * width * s
            ce = e + dl * length * s + dw * width * c
            corners.append(self.grid.ne_to_px(cn, ce))
        cv2.fillPoly(self.image, [np.int32(np.round(corners))], color, lineType=cv2.LINE_AA)

    def _blob(self, color, radius_m) -> None:
        import cv2

        x, y = self.grid.ne_to_px(*self._rand_ne())
        cv2.circle(self.image, (int(x), int(y)), int(radius_m / self.grid.res_m), color, -1, cv2.LINE_AA)

    def place_target(self, cls: str, ne: Optional[tuple[float, float]] = None) -> SimTarget:
        ne = ne or self._rand_ne(margin=15.0)
        if cls == "mannequin":
            t = SimTarget(cls, ne[0], ne[1], 1.8, 0.5, float(self.rng.uniform(0, np.pi)))
            self._rect(ne, t.length_m, t.width_m, t.heading_rad, MANNEQUIN_BGR)
        else:
            t = SimTarget(cls, ne[0], ne[1], 2.4, 2.4, float(self.rng.uniform(0, np.pi / 2)))
            self._rect(ne, t.length_m, t.width_m, t.heading_rad, TENT_BGR)
        self.targets.append(t)
        return t

    def render(self, cam: CameraModel, position: GeoPoint, att: Attitude, noise: float = 3.0) -> np.ndarray:
        """Camera image for a true pose (pinhole, flat ground)."""
        import cv2

        a_ne = np.array(ne_offset(self.grid.origin, position))
        src, dst = [], []
        for u, v in [(0, 0), (cam.width - 1, 0), (cam.width - 1, cam.height - 1), (0, cam.height - 1)]:
            off = pixel_to_ned_offset(cam, u, v, att, position.alt)
            if off is None:
                return np.zeros((cam.height, cam.width, 3), np.uint8)
            n, e = a_ne + off
            src.append((u, v))
            dst.append(self.grid.ne_to_px(n, e))
        H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))  # image -> ortho
        img = cv2.warpPerspective(self.image, H, (cam.width, cam.height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(60, 90, 60))
        if noise > 0:
            img = img + self.rng.normal(0, noise, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


def lawnmower(north_min: float, north_max: float, east_min: float, east_max: float, spacing: float) -> list[tuple[float, float]]:
    """Boustrophedon search pattern with north-south lanes."""
    pts, east, flip = [], east_min, False
    while east <= east_max + 1e-6:
        lane = [(north_min, east), (north_max, east)]
        pts += lane[::-1] if flip else lane
        east += spacing
        flip = not flip
    return pts
