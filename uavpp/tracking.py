"""Fusing repeated geolocations of static ground targets.

A single pixel->ground projection is noisy (attitude error dominates at
altitude), but a target is seen in many frames from different positions. Each
target is a static 2-D Kalman filter in a local north/east frame:

* **Association** - a new fix joins the track of the same class with the
  smallest Mahalanobis distance, if it passes a chi-square gate (99.9 %, 2 DOF)
  *and* a hard metric gate. Otherwise it opens a new tentative track.
* **Update** - information-weighted (Kalman) update with the fix covariance
  from ``geo.geolocate_with_covariance``, so oblique, far-off-centre fixes
  count less than near-nadir ones.
* **Merging** - a fix that fell outside the gate opens a tentative track; if
  that track later turns out to be statistically the same target as an
  existing one, the two are fused (information form), so a single early
  outlier cannot split one target into two half-strength tracks.
* **Confirmation** - a track is confirmed after ``min_hits`` fixes and once its
  1-sigma radius is below ``max_sigma_m``. Only confirmed tracks are offered
  to the delivery planner, which is what keeps one false positive from
  sending a payload into the debris field.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .geo import ne_offset, offset_geopoint
from .types import GeoFix, GeoPoint, TargetEstimate

CHI2_2DOF_999 = 13.82


class TargetTracker:
    def __init__(self, gate_m: float = 12.0, min_hits: int = 5, max_sigma_m: float = 3.0,
                 min_conf: float = 0.25, origin: Optional[GeoPoint] = None):
        self.gate_m, self.min_hits, self.max_sigma_m, self.min_conf = gate_m, min_hits, max_sigma_m, min_conf
        self.origin = origin
        self.tracks: list[TargetEstimate] = []
        self._x: dict[int, np.ndarray] = {}
        self._next_id = 1

    # ------------------------------------------------------------------ helpers
    def _to_ne(self, p: GeoPoint) -> np.ndarray:
        if self.origin is None:
            self.origin = GeoPoint(p.lat, p.lon, 0.0)
        return np.array(ne_offset(self.origin, p))

    def _publish(self, tr: TargetEstimate) -> None:
        x = self._x[tr.track_id]
        tr.point = offset_geopoint(self.origin, float(x[0]), float(x[1]))
        tr.confirmed = tr.hits >= self.min_hits and tr.sigma_m <= self.max_sigma_m

    # ------------------------------------------------------------------ main API
    def update(self, fix: GeoFix) -> Optional[TargetEstimate]:
        if fix.conf < self.min_conf:
            return None
        z = self._to_ne(fix.point)
        R = np.asarray(fix.cov_ne, dtype=float)

        best, best_d2 = None, np.inf
        for tr in self.tracks:
            if tr.cls != fix.cls or tr.delivered:
                continue
            x = self._x[tr.track_id]
            r = z - x
            if np.linalg.norm(r) > self.gate_m:
                continue
            d2 = float(r @ np.linalg.solve(tr.cov_ne + R, r))
            if d2 < CHI2_2DOF_999 and d2 < best_d2:
                best, best_d2 = tr, d2

        if best is None:
            tr = TargetEstimate(self._next_id, fix.cls, fix.point, R.copy(), hits=1, mean_conf=fix.conf)
            self._x[tr.track_id] = z
            self._next_id += 1
            self.tracks.append(tr)
            tr.history.append((fix.t, float(z[0]), float(z[1])))
            self._publish(tr)
            return tr

        x, P = self._x[best.track_id], best.cov_ne
        K = P @ np.linalg.inv(P + R)
        self._x[best.track_id] = x + K @ (z - x)
        best.cov_ne = (np.eye(2) - K) @ P
        best.mean_conf = (best.mean_conf * best.hits + fix.conf) / (best.hits + 1)
        best.hits += 1
        best.history.append((fix.t, float(z[0]), float(z[1])))
        best = self._merge_into(best)
        self._publish(best)
        return best

    def _merge_into(self, tr: TargetEstimate) -> TargetEstimate:
        """Fuse any other same-class track that is statistically the same target as ``tr``."""
        for other in list(self.tracks):
            if other is tr or other.cls != tr.cls or other.delivered:
                continue
            xa, xb = self._x[tr.track_id], self._x[other.track_id]
            d = xa - xb
            if np.linalg.norm(d) > self.gate_m or float(d @ np.linalg.solve(tr.cov_ne + other.cov_ne, d)) > CHI2_2DOF_999:
                continue
            keep, drop = (tr, other) if tr.hits >= other.hits else (other, tr)
            Ia, Ib = np.linalg.inv(keep.cov_ne), np.linalg.inv(drop.cov_ne)
            P = np.linalg.inv(Ia + Ib)
            self._x[keep.track_id] = P @ (Ia @ self._x[keep.track_id] + Ib @ self._x[drop.track_id])
            keep.cov_ne = P
            keep.mean_conf = (keep.mean_conf * keep.hits + drop.mean_conf * drop.hits) / (keep.hits + drop.hits)
            keep.hits += drop.hits
            keep.history += drop.history
            self.tracks.remove(drop)
            del self._x[drop.track_id]
            tr = keep
        return tr

    def confirmed(self, cls: Optional[str] = None) -> list[TargetEstimate]:
        out = [t for t in self.tracks if t.confirmed and not t.delivered and (cls is None or t.cls == cls)]
        return sorted(out, key=lambda t: t.hits * t.mean_conf, reverse=True)

    def best(self, cls: str) -> Optional[TargetEstimate]:
        c = self.confirmed(cls)
        return c[0] if c else None
