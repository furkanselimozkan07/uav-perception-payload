"""Object detectors behind a single interface.

* ``YoloDetector`` wraps Ultralytics YOLO (``.pt`` for development, a
  TensorRT ``.engine`` exported with ``scripts/export_tensorrt.py`` on Jetson).
* ``ColorDetector`` is a deterministic HSV detector used by the simulator and
  the unit tests, so the whole pipeline runs on any laptop without weights or
  a GPU.

Both return ``Detection`` objects with *mission* class names (``mannequin``,
``tent``), so the rest of the system does not care which model produced them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence

import numpy as np

from .types import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        ...

    def warmup(self, shape: Sequence[int] = (720, 1280, 3)) -> None:
        self.detect(np.zeros(shape, dtype=np.uint8))


class YoloDetector(Detector):
    """Ultralytics YOLO detector with class-name remapping and size filtering.

    ``class_map`` maps model class names to mission names; detections of
    unmapped classes are dropped (for example a COCO ``person`` -> ``mannequin``).
    """

    def __init__(self, weights: str, class_map: Optional[Mapping[str, str]] = None, conf: float = 0.35,
                 iou: float = 0.5, imgsz: int = 960, device: Optional[str] = None, min_area_px: float = 40.0,
                 half: bool = True):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install ultralytics  (or use ColorDetector for the simulator)") from e
        self.model = YOLO(weights)
        self.names = self.model.names
        self.class_map = dict(class_map) if class_map else {n: n for n in self.names.values()}
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.min_area_px = min_area_px
        self.half = half

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, half=self.half, verbose=False)[0]
        out: list[Detection] = []
        if res.boxes is None:
            return out
        for xyxy, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                              res.boxes.cls.cpu().numpy().astype(int)):
            name = self.class_map.get(self.names[int(k)])
            if name is None:
                continue
            d = Detection(name, float(c), tuple(float(v) for v in xyxy))  # type: ignore[arg-type]
            if d.area >= self.min_area_px:
                out.append(d)
        return out


class ColorDetector(Detector):
    """HSV-threshold detector: one or more colour ranges per class.

    ``ranges`` = {cls: [(h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)]}. Confidence is
    the fraction of the bounding box filled by the mask, a crude but monotonic
    proxy that lets the tracker weight clean blobs above ragged ones.
    """

    def __init__(self, ranges: Mapping[str, Sequence[Sequence[int]]], min_area_px: float = 30.0,
                 max_area_px: float = 1e6, kernel: int = 3):
        self.ranges = {k: (np.array(v[0], np.uint8), np.array(v[1], np.uint8)) for k, v in ranges.items()}
        self.min_area_px, self.max_area_px = min_area_px, max_area_px
        self.kernel = np.ones((kernel, kernel), np.uint8)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        import cv2

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        out: list[Detection] = []
        for cls, (lo, hi) in self.ranges.items():
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if not (self.min_area_px <= area <= self.max_area_px):
                    continue
                fill = area / float(w * h)
                out.append(Detection(cls, float(np.clip(fill, 0.05, 1.0)),
                                     (float(x), float(y), float(x + w), float(y + h))))
        return out


def draw_detections(frame_bgr: np.ndarray, dets: Sequence[Detection]) -> np.ndarray:
    """Annotated copy of the frame for the ground-station video stream."""
    import cv2

    img = frame_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, f"{d.cls} {d.conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return img
