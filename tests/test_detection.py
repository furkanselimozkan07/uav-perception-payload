"""Tests for detection.py: ColorDetector on synthetic frames, YoloDetector with a stub model."""
import sys
import types

import numpy as np
import pytest

from uavpp.detection import ColorDetector, Detector, YoloDetector, draw_detections
from uavpp.types import Detection

RED = [(0, 150, 100), (10, 255, 255)]  # HSV
BLUE = [(100, 150, 100), (130, 255, 255)]


def _frame(h=120, w=160):
    return np.zeros((h, w, 3), np.uint8)


def _fill(img, x1, y1, x2, y2, bgr):
    img[y1:y2, x1:x2] = bgr


RED_BGR, BLUE_BGR = (0, 0, 255), (255, 0, 0)


# ------------------------------------------------------------------ ColorDetector
def test_color_detector_finds_each_class_with_exact_box():
    img = _frame()
    _fill(img, 20, 30, 50, 60, RED_BGR)
    _fill(img, 100, 70, 130, 100, BLUE_BGR)
    dets = ColorDetector({"mannequin": RED, "tent": BLUE}).detect(img)
    by_cls = {d.cls: d for d in dets}
    assert set(by_cls) == {"mannequin", "tent"}
    assert by_cls["mannequin"].xyxy == (20.0, 30.0, 50.0, 60.0)
    assert by_cls["tent"].xyxy == (100.0, 70.0, 130.0, 100.0)
    assert by_cls["mannequin"].conf == pytest.approx(1.0)


def test_color_detector_empty_frame_returns_nothing():
    assert ColorDetector({"mannequin": RED}).detect(_frame()) == []


def test_color_detector_min_and_max_area_filter():
    img = _frame()
    _fill(img, 5, 5, 11, 11, RED_BGR)      # 36 px
    _fill(img, 40, 40, 80, 80, RED_BGR)    # 1600 px
    small_ok = ColorDetector({"m": RED}, min_area_px=30, max_area_px=1e6).detect(img)
    assert len(small_ok) == 2
    only_big = ColorDetector({"m": RED}, min_area_px=100).detect(img)
    assert [d.xyxy for d in only_big] == [(40.0, 40.0, 80.0, 80.0)]
    only_small = ColorDetector({"m": RED}, min_area_px=30, max_area_px=500).detect(img)
    assert [d.xyxy for d in only_small] == [(5.0, 5.0, 11.0, 11.0)]


def test_color_detector_morphological_open_removes_speckle():
    img = _frame()
    img[10, 10] = RED_BGR  # single isolated pixel
    _fill(img, 50, 50, 70, 70, RED_BGR)
    dets = ColorDetector({"m": RED}, min_area_px=1).detect(img)
    assert len(dets) == 1 and dets[0].xyxy == (50.0, 50.0, 70.0, 70.0)


def test_color_detector_confidence_is_fill_fraction():
    img = _frame()
    _fill(img, 20, 20, 60, 60, RED_BGR)
    img[20:40, 40:60] = 0  # remove the top-right quadrant -> L-shape filling 3/4 of its bbox
    dets = ColorDetector({"m": RED}, min_area_px=10).detect(img)
    assert len(dets) == 1
    assert dets[0].xyxy == (20.0, 20.0, 60.0, 60.0)
    assert dets[0].conf == pytest.approx(0.75)


def test_color_detector_confidence_is_floored():
    # a thin diagonal-ish ragged blob can have a tiny fill ratio; conf never drops below 0.05
    img = _frame()
    for i in range(40):
        _fill(img, 10 + i, 10 + i, 13 + i, 13 + i, RED_BGR)
    dets = ColorDetector({"m": RED}, min_area_px=10).detect(img)
    assert dets and all(d.conf >= 0.05 for d in dets)


def test_warmup_runs_detect_on_blank_frame():
    calls = []

    class Spy(Detector):
        def detect(self, frame):
            calls.append(frame.shape)
            return []

    Spy().warmup((10, 20, 3))
    assert calls == [(10, 20, 3)]


# ------------------------------------------------------------------ YoloDetector (stub)
class _T:
    """Minimal torch.Tensor stand-in supporting .cpu().numpy()."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _StubModel:
    names = {0: "person", 1: "tent", 2: "car"}

    def __init__(self, weights):
        self.weights = weights
        self.kwargs = None

    def predict(self, frame, **kw):
        self.kwargs = kw
        boxes = types.SimpleNamespace(
            xyxy=_T([[0, 0, 20, 20], [10, 10, 12, 12], [5, 5, 50, 50], [30, 30, 60, 90]]),
            conf=_T([0.9, 0.8, 0.7, 0.6]),
            cls=_T([0.0, 0.0, 2.0, 1.0]),
        )
        return [types.SimpleNamespace(boxes=boxes)]


@pytest.fixture
def stub_ultralytics(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _StubModel
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def test_yolo_remaps_classes_and_drops_unmapped_and_tiny(stub_ultralytics):
    det = YoloDetector("w.pt", class_map={"person": "mannequin", "tent": "tent"}, min_area_px=40)
    out = det.detect(_frame())
    # car is unmapped; the 2x2 person is below min_area_px
    assert [(d.cls, d.conf) for d in out] == [("mannequin", pytest.approx(0.9)), ("tent", pytest.approx(0.6))]
    assert out[0].xyxy == (0.0, 0.0, 20.0, 20.0)


def test_yolo_default_class_map_is_identity(stub_ultralytics):
    det = YoloDetector("w.pt", min_area_px=0)
    assert {d.cls for d in det.detect(_frame())} == {"person", "car", "tent"}


def test_yolo_passes_inference_parameters(stub_ultralytics):
    det = YoloDetector("w.engine", conf=0.5, iou=0.4, imgsz=640, device="cuda:0", half=False)
    det.detect(_frame())
    assert det.model.weights == "w.engine"
    assert det.model.kwargs == dict(conf=0.5, iou=0.4, imgsz=640, device="cuda:0", half=False, verbose=False)


def test_yolo_no_boxes_returns_empty(stub_ultralytics, monkeypatch):
    det = YoloDetector("w.pt")
    monkeypatch.setattr(det.model, "predict", lambda *a, **k: [types.SimpleNamespace(boxes=None)])
    assert det.detect(_frame()) == []


def test_yolo_missing_dependency_gives_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # makes `import ultralytics` raise ImportError
    with pytest.raises(ImportError, match="ultralytics"):
        YoloDetector("w.pt")


# ------------------------------------------------------------------ draw_detections
def test_draw_detections_returns_annotated_copy():
    img = _frame()
    out = draw_detections(img, [Detection("tent", 0.77, (20.0, 30.0, 60.0, 70.0))])
    assert out is not img and out.shape == img.shape
    assert img.sum() == 0          # original untouched
    assert out.sum() > 0           # box + label drawn
    assert tuple(out[30, 40]) == (0, 255, 255)  # top edge of the rectangle


def test_draw_detections_no_dets_is_identical_copy():
    img = _frame()
    out = draw_detections(img, [])
    assert np.array_equal(out, img) and out is not img
