import numpy as np
import pytest

from uavpp.geo import ne_offset, offset_geopoint
from uavpp.tracking import TargetTracker
from uavpp.types import Attitude, GeoFix, GeoPoint, Telemetry
from uavpp.video import FpsMeter, TelemetryBuffer

ORIGIN = GeoPoint(39.81, 30.53)


def _fix(cls, n, e, sigma=1.0, conf=0.8, t=0.0):
    return GeoFix(cls, conf, offset_geopoint(ORIGIN, n, e), np.eye(2) * sigma ** 2, t)


def test_repeated_fixes_converge_and_confirm():
    rng = np.random.default_rng(1)
    tr = TargetTracker(min_hits=5, max_sigma_m=1.0, origin=ORIGIN)
    for i in range(30):
        n, e = rng.normal([20.0, 35.0], 1.0)
        tr.update(_fix("tent", n, e, t=i))
    best = tr.best("tent")
    assert best is not None and best.confirmed and best.hits == 30
    n, e = ne_offset(ORIGIN, best.point)
    assert np.hypot(n - 20.0, e - 35.0) < 0.6
    assert best.sigma_m == pytest.approx(1.0 / np.sqrt(30), rel=0.05)


def test_far_or_other_class_fixes_open_new_tracks():
    tr = TargetTracker(gate_m=10.0, origin=ORIGIN)
    tr.update(_fix("tent", 0, 0))
    tr.update(_fix("tent", 50, 0))  # far away -> new track
    tr.update(_fix("mannequin", 0.5, 0))  # other class -> new track
    assert len(tr.tracks) == 3


def test_single_false_positive_is_never_confirmed():
    tr = TargetTracker(min_hits=5, origin=ORIGIN)
    tr.update(_fix("mannequin", 10, 10))
    assert tr.best("mannequin") is None


def test_low_confidence_fixes_are_ignored():
    tr = TargetTracker(min_conf=0.5, origin=ORIGIN)
    assert tr.update(_fix("tent", 0, 0, conf=0.2)) is None
    assert not tr.tracks


def _tel(t, yaw, lat=39.81):
    return Telemetry(t, GeoPoint(lat, 30.53, 30.0), Attitude(0.0, 0.0, yaw))


def test_telemetry_interpolation_and_yaw_wrap():
    buf = TelemetryBuffer()
    buf.push(_tel(0.0, np.radians(350)))
    buf.push(_tel(1.0, np.radians(10), lat=39.82))
    mid = buf.at(0.5, max_gap=2.0)
    assert np.degrees(mid.attitude.yaw) == pytest.approx(0.0, abs=1e-6) or \
        np.degrees(mid.attitude.yaw) == pytest.approx(360.0, abs=1e-6)
    assert mid.position.lat == pytest.approx(39.815)
    assert buf.at(-0.1) is None and buf.at(1.1) is None


def test_telemetry_gap_and_out_of_order():
    buf = TelemetryBuffer()
    buf.push(_tel(0.0, 0.0))
    buf.push(_tel(2.0, 0.0))
    buf.push(_tel(1.0, 0.0))  # out of order -> ignored
    assert len(buf) == 2
    assert buf.at(1.0, max_gap=0.5) is None


def test_fps_meter():
    m = FpsMeter(alpha=1.0)
    for i in range(5):
        m.tick(i * 0.1)
    assert m.fps == pytest.approx(10.0)
