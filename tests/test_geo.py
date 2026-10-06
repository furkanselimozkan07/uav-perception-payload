import numpy as np
import pytest

from uavpp.geo import (CameraModel, geolocate_pixel, geolocate_with_covariance, ground_to_pixel, ne_offset,
                       offset_geopoint, pixel_to_ned_offset)
from uavpp.types import Attitude, GeoPoint

CAM = CameraModel.from_hfov(960, 540, 70.0)
ORIGIN = GeoPoint(39.81, 30.53, 30.0)


def test_center_pixel_looks_straight_down():
    off = pixel_to_ned_offset(CAM, CAM.cx, CAM.cy, Attitude(0, 0, 0), 30.0)
    assert np.allclose(off, [0, 0], atol=1e-9)


def test_image_top_is_forward_and_right_is_right():
    # heading north: image top -> north, image right -> east
    top = pixel_to_ned_offset(CAM, CAM.cx, 0, Attitude(0, 0, 0), 30.0)
    right = pixel_to_ned_offset(CAM, CAM.width, CAM.cy, Attitude(0, 0, 0), 30.0)
    assert top[0] > 0 and abs(top[1]) < 1e-9
    assert right[1] > 0 and abs(right[0]) < 1e-9
    # heading east: image top -> east
    top_e = pixel_to_ned_offset(CAM, CAM.cx, 0, Attitude(0, 0, np.pi / 2), 30.0)
    assert top_e[1] > 0 and abs(top_e[0]) < 1e-9


def test_roll_right_shifts_nadir_point_to_the_right():
    # rolling right wing down tilts a body-fixed camera to look left of track
    off = pixel_to_ned_offset(CAM, CAM.cx, CAM.cy, Attitude(np.radians(10), 0, 0), 30.0)
    assert off[1] == pytest.approx(-30.0 * np.tan(np.radians(10)), rel=1e-6)


def test_stabilized_gimbal_ignores_roll_pitch():
    cam = CameraModel.from_hfov(960, 540, 70.0, stabilized=True)
    a = pixel_to_ned_offset(cam, 100, 400, Attitude(0.2, -0.1, 1.0), 30.0)
    b = pixel_to_ned_offset(cam, 100, 400, Attitude(0.0, 0.0, 1.0), 30.0)
    assert np.allclose(a, b)


@pytest.mark.parametrize("att", [Attitude(0, 0, 0), Attitude(0.1, -0.05, 2.0), Attitude(-0.2, 0.15, 5.5)])
def test_project_then_geolocate_roundtrip(att):
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, e = rng.uniform(-8, 8, 2)
        uv = ground_to_pixel(CAM, n, e, att, 30.0)
        if uv is None:
            continue
        off = pixel_to_ned_offset(CAM, uv[0], uv[1], att, 30.0)
        assert np.allclose(off, [n, e], atol=1e-6)


def test_ray_above_horizon_returns_none():
    cam = CameraModel.from_hfov(960, 540, 70.0, mount_rpy=(0.0, np.radians(95), 0.0))
    assert pixel_to_ned_offset(cam, cam.cx, cam.cy, Attitude(0, 0, 0), 30.0) is None


def test_wgs84_offset_roundtrip():
    p = offset_geopoint(ORIGIN, 123.4, -56.7)
    n, e = ne_offset(ORIGIN, p)
    assert n == pytest.approx(123.4, abs=1e-6) and e == pytest.approx(-56.7, abs=1e-6)
    # one degree of latitude is ~111 km
    assert ne_offset(GeoPoint(39.0, 30.0), GeoPoint(40.0, 30.0))[0] == pytest.approx(111_050, rel=2e-3)


def test_geolocate_pixel_matches_offset():
    p = geolocate_pixel(CAM, 700, 120, ORIGIN, Attitude(0.05, 0.02, 0.3))
    off = pixel_to_ned_offset(CAM, 700, 120, Attitude(0.05, 0.02, 0.3), ORIGIN.alt)
    assert np.allclose(ne_offset(ORIGIN, p), off, atol=1e-6)


def test_covariance_grows_off_centre_and_with_altitude():
    _, c_center = geolocate_with_covariance(CAM, CAM.cx, CAM.cy, ORIGIN, Attitude(0, 0, 0), sigma_pos_m=0.0)
    _, c_corner = geolocate_with_covariance(CAM, 10, 10, ORIGIN, Attitude(0, 0, 0), sigma_pos_m=0.0)
    _, c_high = geolocate_with_covariance(CAM, CAM.cx, CAM.cy, GeoPoint(ORIGIN.lat, ORIGIN.lon, 60.0),
                                          Attitude(0, 0, 0), sigma_pos_m=0.0)
    assert np.trace(c_corner) > np.trace(c_center)
    assert np.trace(c_high) > np.trace(c_center)
    assert np.all(np.linalg.eigvalsh(c_corner) > 0)
