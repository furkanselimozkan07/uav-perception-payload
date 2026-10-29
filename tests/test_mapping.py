import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from uavpp.geo import CameraModel, ne_offset, offset_geopoint  # noqa: E402
from uavpp.mapping import GeoMosaic, MosaicGrid  # noqa: E402
from uavpp.types import Attitude, FramePacket, GeoPoint, Telemetry  # noqa: E402

ORIGIN = GeoPoint(39.81, 32.77, 0.0)
CAM = CameraModel.from_hfov(320, 240, 70.0)
LEVEL = Attitude(0.0, 0.0, 0.0)


def _pkt(n, e, alt=20.0, color=(40, 120, 200), att=LEVEL):
    frame = np.full((CAM.height, CAM.width, 3), color, np.uint8)
    tel = Telemetry(0.0, offset_geopoint(ORIGIN, n, e, alt), att)
    return FramePacket(frame, 0.0, tel)


def _grid(res=0.2):
    return MosaicGrid(ORIGIN, 0.0, 40.0, 0.0, 40.0, res_m=res)


def test_grid_pixel_roundtrip():
    g = _grid()
    for n, e in [(0.0, 0.0), (12.3, 30.1), (40.0, 40.0)]:
        x, y = g.ne_to_px(n, e)
        assert g.px_to_ne(x, y) == pytest.approx((n, e))
    assert g.ne_to_px(40.0, 0.0) == (0.0, 0.0)  # north-west corner is the image origin


def test_nadir_frame_lands_under_the_aircraft():
    m = GeoMosaic(CAM, _grid(), refine=False)
    assert m.add(_pkt(20.0, 20.0))
    img = m.render()
    x, y = m.grid.ne_to_px(20.0, 20.0)
    assert tuple(img[int(y), int(x)]) == (40, 120, 200)
    assert tuple(img[0, 0]) == (0, 0, 0)  # far corner untouched
    # footprint half-width at 20 m AGL, hfov 70 deg: 20*tan(35deg) ~ 14 m -> about a third of the 40x40 m area
    assert 0.25 < m.coverage() < 0.45


def test_keyframes_respect_min_spacing_unless_forced():
    m = GeoMosaic(CAM, _grid(), min_spacing_m=8.0, refine=False)
    assert m.add(_pkt(20.0, 20.0))
    assert not m.add(_pkt(22.0, 20.0))
    assert m.add(_pkt(22.0, 20.0), force=True)
    assert m.add(_pkt(31.0, 20.0))
    assert m.n_frames == 3


def test_frame_without_telemetry_or_above_horizon_is_skipped():
    m = GeoMosaic(CAM, _grid(), refine=False)
    assert not m.add(FramePacket(np.zeros((240, 320, 3), np.uint8), 0.0, None))
    assert not m.add(_pkt(20.0, 20.0, att=Attitude(0.0, np.radians(80), 0.0)))  # camera sees the sky
    assert m.n_frames == 0 and m.coverage() == 0.0


def test_overlap_is_blended_not_overwritten():
    m = GeoMosaic(CAM, _grid(), min_spacing_m=1.0, refine=False)
    m.add(_pkt(20.0, 14.0, color=(0, 0, 0)))
    m.add(_pkt(20.0, 26.0, color=(200, 200, 200)))
    img = m.render()
    x, y = m.grid.ne_to_px(20.0, 20.0)  # in the overlap
    assert 40 < int(img[int(y), int(x), 0]) < 160


def test_world_file_matches_grid():
    g = _grid(res=0.5)
    a, d, b, e, c, f = (float(v) for v in g.world_file().split())
    assert (d, b) == (0.0, 0.0) and e < 0 < a
    ul = g.px_to_geopoint(0.5, 0.5)
    assert (c, f) == pytest.approx((ul.lon, ul.lat))
    # one pixel east / south is res_m metres on the ground
    east = GeoPoint(f, c + a)
    south = GeoPoint(f + e, c)
    assert ne_offset(ul, east)[1] == pytest.approx(0.5, abs=1e-3)
    assert ne_offset(ul, south)[0] == pytest.approx(-0.5, abs=1e-3)


def test_save_writes_image_world_file_and_metadata(tmp_path):
    m = GeoMosaic(CAM, _grid(), refine=False)
    m.add(_pkt(20.0, 20.0))
    paths = m.save(tmp_path / "out" / "map.png", extra={"mission": "unit-test"})
    assert [p.name for p in paths] == ["map.png", "map.pgw", "map.json"]
    assert all(p.exists() for p in paths)
    assert cv2.imread(str(paths[0])).shape == (200, 200, 3)
    meta = json.loads(paths[2].read_text())
    assert meta["size_px"] == {"width": 200, "height": 200}
    assert meta["frames"] == 1 and meta["mission"] == "unit-test" and meta["crs"] == "EPSG:4326"
    nw, se = meta["corners"]["nw"], meta["corners"]["se"]
    assert nw["lat"] > se["lat"] and nw["lon"] < se["lon"]
    assert (nw["lat"], nw["lon"]) == pytest.approx((offset_geopoint(ORIGIN, 40.0, 0.0).lat, offset_geopoint(ORIGIN, 40.0, 0.0).lon))


def test_save_uses_jpeg_world_file_extension(tmp_path):
    m = GeoMosaic(CAM, _grid(), refine=False)
    m.add(_pkt(20.0, 20.0))
    assert [p.suffix for p in m.save(tmp_path / "map.jpg")] == [".jpg", ".jgw", ".json"]


def test_registration_recovers_a_pose_error():
    """A textured scene seen with a 1.5 m pose offset is pulled back by phase correlation."""
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8), (0, 0), 2)
    g = MosaicGrid(ORIGIN, 0.0, 40.0, 0.0, 40.0, res_m=0.1)

    def view(n, e):
        # texture drawn in canvas px; sample the camera footprint exactly as the mosaic would see it
        H = GeoMosaic(CAM, g, refine=False)._footprint_homography(_pkt(n, e))
        return cv2.warpPerspective(tex, np.linalg.inv(H), (CAM.width, CAM.height))

    m = GeoMosaic(CAM, g, min_spacing_m=1.0, max_shift_m=3.0)
    m.add(FramePacket(view(20.0, 14.0), 0.0, _pkt(20.0, 14.0).telemetry))
    # second frame is reported 1.5 m east of where it truly was
    truth = view(20.0, 22.0)
    m.add(FramePacket(truth, 0.0, _pkt(20.0, 23.5).telemetry))
    assert m.corrections_m, "registration should have fired"
    assert m.corrections_m[-1] == pytest.approx(1.5, abs=0.4)
