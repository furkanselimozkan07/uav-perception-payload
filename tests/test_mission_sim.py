import numpy as np

from uavpp.autopilot import SimAutopilot
from uavpp.ballistics import Payload
from uavpp.detection import ColorDetector
from uavpp.geo import CameraModel, geolocate_with_covariance, ne_offset, offset_geopoint
from uavpp.mapping import MosaicGrid
from uavpp.mission import MissionConfig, PayloadBay, SearchDetectDeliver, State
from uavpp.sim import COLOR_RANGES, GroundScene
from uavpp.tracking import TargetTracker
from uavpp.types import GeoFix, GeoPoint

ORIGIN = GeoPoint(39.81, 30.53)
BOTTLE = Payload("water_bottle", 0.255, 1.0, 0.0075, "mannequin")


def _feed_confirmed(tracker, cls, n, e, k=8):
    for i in range(k):
        tracker.update(GeoFix(cls, 0.9, offset_geopoint(ORIGIN, n, e), np.eye(2) * 0.25, float(i)))


def test_no_delivery_before_a_full_waypoint_lap():
    lap = [(0, 0), (60, 0), (60, 60), (0, 60), (0, 0)]
    ap = SimAutopilot(ORIGIN, lap, alt_agl=30.0, att_noise_rad=0.0, pos_noise_m=0.0)
    tracker = TargetTracker(origin=ORIGIN)
    _feed_confirmed(tracker, "mannequin", 10.0, 10.0)  # target known from the very start
    m = SearchDetectDeliver(ap, tracker, [PayloadBay(BOTTLE, 9)], MissionConfig(release_alt_m=30.0))
    released_during_lap = False
    for _ in range(int(200 / 0.05)):
        ap.step(0.05)
        m.step()
        if ap.laps_completed == 0 and ap.servo_log:
            released_during_lap = True
        if m.state == State.DONE:
            break
    assert not released_during_lap
    assert m.state == State.DONE and ap.servo_log and ap.servo_log[0][1] == 9
    # the release happened inside the tolerance of the planned release point
    rel = np.array(ne_offset(ORIGIN, m.bays[0].release_point))
    assert np.linalg.norm(ap.p - rel) < 2.0


def test_payload_goes_to_its_own_target_class():
    lap = [(0, 0), (5, 0), (0, 0)]
    ap = SimAutopilot(ORIGIN, lap, alt_agl=30.0, att_noise_rad=0.0, pos_noise_m=0.0)
    tracker = TargetTracker(origin=ORIGIN)
    _feed_confirmed(tracker, "tent", 30.0, 0.0)  # only the tent is known
    m = SearchDetectDeliver(ap, tracker, [PayloadBay(BOTTLE, 9)], MissionConfig(release_alt_m=30.0))
    for _ in range(int(60 / 0.05)):
        ap.step(0.05)
        m.step()
    assert not ap.servo_log  # bottle is for the mannequin, never dropped on the tent


def test_mission_time_margin_stops_deliveries():
    ap = SimAutopilot(ORIGIN, [(0, 0), (1, 0), (0, 0)], alt_agl=30.0)
    m = SearchDetectDeliver(ap, TargetTracker(origin=ORIGIN), [PayloadBay(BOTTLE, 9)],
                            MissionConfig(mission_time_s=10.0, time_margin_s=5.0))
    for _ in range(200):
        ap.step(0.05)
        m.step()
    assert m.state == State.DONE


def test_rendered_target_is_detected_and_geolocated():
    rng = np.random.default_rng(3)
    grid = MosaicGrid(ORIGIN, 0, 60, 0, 60, res_m=0.05)
    scene = GroundScene(grid, rng, n_debris=10, n_bushes=5)
    tgt = scene.place_target("tent", (30.0, 30.0))
    cam = CameraModel.from_hfov(960, 540, 70.0)
    pos = offset_geopoint(ORIGIN, 28.0, 31.0, 30.0)
    from uavpp.types import Attitude

    att = Attitude(0.03, -0.02, 0.4)
    img = scene.render(cam, pos, att, noise=0)
    dets = [d for d in ColorDetector(COLOR_RANGES).detect(img) if d.cls == "tent"]
    assert len(dets) == 1
    p, _ = geolocate_with_covariance(cam, *dets[0].center, pos, att)
    n, e = ne_offset(ORIGIN, p)
    assert np.hypot(n - tgt.north, e - tgt.east) < 0.3  # exact pose -> only pixel quantisation error


def test_hover_planned_release_never_fires_while_passing_over_the_point():
    # Regression: a release point planned for a hover was once triggered while the aircraft flew
    # through it at cruise speed, throwing the payload ~16 m long in simulation.
    lap = [(0, 0), (5, 0), (0, 0)]
    ap = SimAutopilot(ORIGIN, lap, search_ne=[(0, 0), (80, 0)], alt_agl=30.0, att_noise_rad=0.0, pos_noise_m=0.0)
    speeds = []
    orig = ap.set_servo
    ap.set_servo = lambda ch, pwm: (speeds.append(float(np.hypot(*ap.v))), orig(ch, pwm))
    tracker = TargetTracker(origin=ORIGIN)
    _feed_confirmed(tracker, "mannequin", 40.0, 0.0)  # on the search line, so the aircraft flies over it
    cfg = MissionConfig(release_alt_m=30.0, hover_speed_mps=0.6)
    m = SearchDetectDeliver(ap, tracker, [PayloadBay(BOTTLE, 9)], cfg)
    for _ in range(int(120 / 0.05)):
        ap.step(0.05)
        m.step()
        if m.state == State.DONE:
            break
    assert speeds and max(speeds) <= 0.6 + 1e-6
