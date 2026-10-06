import numpy as np
import pytest

from uavpp.ballistics import G, Payload, plan_release_point, should_release, simulate_fall

BOTTLE = Payload("water_bottle", 0.255, 1.0, 0.0075, "mannequin")


def test_vacuum_fall_matches_closed_form():
    h, v = 30.0, 6.0
    r = simulate_fall(BOTTLE, h, (v, 0, 0), drag=False)
    t = np.sqrt(2 * h / G)
    assert r.t_fall == pytest.approx(t, rel=1e-4)
    assert r.d_north == pytest.approx(v * t, rel=1e-4)
    assert r.d_east == pytest.approx(0.0, abs=1e-9)


def test_drag_slows_the_fall_and_shortens_the_throw():
    vac = simulate_fall(BOTTLE, 30.0, (6, 0, 0), drag=False)
    air = simulate_fall(BOTTLE, 30.0, (6, 0, 0))
    assert air.t_fall > vac.t_fall
    assert air.d_north < vac.d_north
    assert air.impact_speed < BOTTLE.terminal_velocity + 1e-6 or air.impact_speed < vac.impact_speed


def test_wind_drifts_a_hover_release_downwind():
    r = simulate_fall(BOTTLE, 30.0, (0, 0, 0), wind_ne=(0.0, 4.0))
    assert r.d_east > 0.5 and abs(r.d_north) < 1e-6


def test_planned_release_point_lands_on_target():
    target = np.array([40.0, -12.0])
    vel, wind = (5.0, 1.0, 0.0), (-2.0, 3.0)
    rel = plan_release_point(target, BOTTLE, 25.0, vel, wind)
    r = simulate_fall(BOTTLE, 25.0, vel, wind)
    assert np.allclose(rel + [r.d_north, r.d_east], target, atol=1e-9)


def test_hover_release_trigger():
    assert should_release(np.array([0.5, 0.5]), np.array([0.0, 0.0]), groundspeed=0.2)
    assert not should_release(np.array([3.0, 0.0]), np.array([0.0, 0.0]), groundspeed=0.2)


def test_moving_release_fires_when_point_reached_with_latency_lead():
    vel = (10.0, 0.0)
    rel = np.array([0.0, 0.0])
    # 2 m before the point at 10 m/s with 0.15 s latency -> look-ahead 1.5 m, not yet
    assert not should_release(np.array([-2.0, 0.0]), rel, 10.0, servo_latency_s=0.15, vel_ne=vel)
    # 1 m before: look-ahead passes the point -> fire
    assert should_release(np.array([-1.0, 0.0]), rel, 10.0, servo_latency_s=0.15, vel_ne=vel)
    # passing 5 m to the side never fires
    assert not should_release(np.array([0.0, 5.0]), rel, 10.0, vel_ne=vel)
