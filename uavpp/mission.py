"""Search - Detect - Deliver mission state machine.

Encodes the SUAS 2026 rules that bite in practice:

* at least **one full waypoint lap** must be flown before any delivery
  (deliveries without it score zero);
* both payloads are carried at once; each goes to its own target class
  (water bottle -> mannequin, beacon -> tent);
* everything must finish inside the mission time window.

States::

    LAP --(lap done)--> SEARCH --(confirmed target with undelivered payload)--> APPROACH
    APPROACH --(release condition)--> RELEASE --> SEARCH (next payload) ... --> DONE

The machine is driven by ``step()`` once per perception cycle and is pure
logic: it reads the tracker and autopilot and issues autopilot commands, so
it can be unit-tested with ``SimAutopilot``.
"""
from __future__ import annotations

import copy
import enum
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .autopilot import Autopilot
from .ballistics import Payload, plan_release_point, should_release
from .geo import ne_offset, offset_geopoint
from .tracking import TargetTracker
from .types import GeoPoint, TargetEstimate

log = logging.getLogger("uavpp.mission")


class State(enum.Enum):
    LAP = "LAP"
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    RELEASE = "RELEASE"
    DONE = "DONE"


@dataclass
class PayloadBay:
    payload: Payload
    servo_channel: int
    pwm_closed: int = 1100
    pwm_open: int = 1900
    released: bool = False
    release_point: Optional[GeoPoint] = None
    release_t: Optional[float] = None
    target_id: Optional[int] = None
    target_snapshot: Optional[TargetEstimate] = None  # copy of the estimate the release was planned on


@dataclass
class MissionConfig:
    release_alt_m: float = 20.0
    release_tol_m: float = 1.5
    hover_speed_mps: float = 0.6
    mission_time_s: float = 30 * 60
    time_margin_s: float = 120.0
    min_laps_before_delivery: int = 1
    wind_ne: tuple[float, float] = (0.0, 0.0)
    # A moving release must be planned with the velocity at release; a release point planned for a hover
    # is only valid at hover speed. Off by default: hover releases are slower but far more repeatable.
    allow_moving_release: bool = False


@dataclass
class MissionEvent:
    t: float
    state: str
    msg: str


@dataclass
class SearchDetectDeliver:
    autopilot: Autopilot
    tracker: TargetTracker
    bays: list[PayloadBay]
    cfg: MissionConfig = field(default_factory=MissionConfig)
    state: State = State.LAP
    t0: Optional[float] = None
    events: list[MissionEvent] = field(default_factory=list)
    _active: Optional[PayloadBay] = None
    _active_target: Optional[TargetEstimate] = None

    def _emit(self, t: float, msg: str) -> None:
        self.events.append(MissionEvent(t, self.state.value, msg))
        log.info("[%7.1fs] %-8s %s", t, self.state.value, msg)

    def _goto_state(self, t: float, s: State, msg: str) -> None:
        self.state = s
        self._emit(t, msg)

    def pending_bays(self) -> list[PayloadBay]:
        return [b for b in self.bays if not b.released]

    # ------------------------------------------------------------------
    def step(self) -> State:
        tel = self.autopilot.telemetry()
        if tel is None:
            return self.state
        t = tel.t
        if self.t0 is None:
            self.t0 = t
        elapsed = t - self.t0

        if self.state != State.DONE and elapsed > self.cfg.mission_time_s - self.cfg.time_margin_s:
            self.autopilot.resume_mission()
            self._goto_state(t, State.DONE, "mission time margin reached, stopping deliveries")
            return self.state

        if self.state == State.LAP:
            if self.autopilot.laps_completed >= self.cfg.min_laps_before_delivery:
                self._goto_state(t, State.SEARCH, f"{self.autopilot.laps_completed} waypoint lap(s) complete")

        elif self.state == State.SEARCH:
            if not self.pending_bays():
                self.autopilot.resume_mission()
                self._goto_state(t, State.DONE, "all payloads delivered")
                return self.state
            for bay in self.pending_bays():
                tgt = self.tracker.best(bay.payload.target_cls)
                if tgt is None:
                    continue
                origin = self.tracker.origin
                tgt_ne = np.array(ne_offset(origin, tgt.point))
                rel_ne = plan_release_point(tgt_ne, bay.payload, self.cfg.release_alt_m, (0.0, 0.0, 0.0),
                                            self.cfg.wind_ne)
                bay.release_point = offset_geopoint(origin, float(rel_ne[0]), float(rel_ne[1]), self.cfg.release_alt_m)
                bay.target_id = tgt.track_id
                bay.target_snapshot = copy.deepcopy(tgt)
                self._active, self._active_target = bay, tgt
                self.autopilot.goto(bay.release_point)
                self._goto_state(t, State.APPROACH,
                                 f"{bay.payload.name} -> {tgt.cls} #{tgt.track_id} "
                                 f"({tgt.hits} fixes, 1σ {tgt.sigma_m:.1f} m)")
                break

        elif self.state == State.APPROACH:
            bay = self._active
            origin = self.tracker.origin
            here = np.array(ne_offset(origin, tel.position))
            if self.cfg.allow_moving_release and tel.groundspeed > self.cfg.hover_speed_mps:
                # re-plan with the current velocity and height so the throw is accounted for
                tgt_ne = np.array(ne_offset(origin, self._active_target.point))
                rel = plan_release_point(tgt_ne, bay.payload, tel.position.alt, tel.vel_ned, self.cfg.wind_ne)
                fire = should_release(here, rel, tel.groundspeed, tol_m=self.cfg.release_tol_m,
                                      max_speed_hover=self.cfg.hover_speed_mps, vel_ne=tel.vel_ned[:2])
            else:
                rel = np.array(ne_offset(origin, bay.release_point))
                fire = (tel.groundspeed <= self.cfg.hover_speed_mps
                        and float(np.linalg.norm(here - rel)) <= self.cfg.release_tol_m)
            if fire:
                self.autopilot.set_servo(bay.servo_channel, bay.pwm_open)
                bay.released, bay.release_t = True, t
                self._active_target.delivered = True
                self._goto_state(t, State.RELEASE, f"released {bay.payload.name} at "
                                 f"{tel.position.lat:.6f},{tel.position.lon:.6f} alt {tel.position.alt:.1f} m")

        elif self.state == State.RELEASE:
            self._active, self._active_target = None, None
            self.autopilot.resume_mission()
            self._goto_state(t, State.SEARCH, "back to search pattern")

        return self.state
