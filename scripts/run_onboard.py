#!/usr/bin/env python3
"""Onboard entry point (Jetson + ArduPilot).

Runs the same pipeline as the simulator against a real camera and a MAVLink
link. Safe by default: without ``--enable-delivery`` the program only
perceives, geolocates, tracks and logs - it never sends GUIDED targets or
servo commands. That is how it should be flown first (perception-only
flights), and how it is bench-tested against ArduPilot SITL.

    # perception only, CSI camera, SITL over UDP
    python scripts/run_onboard.py --mavlink udpin:0.0.0.0:14550 --source csi

    # full mission (after perception-only validation), stream annotated video to the GCS
    python scripts/run_onboard.py --enable-delivery --stream
"""
from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uavpp.autopilot import MavlinkAutopilot  # noqa: E402
from uavpp.ballistics import Payload  # noqa: E402
from uavpp.detection import ColorDetector, YoloDetector, draw_detections  # noqa: E402
from uavpp.geo import CameraModel, geolocate_with_covariance  # noqa: E402
from uavpp.mission import MissionConfig, PayloadBay, SearchDetectDeliver  # noqa: E402
from uavpp.sim import COLOR_RANGES  # noqa: E402
from uavpp.tracking import TargetTracker  # noqa: E402
from uavpp.types import GeoFix  # noqa: E402
from uavpp.video import FpsMeter, LatestFrameSource, TelemetryBuffer, jetson_csi_pipeline, udp_h264_sender_pipeline  # noqa: E402

log = logging.getLogger("uavpp.onboard")


def build_detector(cfg: dict):
    d = cfg["detector"]
    if d["type"] == "color":
        return ColorDetector(COLOR_RANGES)
    return YoloDetector(str(ROOT / d["weights"]), class_map=d.get("class_map"), conf=d["conf"], imgsz=d["imgsz"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "mission.yaml"))
    ap.add_argument("--source", default="csi", help="'csi', a device index, a file or an RTSP URL")
    ap.add_argument("--mavlink", default=None, help="override mavlink.url from the config")
    ap.add_argument("--enable-delivery", action="store_true", help="allow GUIDED targets and servo commands")
    ap.add_argument("--stream", action="store_true", help="send annotated H.264 video to the GCS over UDP")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = yaml.safe_load(open(args.config))
    cam = CameraModel.from_config(cfg["camera"])
    det = build_detector(cfg)
    det.warmup((cam.height, cam.width, 3))

    mcfg = cfg["mavlink"]
    autopilot = MavlinkAutopilot(args.mavlink or mcfg["url"], baud=mcfg["baud"], lap_last_seq=mcfg["lap_last_seq"])
    buf = TelemetryBuffer()
    autopilot.add_listener(buf.push)

    g, t = cfg["geolocation"], cfg["tracker"]
    tracker = TargetTracker(gate_m=t["gate_m"], min_hits=t["min_hits"], max_sigma_m=t["max_sigma_m"],
                            min_conf=t["min_conf"])
    bays = [PayloadBay(Payload.from_config(n, p), int(p["servo_channel"])) for n, p in cfg["payloads"].items()]
    m = cfg["mission"]
    mission = SearchDetectDeliver(autopilot, tracker, bays, MissionConfig(
        release_alt_m=m["release_alt_m"], release_tol_m=m["release_tol_m"], hover_speed_mps=m["hover_speed_mps"],
        mission_time_s=m["mission_time_s"], time_margin_s=m["time_margin_s"],
        min_laps_before_delivery=m["min_laps_before_delivery"], wind_ne=tuple(m["wind_ne"])))

    if args.source == "csi":
        src = LatestFrameSource(jetson_csi_pipeline(out_width=cam.width, out_height=cam.height), use_gstreamer=True,
                                latency_s=cfg["camera"].get("latency_s", 0.0))
    else:
        src = LatestFrameSource(int(args.source) if args.source.isdigit() else args.source,
                                latency_s=cfg["camera"].get("latency_s", 0.0))

    writer = None
    if args.stream:
        import cv2

        s = cfg["stream"]
        writer = cv2.VideoWriter(udp_h264_sender_pipeline(s["gcs_host"], s["gcs_port"], s["bitrate_kbps"]),
                                 cv2.CAP_GSTREAMER, 0, 15.0, (cam.width, cam.height))

    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    fixes_csv = csv.writer(open(log_dir / f"fixes_{int(time.time())}.csv", "w", newline=""))
    fixes_csv.writerow(["t", "cls", "conf", "lat", "lon", "sigma_n", "sigma_e", "track_id", "hits", "confirmed"])
    fps = FpsMeter()
    log.info("running (delivery %s)", "ENABLED" if args.enable_delivery else "disabled: perception only")

    try:
        while True:
            got = src.read(timeout=2.0)
            if got is None:
                log.warning("no frame")
                continue
            frame, t_cap = got
            tel = buf.at(t_cap)
            if tel is None:
                continue
            dets = det.detect(frame)
            for d in dets:
                res = geolocate_with_covariance(cam, *d.center, tel.position, tel.attitude, sigma_px=g["sigma_px"],
                                                sigma_att_rad=np.radians(g["sigma_att_deg"]),
                                                sigma_alt_m=g["sigma_alt_m"], sigma_pos_m=g["sigma_pos_m"])
                if res is None:
                    continue
                p, cov = res
                tr = tracker.update(GeoFix(d.cls, d.conf, p, cov, t_cap))
                fixes_csv.writerow([f"{t_cap:.3f}", d.cls, f"{d.conf:.2f}", f"{p.lat:.7f}", f"{p.lon:.7f}",
                                    f"{np.sqrt(cov[0, 0]):.2f}", f"{np.sqrt(cov[1, 1]):.2f}",
                                    tr.track_id if tr else "", tr.hits if tr else "", tr.confirmed if tr else ""])
            if args.enable_delivery:
                mission.step()
            if writer is not None:
                writer.write(draw_detections(frame, dets))
            fps.tick()
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        log.info("fps %.1f, captured %d, dropped %d", fps.fps, src.captured, src.dropped)
        for tr in tracker.confirmed():
            log.info("confirmed %s #%d at %.7f, %.7f (1σ %.2f m, %d fixes)", tr.cls, tr.track_id, tr.point.lat,
                     tr.point.lon, tr.sigma_m, tr.hits)


if __name__ == "__main__":
    main()
