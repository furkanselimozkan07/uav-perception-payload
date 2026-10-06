#!/usr/bin/env python3
"""Closed-loop Search-Detect-Deliver simulation.

Flies the waypoint lap and a lawnmower search over a synthetic field,
renders camera frames from the *true* pose, runs detection -> geolocation
(with *noisy, biased* telemetry interpolated to frame time) -> tracking ->
release planning, then integrates each released payload's fall in the *true*
wind and scores it with the SUAS 2026 delivery rubric (within 50 ft of the
correct target).

    python scripts/run_sim.py                 # one run, writes docs/ figures
    python scripts/run_sim.py --runs 20       # Monte Carlo over seeds
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uavpp.autopilot import SimAutopilot  # noqa: E402
from uavpp.ballistics import Payload, simulate_fall  # noqa: E402
from uavpp.detection import ColorDetector, draw_detections  # noqa: E402
from uavpp.geo import CameraModel, geolocate_with_covariance, ne_offset  # noqa: E402
from uavpp.mapping import GeoMosaic, MosaicGrid  # noqa: E402
from uavpp.mission import MissionConfig, PayloadBay, SearchDetectDeliver, State  # noqa: E402
from uavpp.sim import COLOR_RANGES, GroundScene, lawnmower  # noqa: E402
from uavpp.tracking import TargetTracker  # noqa: E402
from uavpp.types import FramePacket, GeoFix, GeoPoint  # noqa: E402
from uavpp.video import TelemetryBuffer  # noqa: E402

FT50 = 50 * 0.3048  # SUAS delivery radius


def plot_overview(path, traj, lap, truth, tracker, origin, bays, impacts) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    fig, ax = plt.subplots(figsize=(8, 6.2), dpi=120)
    tr = np.array(traj)
    ax.plot(tr[:, 1], tr[:, 0], lw=0.8, color="0.55", label="flown path")
    lp = np.array(lap)
    ax.plot(lp[:, 1], lp[:, 0], "--", lw=1.0, color="0.3", label="waypoint lap")
    ax.add_patch(plt.Rectangle((0, 0), 150, 110, fill=False, ec="tab:green", lw=1.2, label="search area"))
    colors = {"mannequin": "tab:orange", "tent": "tab:blue"}
    for cls, t in truth.items():
        ax.plot(t.east, t.north, "x", color=colors[cls], ms=10, mew=2, label=f"{cls} (truth)")
        ax.add_patch(Circle((t.east, t.north), FT50, fill=False, ls=":", ec=colors[cls], lw=1))
    for trk in tracker.tracks:
        if not trk.confirmed and not trk.delivered:
            continue
        n, e = ne_offset(origin, trk.point)
        w, v = np.linalg.eigh(trk.cov_ne)
        ang = np.degrees(np.arctan2(v[0, 1], v[1, 1]))
        ax.add_patch(Ellipse((e, n), 6 * np.sqrt(w[1]), 6 * np.sqrt(w[0]), angle=ang, fill=False,
                             ec=colors.get(trk.cls, "k"), lw=1.5))
        ax.plot(e, n, "o", color=colors.get(trk.cls, "k"), ms=4)
    for bay in bays:
        if bay.payload.name in impacts:
            rn, re_ = ne_offset(origin, bay.release_point)
            imp = impacts[bay.payload.name]
            c = colors[bay.payload.target_cls]
            ax.plot(re_, rn, "^", color=c, ms=7)
            ax.plot(imp["east"], imp["north"], "*", color=c, ms=12, mec="k")
            ax.annotate("", (imp["east"], imp["north"]), (re_, rn), arrowprops=dict(arrowstyle="->", color=c))
    ax.plot([], [], "o", color="k", ms=4, label="fused estimate (3σ ellipse)")
    ax.plot([], [], "^", color="k", label="release point")
    ax.plot([], [], "*", color="k", ms=10, label="payload impact (true wind)")
    ax.set_xlabel("east [m]"); ax.set_ylabel("north [m]"); ax.set_aspect("equal")
    ax.set_title("Simulated Search-Detect-Deliver run (dotted circles: 50 ft scoring radius)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def run_episode(cfg: dict, seed: int, save_dir: pathlib.Path | None = None, verbose: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    origin = GeoPoint(39.8100, 30.5300, 0.0)  # arbitrary reference point
    search = MosaicGrid(origin, north_min=0.0, north_max=110.0, east_min=0.0, east_max=150.0, res_m=0.05)
    scene = GroundScene(search, rng)
    truth = {t.cls: t for t in (scene.place_target("mannequin"), scene.place_target("tent"))}

    cam = CameraModel.from_config(cfg["camera"])
    alt = float(cfg["mission"]["release_alt_m"])
    lap = [(-20.0, -20.0), (-20.0, 170.0), (130.0, 170.0), (130.0, -20.0), (-20.0, -20.0)]
    search_pts = lawnmower(5.0, 105.0, 10.0, 140.0, spacing=26.0)

    wind_true = tuple(rng.uniform(-4, 4, 2))
    wind_est = tuple(np.array(wind_true) + rng.normal(0, 0.7, 2))  # autopilot wind estimate error
    att_bias = tuple(rng.normal(0, np.radians(0.7), 3))  # residual IMU / mount misalignment
    ap = SimAutopilot(origin, lap, search_pts, alt_agl=alt, speed=8.0, rng=rng,
                      att_noise_rad=np.radians(0.4), att_bias_rad=att_bias, pos_noise_m=0.6)

    buf = TelemetryBuffer()
    ap.add_listener(buf.push)
    det = ColorDetector(COLOR_RANGES, min_area_px=25)
    g = cfg["geolocation"]
    tr_cfg = cfg["tracker"]
    tracker = TargetTracker(gate_m=tr_cfg["gate_m"], min_hits=tr_cfg["min_hits"], max_sigma_m=tr_cfg["max_sigma_m"],
                            min_conf=tr_cfg["min_conf"], origin=origin)
    bays = [PayloadBay(Payload.from_config(name, p), int(p["servo_channel"])) for name, p in cfg["payloads"].items()]
    mcfg = MissionConfig(release_alt_m=alt, release_tol_m=cfg["mission"]["release_tol_m"],
                         hover_speed_mps=cfg["mission"]["hover_speed_mps"],
                         mission_time_s=cfg["mission"]["mission_time_s"],
                         time_margin_s=cfg["mission"]["time_margin_s"], wind_ne=wind_est)
    mission = SearchDetectDeliver(ap, tracker, bays, mcfg)
    mgrid = MosaicGrid(origin, -5.0, 115.0, -5.0, 155.0, res_m=0.10)
    mosaic = GeoMosaic(cam, mgrid, min_spacing_m=7.0)
    mosaic_raw = GeoMosaic(cam, mgrid, min_spacing_m=7.0, refine=False) if save_dir is not None else None

    dt, cam_period, t_next_frame = 0.05, 0.25, 0.0
    latency = float(cfg["camera"].get("latency_s", 0.0))
    fix_err = {"mannequin": [], "tent": []}
    pending_frames: list[tuple[float, np.ndarray]] = []
    sample_frame, best_score = None, 0.0
    impacts: dict[str, dict] = {}
    wall0 = time.time()

    traj = []
    # fly until deliveries are done AND the search pattern has been covered once (for the risk map)
    while not (mission.state == State.DONE and ap.search_passes >= 1) and ap.t < 1500:
        ap.step(dt)
        if int(round(ap.t / dt)) % 10 == 0:
            traj.append((float(ap.p[0]), float(ap.p[1])))
        if ap.t >= t_next_frame:
            t_next_frame += cam_period
            # render from TRUE pose; the pipeline receives it `latency` later, stamped with capture time
            frame = scene.render(cam, ap.true_position, ap.true_attitude)
            pending_frames.append((ap.t, frame))
        while pending_frames and pending_frames[0][0] + latency <= ap.t:
            t_cap, frame = pending_frames.pop(0)
            tel = buf.at(t_cap)
            if tel is None:
                continue
            dets = det.detect(frame)
            for d in dets:
                u, v = d.center
                res = geolocate_with_covariance(cam, u, v, tel.position, tel.attitude, sigma_px=g["sigma_px"],
                                                sigma_att_rad=np.radians(g["sigma_att_deg"]),
                                                sigma_alt_m=g["sigma_alt_m"], sigma_pos_m=g["sigma_pos_m"])
                if res is None:
                    continue
                p, cov = res
                tracker.update(GeoFix(d.cls, d.conf, p, cov, t_cap))
                if d.cls in truth:
                    n, e = ne_offset(origin, p)
                    fix_err[d.cls].append(float(np.hypot(n - truth[d.cls].north, e - truth[d.cls].east)))
            # keep the most informative frame for the README: most targets, fully inside, scene fully in view
            if dets and ap.laps_completed >= 1:
                inside = all(15 < x.xyxy[0] and x.xyxy[2] < cam.width - 15 and 15 < x.xyxy[1]
                             and x.xyxy[3] < cam.height - 15 for x in dets)
                score = len({x.cls for x in dets}) + 0.001 * sum(x.area for x in dets)
                if inside and score > best_score:
                    best_score, sample_frame = score, draw_detections(frame, dets)
            if ap.laps_completed >= 1:
                mosaic.add(FramePacket(frame, t_cap, tel))
                if mosaic_raw is not None:
                    mosaic_raw.add(FramePacket(frame, t_cap, tel))
        mission.step()
        for bay in bays:
            if bay.released and bay.payload.name not in impacts:
                p_true = ne_offset(origin, ap.true_position)
                fall = simulate_fall(bay.payload, alt, (float(ap.v[0]), float(ap.v[1]), 0.0), wind_true)
                impacts[bay.payload.name] = {"north": p_true[0] + fall.d_north, "east": p_true[1] + fall.d_east,
                                             "t_fall": fall.t_fall, "v": fall.impact_speed}

    # ---- score deliveries (impacts were computed from the TRUE state at the moment of release)
    deliveries = []
    for bay in bays:
        if bay.payload.name not in impacts:
            deliveries.append({"payload": bay.payload.name, "released": False})
            continue
        imp = impacts[bay.payload.name]
        tgt = truth[bay.payload.target_cls]
        trk = bay.target_snapshot  # the estimate the release was planned on
        est_n, est_e = ne_offset(origin, trk.point)
        miss = float(np.hypot(imp["north"] - tgt.north, imp["east"] - tgt.east))
        deliveries.append({
            "payload": bay.payload.name, "released": True, "t_release_s": round(bay.release_t, 1),
            "target_cls": tgt.cls, "correct_target": trk.cls == tgt.cls,
            "target_estimate_err_m": round(float(np.hypot(est_n - tgt.north, est_e - tgt.east)), 2),
            "track_hits": trk.hits, "track_sigma_m": round(trk.sigma_m, 2),
            "impact_miss_m": round(miss, 2), "within_50ft": miss <= FT50,
            "fall_time_s": round(imp["t_fall"], 2), "impact_speed_mps": round(imp["v"], 1),
        })
    metrics = {
        "seed": seed, "sim_time_s": round(ap.t, 1), "wall_s": round(time.time() - wall0, 1),
        "wind_true_ne": [round(float(w), 2) for w in wind_true],
        "wind_est_ne": [round(float(w), 2) for w in wind_est],
        "att_bias_deg": [round(float(np.degrees(b)), 2) for b in att_bias],
        "single_fix_err_median_m": {k: (round(float(np.median(v)), 2) if v else None) for k, v in fix_err.items()},
        "n_fixes": {k: len(v) for k, v in fix_err.items()},
        "mosaic_frames": mosaic.n_frames, "mosaic_coverage": round(mosaic.coverage(), 3),
        "mosaic_median_correction_m": round(float(np.median(mosaic.corrections_m)), 2) if mosaic.corrections_m else None,
        "deliveries": deliveries, "events": [(round(e.t, 1), e.state, e.msg) for e in mission.events],
    }
    if save_dir is not None:
        import cv2

        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / "risk_map_mosaic.jpg"), mosaic.render(), [cv2.IMWRITE_JPEG_QUALITY, 85])
        # pose-only vs pose + phase-correlation refinement, cropped around the targets
        n_c = 0.5 * (truth["mannequin"].north + truth["tent"].north)
        e_c = 0.5 * (truth["mannequin"].east + truth["tent"].east)
        crops = []
        for m, label in ((mosaic_raw, "pose only"), (mosaic, "pose + image registration")):
            x, y = mgrid.ne_to_px(n_c, e_c)
            img = m.render()
            y0, x0 = int(max(0, y - 180)), int(max(0, x - 260))
            c = cv2.resize(img[y0:y0 + 360, x0:x0 + 520].copy(), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(c, (0, 0), (len(label) * 12 + 16, 34), (0, 0, 0), -1)
            cv2.putText(c, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            crops.append(c)
        cv2.imwrite(str(save_dir / "mosaic_refinement.jpg"), np.hstack([crops[0], np.full((crops[0].shape[0], 6, 3), 255, np.uint8), crops[1]]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        plot_overview(save_dir / "sim_overview.png", traj, lap, truth, tracker, origin, bays, impacts)
        if sample_frame is not None:
            cv2.imwrite(str(save_dir / "detections_frame.jpg"), sample_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "mission.yaml"))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "docs"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    results = []
    for i in range(args.runs):
        r = run_episode(cfg, args.seed + i, save_dir=pathlib.Path(args.out) if i == 0 else None)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("events",)}, default=float), flush=True)
    if args.runs == 1:
        for e in results[0]["events"]:
            print(f"  [{e[0]:7.1f}s] {e[1]:<8} {e[2]}")
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sim_results.json").write_text(json.dumps({"summary": summary, "runs": results}, indent=1, default=float))


def summarize(results: list[dict]) -> dict:
    d = [x for r in results for x in r["deliveries"]]
    rel = [x for x in d if x["released"]]
    miss = np.array([x["impact_miss_m"] for x in rel]) if rel else np.array([np.nan])
    est = np.array([x["target_estimate_err_m"] for x in rel]) if rel else np.array([np.nan])
    fix = [v for r in results for v in r["single_fix_err_median_m"].values() if v is not None]
    return {
        "runs": len(results),
        "deliveries_attempted": len(d),
        "deliveries_released": len(rel),
        "correct_target_rate": round(float(np.mean([x["correct_target"] for x in rel])), 3) if rel else None,
        "within_50ft_rate": round(float(np.mean([x["within_50ft"] for x in rel])), 3) if rel else None,
        "impact_miss_m": {"median": round(float(np.median(miss)), 2), "p90": round(float(np.percentile(miss, 90)), 2),
                          "max": round(float(np.max(miss)), 2)},
        "fused_target_error_m": {"median": round(float(np.median(est)), 2), "p90": round(float(np.percentile(est, 90)), 2)},
        "single_fix_error_median_m": round(float(np.median(fix)), 2) if fix else None,
        "mission_time_s_median": round(float(np.median([r["sim_time_s"] for r in results])), 1),
    }


if __name__ == "__main__":
    main()
