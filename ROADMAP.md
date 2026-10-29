# Roadmap

## Vision

A small, fully testable perception → payload chain for SUAS-style UAV missions: every stage
(video sync, detection, geolocation, tracking, ballistics, mission rules, mapping) runs and is
verified on a laptop through the closed-loop simulator, and each stage is honest about its
uncertainty. Hardware-facing pieces (MAVLink, TensorRT, GStreamer) stay thin adapters behind
interfaces that the simulator can replace.

## Backlog

- [x] Georeference the risk-map mosaic: world file + JSON metadata export, and unit tests for `mapping.py`
- [ ] Unit tests for `detection.py` (ColorDetector class remapping, NMS/min-area, YOLO adapter with a stub model)
- [ ] Unit tests for `autopilot.py` (SimAutopilot kinematics/noise, MAVLink lap counting with a fake connection)
- [ ] Mission event log as JSONL + a `scripts/replay_log.py` that re-plots a flight from logs
- [ ] Coverage report for the lawnmower search: fraction of the search area seen vs. time
- [ ] Lawnmower planner: choose lane spacing from camera footprint, altitude and desired overlap
- [ ] Terrain-aware geolocation (DEM lookup hook instead of flat ground) with tests
- [ ] Fit payload drag coefficient from logged drop tests (least-squares CLI)
- [ ] Wind estimator from groundspeed/airspeed during the waypoint lap, feeding the ballistics planner
- [ ] Detector evaluation script: precision/recall and geolocation error against ground-truth CSV
- [ ] Mosaic export as GeoTIFF/tiles when `rasterio` is available (optional extra)
- [ ] Config validation for `configs/mission.yaml` with clear error messages
- [ ] Motion-blur and exposure model in the simulator renderer
- [ ] ROS 2 node wrappers (optional, separate `ros/` package)
- [ ] YOLO dataset export from the simulator (images + YOLO-format labels) and a training notebook
- [ ] Multi-target-class tracking stress test with clutter and false-positive bursts

## Changelog

- 2026-10-29 — Mosaic export: `GeoMosaic.save()` writes the image with an ESRI world file (EPSG:4326) and
  JSON metadata, so the risk map opens georeferenced in QGIS; first test suite for `mapping.py`
  (grid maths, placement, blending, spacing, phase-correlation recovery).
