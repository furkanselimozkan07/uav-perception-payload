"""uavpp - onboard perception and payload delivery for small UAVs.

Modules
-------
types      shared dataclasses (Telemetry, Detection, GeoFix, TargetEstimate)
geo        camera model, pixel -> ground geolocation with covariance, WGS-84 helpers
video      low-latency capture, frame/telemetry time sync, GStreamer pipelines
detection  YOLO (Ultralytics / TensorRT) and HSV detectors behind one interface
tracking   Kalman fusion of repeated geolocations into confirmed targets
ballistics payload free-fall with drag + wind, release-point planning
autopilot  ArduPilot MAVLink link and a kinematic simulator
mission    Search-Detect-Deliver state machine (SUAS 2026 rules)
mapping    georeferenced mosaic for the Risk Mapping task
sim        synthetic search area and exact camera renderer
"""

__version__ = "0.1.0"
