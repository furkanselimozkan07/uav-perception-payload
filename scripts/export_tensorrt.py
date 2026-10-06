#!/usr/bin/env python3
"""Export a trained YOLO model to a TensorRT engine for Jetson and benchmark it.

Run this *on the Jetson* (engines are specific to the GPU and TensorRT version):

    python scripts/export_tensorrt.py --weights runs/detect/train/weights/best.pt --imgsz 960 --half
    python scripts/export_tensorrt.py --weights models/sdd_yolo11n.engine --bench-only

FP16 is the usual choice on Jetson; INT8 needs a calibration set
(``--int8 --data data.yaml``) and should be validated against the FP16 mAP
before it is trusted.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def bench(weights: str, imgsz: int, n: int = 200) -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    frame = (np.random.default_rng(0).integers(0, 255, (540, 960, 3))).astype(np.uint8)
    for _ in range(10):
        model.predict(frame, imgsz=imgsz, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(frame, imgsz=imgsz, verbose=False)
    dt = (time.perf_counter() - t0) / n
    print(f"{weights}: {dt * 1000:.1f} ms/frame  ({1 / dt:.1f} FPS) at imgsz={imgsz}, end-to-end incl. pre/post")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--data", default=None, help="dataset yaml for INT8 calibration")
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()

    if not args.bench_only:
        from ultralytics import YOLO

        path = YOLO(args.weights).export(format="engine", imgsz=args.imgsz, half=args.half, int8=args.int8,
                                         data=args.data, workspace=2, simplify=True)
        print("exported:", path)
        bench(str(path), args.imgsz)
    else:
        bench(args.weights, args.imgsz)


if __name__ == "__main__":
    main()
