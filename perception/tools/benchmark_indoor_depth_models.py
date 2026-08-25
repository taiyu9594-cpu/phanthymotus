#!/usr/bin/env python3
"""Extract strict-ROI raw depth features from DA2 and YOLO TRT engines."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PERCEPTION_DIR = Path(__file__).resolve().parents[1]
if str(PERCEPTION_DIR) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_DIR))

from plugins.obstacle import (  # noqa: E402
    INDOOR_H,
    INDOOR_MEAN,
    INDOOR_ROI_COLS,
    INDOOR_ROI_ROWS,
    INDOOR_STD,
    INDOOR_W,
    _TrtEngine,
    _letterbox,
    _unwrap_depth,
)


OUTPUT_FIELDS = [
    "sequence", "source_index", "image_path", "gt_distance",
    "da2_min", "da2_p1", "da2_p5", "da2_p10", "da2_p20",
    "da2_median", "da2_inference_ms",
    "yolo_min", "yolo_p1", "yolo_p5", "yolo_p10", "yolo_p20",
    "yolo_median", "yolo_inference_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with sequence/source_index/image_path")
    parser.add_argument("--gt-csv", help="Optional CSV joined by sequence/source_index")
    parser.add_argument("--image-root", help="Base for relative image paths; defaults to manifest directory")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--da2-engine",
        default="/models/obstacle/depth_anything_v2_metric_hypersim_vits_int8.trt",
    )
    parser.add_argument(
        "--yolo-engine",
        default="/models/obstacle/yolo26n-depth_int8.trt",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def require_columns(path: Path, fieldnames: list[str] | None, required: set[str]) -> None:
    missing = required - set(fieldnames or [])
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")


def load_rows(manifest_path: Path, gt_path: Path | None, image_root: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(
            manifest_path, reader.fieldnames,
            {"sequence", "source_index", "image_path"},
        )
        rows = [dict(row) for row in reader]

    gt_by_key = None
    if gt_path is not None:
        with gt_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            require_columns(
                gt_path, reader.fieldnames,
                {"sequence", "source_index", "gt_distance"},
            )
            gt_by_key = {}
            for row in reader:
                key = (row["sequence"].strip(), row["source_index"].strip())
                if key in gt_by_key:
                    raise ValueError(f"duplicate GT key: {key}")
                gt_by_key[key] = row["gt_distance"]
    elif rows and "gt_distance" not in rows[0]:
        raise ValueError("manifest must contain gt_distance when --gt-csv is omitted")

    normalized = []
    seen = set()
    for row in rows:
        sequence = row["sequence"].strip()
        source_index = row["source_index"].strip()
        key = (sequence, source_index)
        if key in seen:
            raise ValueError(f"duplicate manifest key: {key}")
        seen.add(key)
        if gt_by_key is not None:
            if key not in gt_by_key:
                raise ValueError(f"missing GT for: {key}")
            gt_value = gt_by_key[key]
        else:
            gt_value = row["gt_distance"]
        gt_distance = float(gt_value)
        if not np.isfinite(gt_distance) or gt_distance <= 0:
            raise ValueError(f"invalid gt_distance for {key}: {gt_value}")

        image_path = Path(row["image_path"].strip())
        if not image_path.is_absolute():
            image_path = image_root / image_path
        normalized.append({
            "sequence": sequence,
            "source_index": source_index,
            "image_path": image_path.resolve(),
            "gt_distance": gt_distance,
        })
    return normalized


def infer_da2(engine: _TrtEngine, bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (INDOOR_W, INDOOR_H), interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - INDOOR_MEAN) / INDOOR_STD
    tensor = rgb.transpose(2, 0, 1)[None]
    depth = engine.run(tensor)[0][0, 0]
    return cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)


def infer_yolo(engine: _TrtEngine, bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    letterboxed, ratio, dw, dh = _letterbox(bgr, 768)
    tensor = (
        letterboxed[:, :, ::-1]
        .transpose(2, 0, 1)[None]
        .astype(np.float32) / 255.0
    )
    depth768 = engine.run(tensor)[0][0, 0]
    return _unwrap_depth(depth768, height, width, ratio, dw, dh)


def roi_features(depth: np.ndarray, label: str) -> dict[str, float]:
    if depth.shape != (480, 640):
        raise ValueError(f"{label} depth shape is {depth.shape}, expected (480, 640)")
    roi = depth[
        INDOOR_ROI_ROWS[0]:INDOOR_ROI_ROWS[1],
        INDOOR_ROI_COLS[0]:INDOOR_ROI_COLS[1],
    ]
    values = roi[np.isfinite(roi) & (roi > 0)]
    if values.size == 0:
        raise ValueError(f"{label} has no positive finite values in strict ROI")
    percentiles = np.percentile(values, [1.0, 5.0, 10.0, 20.0, 50.0])
    return {
        "min": float(values.min()),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p20": float(percentiles[3]),
        "median": float(percentiles[4]),
    }


def warmup(da2: _TrtEngine, yolo: _TrtEngine, bgr: np.ndarray, count: int) -> None:
    for _ in range(max(0, count)):
        infer_da2(da2, bgr)
        infer_yolo(yolo, bgr)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    gt_path = Path(args.gt_csv).expanduser().resolve() if args.gt_csv else None
    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root else manifest_path.parent
    )
    rows = load_rows(manifest_path, gt_path, image_root)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("manifest contains no rows")

    first_image = cv2.imread(str(rows[0]["image_path"]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(rows[0]["image_path"])
    if first_image.shape[:2] != (480, 640):
        raise ValueError(f"TUM image must be 640x480, got {first_image.shape[1]}x{first_image.shape[0]}")

    da2 = _TrtEngine(str(Path(args.da2_engine).expanduser()))
    yolo = _TrtEngine(str(Path(args.yolo_engine).expanduser()))
    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        warmup(da2, yolo, first_image, args.warmup)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            total = len(rows)
            for index, row in enumerate(rows, 1):
                bgr = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(row["image_path"])
                if bgr.shape[:2] != (480, 640):
                    raise ValueError(
                        f"{row['image_path']} must be 640x480, "
                        f"got {bgr.shape[1]}x{bgr.shape[0]}"
                    )

                started = time.perf_counter()
                da2_depth = infer_da2(da2, bgr)
                da2_ms = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                yolo_depth = infer_yolo(yolo, bgr)
                yolo_ms = (time.perf_counter() - started) * 1000.0
                da2_features = roi_features(da2_depth, "DA2")
                yolo_features = roi_features(yolo_depth, "YOLO")

                output = {
                    "sequence": row["sequence"],
                    "source_index": row["source_index"],
                    "image_path": str(row["image_path"]),
                    "gt_distance": row["gt_distance"],
                    "da2_inference_ms": da2_ms,
                    "yolo_inference_ms": yolo_ms,
                }
                output.update({f"da2_{key}": value for key, value in da2_features.items()})
                output.update({f"yolo_{key}": value for key, value in yolo_features.items()})
                writer.writerow(output)
                handle.flush()
                print(f"[{index}/{total}] {row['sequence']}:{row['source_index']}", flush=True)
    finally:
        da2.close()
        yolo.close()

    print(f"wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
