#!/usr/bin/env python3
"""Fit DA2 dense scale calibration with strict sequence holdouts."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


SEQUENCES = ("slam1", "slam2", "slam3", "pioneer_360")
FEATURES = ("min", "p1", "p5", "p10")
ROI_ROWS = (0, 300)
ROI_COLS = (213, 426)
PREPROCESSING = (
    "BGR->RGB; float32 /255; resize 686x518 INTER_CUBIC; "
    "ImageNet mean/std; NCHW; DA2 TensorRT; resize depth to 640x480 INTER_LINEAR"
)


@dataclass(frozen=True)
class ManifestRow:
    sequence: str
    source_index: str
    image_path: Path
    gt_distance: float


@dataclass(frozen=True)
class Alignment:
    row: ManifestRow
    rgb_timestamp: float | None
    depth_timestamp: float | None
    timestamp_delta: float | None
    depth_txt: Path | None
    depth_path: Path | None
    matched: bool
    failure: str


@dataclass(frozen=True)
class DepthIndex:
    timestamps: tuple[float, ...]
    paths: tuple[Path, ...]


@dataclass
class DenseStats:
    n: int = 0
    sum_x2: float = 0.0
    sum_xy: float = 0.0

    def add_arrays(self, pred: np.ndarray, gt: np.ndarray) -> None:
        x = np.asarray(pred, dtype=np.float64)
        y = np.asarray(gt, dtype=np.float64)
        self.n += int(x.size)
        self.sum_x2 += float(np.dot(x, x))
        self.sum_xy += float(np.dot(x, y))

    def add_stats(self, other: "DenseStats") -> None:
        self.n += other.n
        self.sum_x2 += other.sum_x2
        self.sum_xy += other.sum_xy

    def as_dict(self) -> dict[str, int | float]:
        return {"dense_pixel_count": self.n, "sum_pred_squared": self.sum_x2,
                "sum_pred_gt": self.sum_xy}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one global DA2 metric scale per strict LOSO fold."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--da2-engine",
        default="/models/obstacle/depth_anything_v2_metric_hypersim_vits_int8.trt",
    )
    parser.add_argument("--max-timestamp-delta", type=float, default=0.02)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_self_test and (args.manifest is None or args.output_dir is None):
        parser.error("--manifest and --output-dir are required unless --synthetic-self-test is used")
    return args


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _load_manifest(path: Path, image_root: Path | None) -> list[ManifestRow]:
    root = image_root.expanduser().resolve() if image_root else path.parent
    rows: list[ManifestRow] = []
    seen: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sequence", "source_index", "image_path", "gt_distance"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        for line_no, item in enumerate(reader, start=2):
            sequence = item["sequence"].strip()
            if sequence not in SEQUENCES:
                raise ValueError(f"unexpected sequence at line {line_no}: {sequence!r}")
            source_index = item["source_index"].strip()
            key = (sequence, source_index)
            if key in seen:
                raise ValueError(f"duplicate sequence/source_index at line {line_no}: {key}")
            seen.add(key)
            image_path = Path(item["image_path"].strip()).expanduser()
            if not image_path.is_absolute():
                image_path = root / image_path
            gt_distance = float(item["gt_distance"])
            if not math.isfinite(gt_distance) or gt_distance <= 0.0:
                raise ValueError(f"invalid gt_distance at line {line_no}: {gt_distance}")
            rows.append(ManifestRow(sequence, source_index, image_path.resolve(), gt_distance))
    counts = {sequence: sum(row.sequence == sequence for row in rows) for sequence in SEQUENCES}
    missing_sequences = [sequence for sequence, count in counts.items() if count == 0]
    if missing_sequences:
        raise ValueError(f"manifest has no frames for {missing_sequences}; counts={counts}")
    return rows


def _find_depth_txt(image_path: Path) -> Path | None:
    for directory in (image_path.parent, *image_path.parents[1:]):
        candidate = directory / "depth.txt"
        if candidate.is_file():
            return candidate
    return None


def _read_depth_index(path: Path) -> DepthIndex:
    entries: list[tuple[float, Path]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"invalid depth.txt row {path}:{line_no}")
            depth_path = Path(parts[1])
            if not depth_path.is_absolute():
                depth_path = path.parent / depth_path
            entries.append((float(parts[0]), depth_path.resolve()))
    if not entries:
        raise ValueError(f"no depth entries in {path}")
    entries.sort(key=lambda item: item[0])
    return DepthIndex(tuple(item[0] for item in entries), tuple(item[1] for item in entries))


def _nearest_depth(index: DepthIndex, timestamp: float) -> tuple[float, Path, float]:
    insertion = bisect.bisect_left(index.timestamps, timestamp)
    candidates = []
    if insertion < len(index.timestamps):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    nearest = min(candidates, key=lambda i: abs(index.timestamps[i] - timestamp))
    depth_timestamp = index.timestamps[nearest]
    return depth_timestamp, index.paths[nearest], abs(depth_timestamp - timestamp)


def _align_rows(rows: Sequence[ManifestRow], max_delta: float) -> list[Alignment]:
    if max_delta < 0.0:
        raise ValueError("--max-timestamp-delta must be non-negative")
    cache: dict[Path, DepthIndex] = {}
    alignments: list[Alignment] = []
    for row in rows:
        try:
            rgb_timestamp = float(row.image_path.stem)
        except ValueError:
            alignments.append(Alignment(row, None, None, None, None, None, False,
                                        "invalid_rgb_timestamp"))
            continue
        depth_txt = _find_depth_txt(row.image_path)
        if depth_txt is None:
            alignments.append(Alignment(row, rgb_timestamp, None, None, None, None, False,
                                        "depth_txt_not_found"))
            continue
        try:
            index = cache.get(depth_txt)
            if index is None:
                index = _read_depth_index(depth_txt)
                cache[depth_txt] = index
            depth_timestamp, depth_path, delta = _nearest_depth(index, rgb_timestamp)
        except (OSError, ValueError) as exc:
            alignments.append(Alignment(row, rgb_timestamp, None, None, depth_txt, None,
                                        False, f"depth_index_error:{exc}"))
            continue
        matched = delta <= max_delta and depth_path.is_file()
        failure = "" if matched else (
            "timestamp_delta_exceeded" if delta > max_delta else "depth_image_not_found"
        )
        alignments.append(Alignment(row, rgb_timestamp, depth_timestamp, delta, depth_txt,
                                    depth_path, matched, failure))
    return alignments


def _write_alignment_report(path: Path, alignments: Sequence[Alignment]) -> None:
    fields = ["sequence", "source_index", "image_path", "rgb_timestamp", "depth_txt",
              "depth_timestamp", "depth_path", "timestamp_delta", "matched", "failure"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in alignments:
            writer.writerow({
                "sequence": item.row.sequence, "source_index": item.row.source_index,
                "image_path": item.row.image_path, "rgb_timestamp": item.rgb_timestamp,
                "depth_txt": item.depth_txt or "", "depth_timestamp": item.depth_timestamp,
                "depth_path": item.depth_path or "", "timestamp_delta": item.timestamp_delta,
                "matched": int(item.matched), "failure": item.failure,
            })


def _load_da2_helpers() -> tuple[type, Callable[[object, np.ndarray], np.ndarray], object]:
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    benchmark = importlib.import_module("benchmark_indoor_depth_models")
    if tuple(benchmark.INDOOR_ROI_ROWS) != ROI_ROWS or tuple(benchmark.INDOOR_ROI_COLS) != ROI_COLS:
        raise RuntimeError("benchmark production ROI no longer matches the official Indoor ROI")
    return benchmark._TrtEngine, benchmark.infer_da2, benchmark.cv2


def _read_rgb(path: Path, cv2: object) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[:2] != (480, 640):
        raise ValueError(f"RGB must be 640x480, got {bgr.shape[1]}x{bgr.shape[0]}: {path}")
    return bgr


def _read_tum_depth(path: Path, depth_scale: float, cv2: object) -> np.ndarray:
    if depth_scale <= 0.0:
        raise ValueError("--depth-scale must be positive")
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise FileNotFoundError(path)
    if encoded.ndim != 2 or encoded.shape != (480, 640):
        raise ValueError(f"TUM depth must be 640x480 single-channel: {path} shape={encoded.shape}")
    return encoded.astype(np.float32) / depth_scale


def _valid_dense_pairs(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_roi = pred[ROI_ROWS[0]:ROI_ROWS[1], ROI_COLS[0]:ROI_COLS[1]]
    gt_roi = gt[ROI_ROWS[0]:ROI_ROWS[1], ROI_COLS[0]:ROI_COLS[1]]
    valid = np.isfinite(pred_roi) & (pred_roi > 0.0) & np.isfinite(gt_roi) & (gt_roi > 0.0)
    return pred_roi[valid], gt_roi[valid]


def _features(depth: np.ndarray) -> dict[str, float]:
    roi = depth[ROI_ROWS[0]:ROI_ROWS[1], ROI_COLS[0]:ROI_COLS[1]]
    values = roi[np.isfinite(roi) & (roi > 0.0)]
    if values.size == 0:
        raise ValueError("DA2 depth has no positive finite values in official Indoor ROI")
    percentiles = np.percentile(values, [1.0, 5.0, 10.0])
    return {"min": float(values.min()), "p1": float(percentiles[0]),
            "p5": float(percentiles[1]), "p10": float(percentiles[2])}


def _fit_scale(stats: DenseStats) -> float:
    if stats.n == 0 or stats.sum_x2 <= 0.0:
        raise ValueError("cannot fit dense scale without valid dense pixel pairs")
    scale = stats.sum_xy / stats.sum_x2
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid dense scale: {scale}")
    return scale


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, int | float | None]:
    if gt.size == 0:
        return {"count": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": None, "rmse": None}
    actual = gt < 2.0
    predicted = pred < 2.0
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    error = pred - gt
    return {"count": int(gt.size), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error))))}


def _metric_pair(gt: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    boundary = (gt >= 1.5) & (gt <= 2.5)
    return {"all": _metrics(gt, pred),
            "boundary_1p5_to_2p5": _metrics(gt[boundary], pred[boundary])}


def _calculate_metrics(predictions: Sequence[dict[str, object]]) -> dict[str, object]:
    gt = np.asarray([row["gt_distance"] for row in predictions], dtype=np.float64)
    result: dict[str, object] = {"raw": {}, "dense_scale": {}}
    for calibration in result:
        for feature in FEATURES:
            pred = np.asarray([row[f"{calibration}_{feature}"] for row in predictions],
                              dtype=np.float64)
            result[calibration][feature] = _metric_pair(gt, pred)  # type: ignore[index]
    return result


def _synthetic_self_test() -> int:
    pred = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    gt = 1.25 * pred
    stats = DenseStats()
    stats.add_arrays(pred, gt)
    scale = _fit_scale(stats)
    if not math.isclose(scale, 1.25, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"expected scale 1.25, got {scale}")
    print(f"synthetic_self_test=PASS scale={scale:.12f}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.synthetic_self_test:
        return _synthetic_self_test()

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = Path(args.da2_engine).expanduser().resolve()
    rows = _load_manifest(manifest_path, args.image_root)
    alignments = _align_rows(rows, args.max_timestamp_delta)
    alignment_path = output_dir / "da2_dense_alignment_report.csv"
    _write_alignment_report(alignment_path, alignments)

    engine_type, infer_da2, cv2 = _load_da2_helpers()
    engine = engine_type(str(engine_path))
    sequence_stats = {sequence: DenseStats() for sequence in SEQUENCES}
    matched = [item for item in alignments if item.matched]
    try:
        first_bgr = _read_rgb(rows[0].image_path, cv2)
        for _ in range(max(0, args.warmup)):
            infer_da2(engine, first_bgr)
        for index, item in enumerate(matched, start=1):
            if item.depth_path is None:
                raise RuntimeError("matched alignment has no depth path")
            raw = infer_da2(engine, _read_rgb(item.row.image_path, cv2))
            gt = _read_tum_depth(item.depth_path, args.depth_scale, cv2)
            pred_values, gt_values = _valid_dense_pairs(raw, gt)
            sequence_stats[item.row.sequence].add_arrays(pred_values, gt_values)
            print(f"dense_fit [{index}/{len(matched)}] {item.row.sequence}:"
                  f"{item.row.source_index} pixels={pred_values.size}", flush=True)

        folds: list[dict[str, object]] = []
        scale_by_holdout: dict[str, float] = {}
        for holdout in SEQUENCES:
            train_sequences = [sequence for sequence in SEQUENCES if sequence != holdout]
            train_stats = DenseStats()
            for sequence in train_sequences:
                train_stats.add_stats(sequence_stats[sequence])
            scale = _fit_scale(train_stats)
            scale_by_holdout[holdout] = scale
            folds.append({
                "train_sequences": train_sequences, "holdout_sequence": holdout,
                "dense_pixel_count": train_stats.n,
                "train_dense_pixel_count": train_stats.n,
                "holdout_dense_pixel_count": sequence_stats[holdout].n,
                "scale_a": scale,
                "train_sufficient_statistics": train_stats.as_dict(),
            })

        predictions: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            raw = infer_da2(engine, _read_rgb(row.image_path, cv2))
            scale = scale_by_holdout[row.sequence]
            calibrated = scale * raw
            raw_features = _features(raw)
            scaled_features = _features(calibrated)
            output: dict[str, object] = {
                "holdout_sequence": row.sequence, "sequence": row.sequence,
                "source_index": row.source_index, "image_path": str(row.image_path),
                "gt_distance": row.gt_distance, "scale_a": scale,
            }
            output.update({f"raw_{name}": value for name, value in raw_features.items()})
            output.update({f"dense_scale_{name}": value
                           for name, value in scaled_features.items()})
            predictions.append(output)
            print(f"holdout_eval [{index}/{len(rows)}] {row.sequence}:{row.source_index}",
                  flush=True)
    finally:
        engine.close()

    for fold in folds:
        fold_rows = [row for row in predictions
                     if row["sequence"] == fold["holdout_sequence"]]
        fold_metrics = _calculate_metrics(fold_rows)
        fold["raw_metrics"] = fold_metrics["raw"]
        fold["scaled_metrics"] = fold_metrics["dense_scale"]

    prediction_path = output_dir / "da2_dense_scale_predictions.csv"
    fields = ["holdout_sequence", "sequence", "source_index", "image_path",
              "gt_distance", "scale_a",
              *[f"raw_{feature}" for feature in FEATURES],
              *[f"dense_scale_{feature}" for feature in FEATURES]]
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)

    matched_deltas = [item.timestamp_delta for item in matched
                      if item.timestamp_delta is not None]
    report = {
        "experiment_type": "da2_dense_scale_strict_loso",
        "trt_engine_path": str(engine_path),
        "trt_engine_sha256": _sha256(engine_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "depth_scale": args.depth_scale,
        "max_timestamp_delta_sec": args.max_timestamp_delta,
        "roi_rows": list(ROI_ROWS), "roi_cols": list(ROI_COLS),
        "preprocessing": PREPROCESSING,
        "scale_formula": "a = sum(pred * gt) / sum(pred^2); D_cal = a * D_raw",
        "sequences": list(SEQUENCES),
        "threshold_rule": "gt_distance < 2.0; pred_distance < 2.0",
        "boundary_subset": "1.5 <= gt_distance <= 2.5",
        "alignment": {
            "frames": len(alignments), "matched": len(matched),
            "unmatched": len(alignments) - len(matched),
            "max_matched_delta_sec": max(matched_deltas) if matched_deltas else None,
            "mean_matched_delta_sec": float(np.mean(matched_deltas)) if matched_deltas else None,
        },
        "sequence_dense_statistics": {
            sequence: sequence_stats[sequence].as_dict() for sequence in SEQUENCES
        },
        "folds": folds,
        "aggregate_out_of_fold_metrics": _calculate_metrics(predictions),
    }
    report_path = output_dir / "da2_dense_scale_report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print(f"report={report_path}")
    print(f"predictions={prediction_path}")
    print(f"alignment={alignment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
