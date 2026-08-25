#!/usr/bin/env python3
"""Fit YOLO depth-to-TUM dense calibration with strict sequence holdouts."""

from __future__ import annotations

import argparse
import bisect
import csv
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
CALIBRATIONS = ("raw", "dense_scale", "dense_affine")
ROI_ROWS = (0, 300)
ROI_COLS = (213, 426)


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


@dataclass
class DenseStats:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_xy: float = 0.0
    sum_y2: float = 0.0

    def add_arrays(self, x: np.ndarray, y: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        self.n += int(x64.size)
        self.sum_x += float(np.sum(x64, dtype=np.float64))
        self.sum_y += float(np.sum(y64, dtype=np.float64))
        self.sum_x2 += float(np.dot(x64, x64))
        self.sum_xy += float(np.dot(x64, y64))
        self.sum_y2 += float(np.dot(y64, y64))

    def add_stats(self, other: "DenseStats") -> None:
        self.n += other.n
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_x2 += other.sum_x2
        self.sum_xy += other.sum_xy
        self.sum_y2 += other.sum_y2

    def as_dict(self) -> dict[str, int | float]:
        return {
            "n": self.n,
            "sum_x": self.sum_x,
            "sum_y": self.sum_y,
            "sum_x2": self.sum_x2,
            "sum_xy": self.sum_xy,
            "sum_y2": self.sum_y2,
        }


@dataclass(frozen=True)
class DepthIndex:
    depth_txt: Path
    timestamps: tuple[float, ...]
    paths: tuple[Path, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit dense_pixel_calibration from YOLO raw depth to aligned TUM depth, "
            "then evaluate with leave-one-sequence-out holdouts."
        )
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Base for relative manifest image_path values; defaults to manifest directory.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--yolo-engine", default="/models/obstacle/yolo26n-depth_int8.trt"
    )
    parser.add_argument("--max-timestamp-delta", type=float, default=0.02)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--synthetic-self-test",
        action="store_true",
        help="Test sufficient-statistics fitting without loading ROS/TensorRT.",
    )
    args = parser.parse_args()
    if not args.synthetic_self_test and (args.manifest is None or args.output_dir is None):
        parser.error("--manifest and --output-dir are required unless --synthetic-self-test is used")
    return args


def _load_manifest(path: Path, image_root: Path | None) -> list[ManifestRow]:
    manifest_path = path.expanduser().resolve()
    root = image_root.expanduser().resolve() if image_root else manifest_path.parent
    rows: list[ManifestRow] = []
    seen: set[tuple[str, str]] = set()
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
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
            rows.append(
                ManifestRow(
                    sequence=sequence,
                    source_index=source_index,
                    image_path=image_path.resolve(),
                    gt_distance=gt_distance,
                )
            )
    counts = {sequence: sum(row.sequence == sequence for row in rows) for sequence in SEQUENCES}
    missing_sequences = [sequence for sequence, count in counts.items() if count == 0]
    if missing_sequences:
        raise ValueError(f"manifest has no frames for {missing_sequences}; counts={counts}")
    print(f"manifest_frames={len(rows)} sequence_counts={json.dumps(counts, sort_keys=True)}")
    return rows


def _find_depth_txt(image_path: Path) -> Path | None:
    for directory in (image_path.parent, *image_path.parents[1:]):
        candidate = directory / "depth.txt"
        if candidate.is_file():
            return candidate
    return None


def _read_depth_index(depth_txt: Path) -> DepthIndex:
    entries: list[tuple[float, Path]] = []
    with depth_txt.open(encoding="utf-8-sig") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"invalid depth.txt row {depth_txt}:{line_no}")
            try:
                timestamp = float(parts[0])
            except ValueError as exc:
                raise ValueError(
                    f"invalid depth timestamp {depth_txt}:{line_no}: {parts[0]!r}"
                ) from exc
            depth_path = Path(parts[1])
            if not depth_path.is_absolute():
                depth_path = depth_txt.parent / depth_path
            entries.append((timestamp, depth_path.resolve()))
    if not entries:
        raise ValueError(f"no depth entries in {depth_txt}")
    entries.sort(key=lambda item: item[0])
    return DepthIndex(
        depth_txt=depth_txt,
        timestamps=tuple(item[0] for item in entries),
        paths=tuple(item[1] for item in entries),
    )


def _nearest_depth(index: DepthIndex, timestamp: float) -> tuple[float, Path, float]:
    insertion = bisect.bisect_left(index.timestamps, timestamp)
    candidates = []
    if insertion < len(index.timestamps):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    nearest = min(candidates, key=lambda position: abs(index.timestamps[position] - timestamp))
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
            alignments.append(
                Alignment(row, None, None, None, None, None, False, "invalid_rgb_timestamp")
            )
            continue
        depth_txt = _find_depth_txt(row.image_path)
        if depth_txt is None:
            alignments.append(
                Alignment(row, rgb_timestamp, None, None, None, None, False, "depth_txt_not_found")
            )
            continue
        try:
            index = cache.get(depth_txt)
            if index is None:
                index = _read_depth_index(depth_txt)
                cache[depth_txt] = index
            depth_timestamp, depth_path, delta = _nearest_depth(index, rgb_timestamp)
        except (OSError, ValueError) as exc:
            alignments.append(
                Alignment(
                    row, rgb_timestamp, None, None, depth_txt, None, False,
                    f"depth_index_error:{exc}",
                )
            )
            continue
        matched = delta <= max_delta
        failure = "" if matched else "timestamp_delta_exceeded"
        if matched and not depth_path.is_file():
            matched = False
            failure = "depth_image_not_found"
        alignments.append(
            Alignment(
                row, rgb_timestamp, depth_timestamp, delta, depth_txt, depth_path,
                matched, failure,
            )
        )
    return alignments


def _alignment_summary(alignments: Sequence[Alignment]) -> dict[str, object]:
    matched = [item for item in alignments if item.matched]
    deltas = [item.timestamp_delta for item in matched if item.timestamp_delta is not None]
    return {
        "frames": len(alignments),
        "matched": len(matched),
        "unmatched": len(alignments) - len(matched),
        "max_delta": max(deltas) if deltas else None,
        "mean_delta": float(np.mean(deltas)) if deltas else None,
    }


def _write_alignment_report(path: Path, alignments: Sequence[Alignment]) -> None:
    fields = [
        "sequence", "source_index", "image_path", "rgb_timestamp", "depth_txt",
        "depth_timestamp", "depth_path", "timestamp_delta", "matched", "failure",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in alignments:
            writer.writerow(
                {
                    "sequence": item.row.sequence,
                    "source_index": item.row.source_index,
                    "image_path": item.row.image_path,
                    "rgb_timestamp": item.rgb_timestamp,
                    "depth_txt": item.depth_txt or "",
                    "depth_timestamp": item.depth_timestamp,
                    "depth_path": item.depth_path or "",
                    "timestamp_delta": item.timestamp_delta,
                    "matched": int(item.matched),
                    "failure": item.failure,
                }
            )


def _load_yolo_helpers() -> tuple[type, Callable[[object, np.ndarray], np.ndarray]]:
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    benchmark = importlib.import_module("benchmark_indoor_depth_models")
    if tuple(benchmark.INDOOR_ROI_ROWS) != ROI_ROWS:
        raise RuntimeError(f"production ROI rows changed: {benchmark.INDOOR_ROI_ROWS}")
    if tuple(benchmark.INDOOR_ROI_COLS) != ROI_COLS:
        raise RuntimeError(f"production ROI cols changed: {benchmark.INDOOR_ROI_COLS}")
    return benchmark._TrtEngine, benchmark.infer_yolo


def _load_cv2() -> object:
    return importlib.import_module("cv2")


def _read_rgb(path: Path) -> np.ndarray:
    cv2 = _load_cv2()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[:2] != (480, 640):
        raise ValueError(f"RGB must be 640x480, got {bgr.shape[1]}x{bgr.shape[0]}: {path}")
    return bgr


def _read_tum_depth(path: Path, depth_scale: float) -> np.ndarray:
    if depth_scale <= 0.0:
        raise ValueError("--depth-scale must be positive")
    cv2 = _load_cv2()
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise FileNotFoundError(path)
    if encoded.ndim != 2 or encoded.shape != (480, 640):
        raise ValueError(f"TUM depth must be 640x480 single-channel: {path} shape={encoded.shape}")
    return encoded.astype(np.float32) / depth_scale


def _valid_dense_pairs(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_roi = pred[ROI_ROWS[0] : ROI_ROWS[1], ROI_COLS[0] : ROI_COLS[1]]
    gt_roi = gt[ROI_ROWS[0] : ROI_ROWS[1], ROI_COLS[0] : ROI_COLS[1]]
    valid = np.isfinite(pred_roi) & (pred_roi > 0.0) & np.isfinite(gt_roi) & (gt_roi > 0.0)
    return pred_roi[valid], gt_roi[valid]


def _accumulate_sequence_stats(
    engine: object,
    infer_yolo: Callable[[object, np.ndarray], np.ndarray],
    alignments: Sequence[Alignment],
    depth_scale: float,
) -> tuple[dict[str, DenseStats], dict[str, int]]:
    stats = {sequence: DenseStats() for sequence in SEQUENCES}
    usable_frames = {sequence: 0 for sequence in SEQUENCES}
    matched = [item for item in alignments if item.matched]
    for index, item in enumerate(matched, start=1):
        if item.depth_path is None:
            raise RuntimeError("matched alignment is missing depth_path")
        pred = infer_yolo(engine, _read_rgb(item.row.image_path))
        gt = _read_tum_depth(item.depth_path, depth_scale)
        x, y = _valid_dense_pairs(pred, gt)
        if x.size:
            stats[item.row.sequence].add_arrays(x, y)
            usable_frames[item.row.sequence] += 1
        print(
            f"dense_stats [{index}/{len(matched)}] {item.row.sequence}:"
            f"{item.row.source_index} valid_pairs={x.size}",
            flush=True,
        )
    return stats, usable_frames


def _combine_stats(stats: dict[str, DenseStats], sequences: Sequence[str]) -> DenseStats:
    combined = DenseStats()
    for sequence in sequences:
        combined.add_stats(stats[sequence])
    return combined


def _fit_scale(stats: DenseStats) -> float:
    if stats.n == 0 or stats.sum_x2 <= 0.0:
        raise ValueError("cannot fit dense scale: no valid pairs or zero sum_x2")
    scale = stats.sum_xy / stats.sum_x2
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid dense scale slope: {scale}")
    return scale


def _fit_affine(stats: DenseStats) -> tuple[float, float]:
    if stats.n < 2:
        raise ValueError("cannot fit dense affine: fewer than two valid pairs")
    denominator = stats.n * stats.sum_x2 - stats.sum_x * stats.sum_x
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"cannot fit dense affine: invalid normal-equation denominator {denominator}")
    slope = (stats.n * stats.sum_xy - stats.sum_x * stats.sum_y) / denominator
    intercept = (stats.sum_y - slope * stats.sum_x) / stats.n
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError(f"non-finite dense affine parameters: a={slope} b={intercept}")
    if slope <= 0.0:
        raise ValueError(f"dense affine slope must be positive, got {slope}")
    return slope, intercept


def _build_folds(
    stats: dict[str, DenseStats], usable_frames: dict[str, int], alignments: Sequence[Alignment]
) -> list[dict[str, object]]:
    folds = []
    for holdout in SEQUENCES:
        train_sequences = [sequence for sequence in SEQUENCES if sequence != holdout]
        train_stats = _combine_stats(stats, train_sequences)
        scale = _fit_scale(train_stats)
        affine_a, affine_b = _fit_affine(train_stats)
        alignment_matched = {
            sequence: sum(
                item.matched and item.row.sequence == sequence for item in alignments
            )
            for sequence in SEQUENCES
        }
        folds.append(
            {
                "calibration_type": "dense_pixel_calibration",
                "holdout_sequence": holdout,
                "train_sequences": train_sequences,
                "train_matched_frames": sum(alignment_matched[s] for s in train_sequences),
                "holdout_matched_frames": alignment_matched[holdout],
                "train_frames_with_valid_dense_pairs": sum(
                    usable_frames[s] for s in train_sequences
                ),
                "holdout_frames_with_valid_dense_pairs": usable_frames[holdout],
                "train_dense_valid_pixel_pairs": train_stats.n,
                "holdout_dense_valid_pixel_pairs": stats[holdout].n,
                "train_sufficient_statistics": train_stats.as_dict(),
                "scale": {"a": scale},
                "affine": {"a": affine_a, "b": affine_b},
            }
        )
    return folds


def _depth_features(depth: np.ndarray) -> dict[str, float]:
    roi = depth[ROI_ROWS[0] : ROI_ROWS[1], ROI_COLS[0] : ROI_COLS[1]]
    values = roi[np.isfinite(roi) & (roi > 0.0)]
    if values.size == 0:
        raise ValueError("calibrated YOLO depth has no valid values in official Indoor ROI")
    percentiles = np.percentile(values, [1.0, 5.0, 10.0])
    return {
        "min": float(np.min(values)),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p10": float(percentiles[2]),
    }


def _evaluate_frames(
    engine: object,
    infer_yolo: Callable[[object, np.ndarray], np.ndarray],
    rows: Sequence[ManifestRow],
    folds: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    fold_by_sequence = {str(fold["holdout_sequence"]): fold for fold in folds}
    predictions = []
    for index, row in enumerate(rows, start=1):
        fold = fold_by_sequence[row.sequence]
        raw = infer_yolo(engine, _read_rgb(row.image_path))
        scale_a = float(fold["scale"]["a"])  # type: ignore[index]
        affine_a = float(fold["affine"]["a"])  # type: ignore[index]
        affine_b = float(fold["affine"]["b"])  # type: ignore[index]
        maps = {
            "raw": raw,
            "dense_scale": scale_a * raw,
            "dense_affine": affine_a * raw + affine_b,
        }
        output: dict[str, object] = {
            "calibration_type": "dense_pixel_calibration",
            "holdout_sequence": row.sequence,
            "sequence": row.sequence,
            "source_index": row.source_index,
            "image_path": str(row.image_path),
            "gt_distance": row.gt_distance,
            "dense_scale_a": scale_a,
            "dense_affine_a": affine_a,
            "dense_affine_b": affine_b,
        }
        for calibration, depth in maps.items():
            for feature, value in _depth_features(depth).items():
                output[f"{calibration}_{feature}"] = value
        predictions.append(output)
        print(f"holdout_eval [{index}/{len(rows)}] {row.sequence}:{row.source_index}", flush=True)
    return predictions


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, int | float | None]:
    if gt.size == 0:
        return {
            "count": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": None, "rmse": None,
        }
    positive = gt < 2.0
    predicted_positive = pred < 2.0
    tp = int(np.sum(positive & predicted_positive))
    fp = int(np.sum(~positive & predicted_positive))
    fn = int(np.sum(positive & ~predicted_positive))
    tn = int(np.sum(~positive & ~predicted_positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    error = pred - gt
    return {
        "count": int(gt.size), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _metric_pair(gt: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    boundary = (gt >= 1.5) & (gt <= 2.5)
    return {
        "all": _metrics(gt, pred),
        "boundary_1p5_to_2p5": _metrics(gt[boundary], pred[boundary]),
    }


def _calculate_reports(
    predictions: Sequence[dict[str, object]], folds: list[dict[str, object]]
) -> dict[str, dict[str, object]]:
    aggregate: dict[str, dict[str, object]] = {}
    for calibration in CALIBRATIONS:
        aggregate[calibration] = {}
        for feature in FEATURES:
            gt = np.asarray([row["gt_distance"] for row in predictions], dtype=np.float64)
            pred = np.asarray(
                [row[f"{calibration}_{feature}"] for row in predictions], dtype=np.float64
            )
            aggregate[calibration][feature] = _metric_pair(gt, pred)

    for fold in folds:
        fold_rows = [
            row for row in predictions if row["sequence"] == fold["holdout_sequence"]
        ]
        fold_metrics: dict[str, dict[str, object]] = {}
        for calibration in CALIBRATIONS:
            fold_metrics[calibration] = {}
            for feature in FEATURES:
                gt = np.asarray([row["gt_distance"] for row in fold_rows], dtype=np.float64)
                pred = np.asarray(
                    [row[f"{calibration}_{feature}"] for row in fold_rows], dtype=np.float64
                )
                fold_metrics[calibration][feature] = _metric_pair(gt, pred)
        fold["holdout_metrics"] = fold_metrics
    return aggregate


def _best_feature(metrics: dict[str, object]) -> str:
    def key(feature: str) -> tuple[float, float, float]:
        values = metrics[feature]["all"]  # type: ignore[index]
        return (float(values["f1"]), -float(values["mae"]), -float(values["rmse"]))

    return max(FEATURES, key=key)


def _write_predictions(path: Path, predictions: Sequence[dict[str, object]]) -> None:
    fields = [
        "calibration_type", "holdout_sequence", "sequence", "source_index", "image_path",
        "gt_distance", "dense_scale_a", "dense_affine_a", "dense_affine_b",
        *[f"{calibration}_{feature}" for calibration in CALIBRATIONS for feature in FEATURES],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def _warmup(
    engine: object,
    infer_yolo: Callable[[object, np.ndarray], np.ndarray],
    image: np.ndarray,
    count: int,
) -> None:
    for _ in range(max(0, count)):
        infer_yolo(engine, image)


def _synthetic_self_test() -> None:
    pred = np.linspace(0.2, 8.0, 10000, dtype=np.float64)
    gt = 1.4 * pred + 0.3
    stats = DenseStats()
    stats.add_arrays(pred, gt)
    affine_a, affine_b = _fit_affine(stats)
    scale_a = _fit_scale(stats)
    if not math.isclose(affine_a, 1.4, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError(f"synthetic affine slope mismatch: {affine_a}")
    if not math.isclose(affine_b, 0.3, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError(f"synthetic affine intercept mismatch: {affine_b}")
    print(
        f"synthetic_self_test=passed affine_a={affine_a:.12f} "
        f"affine_b={affine_b:.12f} scale_a={scale_a:.12f}"
    )


def main() -> int:
    args = _parse_args()
    if args.synthetic_self_test:
        _synthetic_self_test()
        return 0

    rows = _load_manifest(args.manifest, args.image_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    alignments = _align_rows(rows, args.max_timestamp_delta)
    alignment_path = output_dir / "dense_alignment_report.csv"
    _write_alignment_report(alignment_path, alignments)
    overall_alignment = _alignment_summary(alignments)
    sequence_alignment = {
        sequence: _alignment_summary(
            [item for item in alignments if item.row.sequence == sequence]
        )
        for sequence in SEQUENCES
    }
    print(f"alignment_overall={json.dumps(overall_alignment, sort_keys=True)}")
    for sequence, summary in sequence_alignment.items():
        print(f"alignment_{sequence}={json.dumps(summary, sort_keys=True)}")
    if int(overall_alignment["unmatched"]) > 0:
        print(f"WARNING unmatched_frames={overall_alignment['unmatched']} details={alignment_path}")

    TrtEngine, infer_yolo = _load_yolo_helpers()
    engine = TrtEngine(str(Path(args.yolo_engine).expanduser()))
    try:
        first_image = _read_rgb(rows[0].image_path)
        _warmup(engine, infer_yolo, first_image, args.warmup)
        sequence_stats, usable_frames = _accumulate_sequence_stats(
            engine, infer_yolo, alignments, args.depth_scale
        )
        folds = _build_folds(sequence_stats, usable_frames, alignments)
        predictions = _evaluate_frames(engine, infer_yolo, rows, folds)
    finally:
        engine.close()

    aggregate = _calculate_reports(predictions, folds)
    best = {calibration: _best_feature(aggregate[calibration]) for calibration in CALIBRATIONS}
    prediction_path = output_dir / "dense_holdout_predictions.csv"
    _write_predictions(prediction_path, predictions)
    report_path = output_dir / "dense_calibration_report.json"
    report = {
        "experiment_type": "dense_pixel_calibration",
        "manifest": str(args.manifest.expanduser().resolve()),
        "yolo_engine": str(Path(args.yolo_engine).expanduser()),
        "roi_rows": list(ROI_ROWS),
        "roi_cols": list(ROI_COLS),
        "depth_scale": args.depth_scale,
        "max_timestamp_delta_sec": args.max_timestamp_delta,
        "threshold_rule": "gt_distance < 2.0; pred_distance < 2.0",
        "boundary_subset": "1.5 <= gt_distance <= 2.5",
        "alignment_overall": overall_alignment,
        "alignment_by_sequence": sequence_alignment,
        "sequence_sufficient_statistics": {
            sequence: sequence_stats[sequence].as_dict() for sequence in SEQUENCES
        },
        "folds": folds,
        "aggregate_holdout_metrics": aggregate,
        "best_features": best,
        "selection_rule": "aggregate holdout F1, then MAE, then RMSE",
    }
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")

    for calibration in CALIBRATIONS:
        print(f"best_{calibration}={best[calibration]}")
    print(f"dense_calibration_report={report_path}")
    print(f"dense_holdout_predictions={prediction_path}")
    print(f"dense_alignment_report={alignment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
