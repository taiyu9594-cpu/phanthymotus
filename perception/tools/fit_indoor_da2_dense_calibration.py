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
GT_RANGES = (
    ("gt_le_2p5", 2.5),
    ("gt_le_3p0", 3.0),
    ("gt_le_4p0", 4.0),
    ("gt_le_5p0", 5.0),
    ("all", None),
)
CALIBRATIONS = ("dense_scale", "dense_affine")
NONLINEAR_RANGES = (("all", None), ("gt_le_5m", 5.0))
NONLINEAR_WEIGHTINGS = ("pixel_equal", "frame_balanced")
RAW_BIN_WIDTH_M = 0.05
RAW_BIN_MAX_M = 100.0
RAW_PROBES_M = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
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
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_xy: float = 0.0

    def add_arrays(self, pred: np.ndarray, gt: np.ndarray) -> None:
        x = np.asarray(pred, dtype=np.float64)
        y = np.asarray(gt, dtype=np.float64)
        self.n += int(x.size)
        self.sum_x += float(np.sum(x, dtype=np.float64))
        self.sum_y += float(np.sum(y, dtype=np.float64))
        self.sum_x2 += float(np.dot(x, x))
        self.sum_xy += float(np.dot(x, y))

    def add_stats(self, other: "DenseStats") -> None:
        self.n += other.n
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_x2 += other.sum_x2
        self.sum_xy += other.sum_xy

    def as_dict(self) -> dict[str, int | float]:
        return {"dense_pixel_count": self.n, "sum_pred": self.sum_x,
                "sum_gt": self.sum_y, "sum_pred_squared": self.sum_x2,
                "sum_pred_gt": self.sum_xy}


@dataclass
class BinnedStats:
    pixel_count: np.ndarray
    pixel_sum_raw: np.ndarray
    pixel_sum_gt: np.ndarray
    frame_weight: np.ndarray
    frame_weighted_sum_raw: np.ndarray
    frame_weighted_sum_gt: np.ndarray
    frame_count: int = 0
    valid_pixel_count: int = 0
    overflow_pixel_count: int = 0

    @classmethod
    def empty(cls) -> "BinnedStats":
        size = int(math.ceil(RAW_BIN_MAX_M / RAW_BIN_WIDTH_M)) + 1
        return cls(
            np.zeros(size, dtype=np.int64),
            np.zeros(size, dtype=np.float64),
            np.zeros(size, dtype=np.float64),
            np.zeros(size, dtype=np.float64),
            np.zeros(size, dtype=np.float64),
            np.zeros(size, dtype=np.float64),
        )

    def add_frame(self, raw: np.ndarray, gt: np.ndarray) -> None:
        x = np.asarray(raw, dtype=np.float64)
        y = np.asarray(gt, dtype=np.float64)
        if x.size == 0:
            return
        last = self.pixel_count.size - 1
        indices = np.minimum((x / RAW_BIN_WIDTH_M).astype(np.int64), last)
        counts = np.bincount(indices, minlength=self.pixel_count.size)
        sum_raw = np.bincount(indices, weights=x, minlength=self.pixel_count.size)
        sum_gt = np.bincount(indices, weights=y, minlength=self.pixel_count.size)
        self.pixel_count += counts
        self.pixel_sum_raw += sum_raw
        self.pixel_sum_gt += sum_gt
        inverse_frame_pixels = 1.0 / float(x.size)
        self.frame_weight += counts * inverse_frame_pixels
        self.frame_weighted_sum_raw += sum_raw * inverse_frame_pixels
        self.frame_weighted_sum_gt += sum_gt * inverse_frame_pixels
        self.frame_count += 1
        self.valid_pixel_count += int(x.size)
        self.overflow_pixel_count += int(np.sum(x >= RAW_BIN_MAX_M))

    def summary(self) -> dict[str, int]:
        return {
            "frame_count": self.frame_count,
            "valid_pixel_count": self.valid_pixel_count,
            "occupied_bin_count": int(np.count_nonzero(self.pixel_count)),
            "overflow_pixel_count": self.overflow_pixel_count,
        }


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
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
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


def _fit_affine(stats: DenseStats) -> tuple[float, float]:
    if stats.n < 2:
        raise ValueError("cannot fit dense affine without at least two valid pixel pairs")
    denominator = stats.n * stats.sum_x2 - stats.sum_x * stats.sum_x
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"invalid dense affine denominator: {denominator}")
    slope = (stats.n * stats.sum_xy - stats.sum_x * stats.sum_y) / denominator
    intercept = (stats.sum_y - slope * stats.sum_x) / stats.n
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0.0:
        raise ValueError(f"invalid dense affine parameters: a={slope} b={intercept}")
    return slope, intercept


def _weighted_pava(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    block_start: list[int] = []
    block_end: list[int] = []
    block_weight: list[float] = []
    block_value: list[float] = []
    for index, (value, weight) in enumerate(zip(y, weights)):
        if not math.isfinite(float(value)) or not math.isfinite(float(weight)) or weight <= 0.0:
            raise ValueError("weighted PAVA requires finite values and positive weights")
        block_start.append(index)
        block_end.append(index)
        block_weight.append(float(weight))
        block_value.append(float(value))
        while len(block_value) >= 2 and block_value[-2] > block_value[-1]:
            combined_weight = block_weight[-2] + block_weight[-1]
            combined_value = (
                block_value[-2] * block_weight[-2]
                + block_value[-1] * block_weight[-1]
            ) / combined_weight
            block_end[-2] = block_end[-1]
            block_weight[-2] = combined_weight
            block_value[-2] = combined_value
            block_start.pop()
            block_end.pop()
            block_weight.pop()
            block_value.pop()
    fitted = np.empty_like(y, dtype=np.float64)
    for start, end, value in zip(block_start, block_end, block_value):
        fitted[start:end + 1] = value
    return fitted


def _fit_binned_isotonic(
    sequence_stats: dict[str, BinnedStats], train_sequences: Sequence[str], weighting: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    size = next(iter(sequence_stats.values())).pixel_count.size
    total_weight = np.zeros(size, dtype=np.float64)
    weighted_sum_raw = np.zeros(size, dtype=np.float64)
    weighted_sum_gt = np.zeros(size, dtype=np.float64)
    sequence_input_weights = {}
    for sequence in train_sequences:
        stats = sequence_stats[sequence]
        if weighting == "pixel_equal":
            denominator = float(stats.valid_pixel_count)
            weights = stats.pixel_count.astype(np.float64)
            sum_raw = stats.pixel_sum_raw
            sum_gt = stats.pixel_sum_gt
        elif weighting == "frame_balanced":
            denominator = float(stats.frame_count)
            weights = stats.frame_weight
            sum_raw = stats.frame_weighted_sum_raw
            sum_gt = stats.frame_weighted_sum_gt
        else:
            raise ValueError(f"unsupported nonlinear weighting: {weighting}")
        if denominator <= 0.0:
            raise ValueError(f"no nonlinear fitting data for sequence {sequence}")
        # Normalize each training sequence to total weight one. Pixel-equal keeps
        # pixels equal within a sequence; frame-balanced additionally gives every
        # frame equal total weight within its sequence.
        total_weight += weights / denominator
        weighted_sum_raw += sum_raw / denominator
        weighted_sum_gt += sum_gt / denominator
        sequence_input_weights[sequence] = float(np.sum(weights / denominator))
    occupied = total_weight > 0.0
    if np.count_nonzero(occupied) < 2:
        raise ValueError("isotonic calibration requires at least two occupied raw-depth bins")
    x_knots = weighted_sum_raw[occupied] / total_weight[occupied]
    bin_gt_means = weighted_sum_gt[occupied] / total_weight[occupied]
    y_knots = _weighted_pava(bin_gt_means, total_weight[occupied])
    if np.any(np.diff(x_knots) <= 0.0) or np.any(np.diff(y_knots) < -1e-12):
        raise RuntimeError("isotonic knots are not monotonic")
    diagnostics: dict[str, object] = {
        "sampled_dense_pixel_count": int(sum(
            sequence_stats[sequence].valid_pixel_count for sequence in train_sequences
        )),
        "occupied_bin_count": int(x_knots.size),
        "sequence_normalized_input_weights": sequence_input_weights,
        "raw_depth_range_m": [float(x_knots[0]), float(x_knots[-1])],
        "calibrated_depth_range_m": [float(y_knots[0]), float(y_knots[-1])],
    }
    return x_knots, y_knots, diagnostics


def _nonlinear_name(range_name: str, weighting: str) -> str:
    prefix = "dense_isotonic" if range_name == "all" else "dense_isotonic_gt_le_5m"
    return f"{prefix}_{weighting}"


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


def _transitions(gt: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict[str, int]:
    actual = gt < 2.0
    raw_positive = raw < 2.0
    calibrated_positive = calibrated < 2.0
    return {
        "fp_to_tn": int(np.sum(~actual & raw_positive & ~calibrated_positive)),
        "tp_to_fn": int(np.sum(actual & raw_positive & ~calibrated_positive)),
        "fn_to_tp": int(np.sum(actual & ~raw_positive & calibrated_positive)),
        "tn_to_fp": int(np.sum(~actual & ~raw_positive & calibrated_positive)),
    }


def _candidate_result(
    predictions: Sequence[dict[str, object]], range_name: str,
    calibration: str, feature: str,
) -> dict[str, object]:
    gt = np.asarray([row["gt_distance"] for row in predictions], dtype=np.float64)
    raw = np.asarray([row[f"raw_{feature}"] for row in predictions], dtype=np.float64)
    calibrated = np.asarray(
        [row[f"{range_name}:{calibration}:{feature}"] for row in predictions],
        dtype=np.float64,
    )
    boundary = (gt >= 1.5) & (gt <= 2.5)
    return {
        "metrics": _metric_pair(gt, calibrated),
        "changes_from_raw": _transitions(gt, raw, calibrated),
        "boundary_changes_from_raw": _transitions(
            gt[boundary], raw[boundary], calibrated[boundary]
        ),
    }


def _named_candidate_result(
    predictions: Sequence[dict[str, object]], candidate: str, feature: str,
) -> dict[str, object]:
    gt = np.asarray([row["gt_distance"] for row in predictions], dtype=np.float64)
    raw = np.asarray([row[f"raw_{feature}"] for row in predictions], dtype=np.float64)
    calibrated = np.asarray(
        [row[f"{candidate}:{feature}"] for row in predictions], dtype=np.float64
    )
    boundary = (gt >= 1.5) & (gt <= 2.5)
    return {
        "metrics": _metric_pair(gt, calibrated),
        "changes_from_raw": _transitions(gt, raw, calibrated),
        "boundary_changes_from_raw": _transitions(
            gt[boundary], raw[boundary], calibrated[boundary]
        ),
    }


def _selection_key(candidate: dict[str, object], boundary: bool) -> tuple[float, float, float]:
    subset = "boundary_1p5_to_2p5" if boundary else "all"
    metrics = candidate["result"]["metrics"][subset]  # type: ignore[index]
    mae = float(metrics["mae"]) if metrics["mae"] is not None else math.inf
    rmse = float(metrics["rmse"]) if metrics["rmse"] is not None else math.inf
    return float(metrics["f1"]), -mae, -rmse


def _synthetic_self_test() -> int:
    pooled = _weighted_pava(
        np.asarray([1.0, 3.0, 2.0, 4.0]), np.ones(4, dtype=np.float64)
    )
    if not np.allclose(pooled, np.asarray([1.0, 2.5, 2.5, 4.0])):
        raise AssertionError(f"weighted PAVA pooling failed: {pooled}")
    pred = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    gt = 1.25 * pred
    stats = DenseStats()
    stats.add_arrays(pred, gt)
    scale = _fit_scale(stats)
    if not math.isclose(scale, 1.25, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"expected scale 1.25, got {scale}")
    affine_a, affine_b = _fit_affine(stats)
    if not math.isclose(affine_a, 1.25, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"expected affine a 1.25, got {affine_a}")
    if not math.isclose(affine_b, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"expected affine b 0, got {affine_b}")
    nonlinear_stats = {sequence: BinnedStats.empty() for sequence in SEQUENCES[:3]}
    nonlinear_raw = np.linspace(0.1, 6.0, 2000, dtype=np.float64)
    nonlinear_gt = nonlinear_raw + 0.08 * np.square(nonlinear_raw)
    for sequence, splits in zip(SEQUENCES[:3], (4, 5, 7)):
        for raw_frame, gt_frame in zip(
            np.array_split(nonlinear_raw, splits), np.array_split(nonlinear_gt, splits)
        ):
            nonlinear_stats[sequence].add_frame(raw_frame, gt_frame)
    nonlinear_errors = {}
    for weighting in NONLINEAR_WEIGHTINGS:
        x_knots, y_knots, _ = _fit_binned_isotonic(
            nonlinear_stats, SEQUENCES[:3], weighting
        )
        probes = np.asarray(RAW_PROBES_M, dtype=np.float64)
        expected = probes + 0.08 * np.square(probes)
        learned = np.interp(probes, x_knots, y_knots)
        error = float(np.max(np.abs(learned - expected)))
        if error > 0.02:
            raise AssertionError(
                f"nonlinear {weighting} max probe error {error} exceeds 0.02m"
            )
        nonlinear_errors[weighting] = error
    print(f"synthetic_self_test=PASS scale={scale:.12f} "
          f"affine_a={affine_a:.12f} affine_b={affine_b:.12f} "
          f"nonlinear_max_errors={json.dumps(nonlinear_errors, sort_keys=True)}")
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
    sequence_stats_by_range = {
        range_name: {sequence: DenseStats() for sequence in SEQUENCES}
        for range_name, _ in GT_RANGES
    }
    nonlinear_stats_by_range = {
        range_name: {sequence: BinnedStats.empty() for sequence in SEQUENCES}
        for range_name, _ in NONLINEAR_RANGES
    }
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
            range_counts = {}
            for range_name, upper_bound in GT_RANGES:
                selected = (np.ones(gt_values.shape, dtype=bool) if upper_bound is None
                            else gt_values <= upper_bound)
                range_pred = pred_values[selected]
                range_gt = gt_values[selected]
                range_counts[range_name] = int(range_pred.size)
                if range_pred.size:
                    sequence_stats_by_range[range_name][item.row.sequence].add_arrays(
                        range_pred, range_gt
                    )
            for range_name, upper_bound in NONLINEAR_RANGES:
                selected = (np.ones(gt_values.shape, dtype=bool) if upper_bound is None
                            else gt_values <= upper_bound)
                nonlinear_stats_by_range[range_name][item.row.sequence].add_frame(
                    pred_values[selected], gt_values[selected]
                )
            print(f"dense_fit [{index}/{len(matched)}] {item.row.sequence}:"
                  f"{item.row.source_index} pixels={json.dumps(range_counts, sort_keys=True)}",
                  flush=True)

        folds_by_range: dict[str, list[dict[str, object]]] = {}
        parameters: dict[str, dict[str, dict[str, float]]] = {}
        for range_name, upper_bound in GT_RANGES:
            range_folds = []
            parameters[range_name] = {}
            for holdout in SEQUENCES:
                train_sequences = [sequence for sequence in SEQUENCES if sequence != holdout]
                train_stats = DenseStats()
                for sequence in train_sequences:
                    train_stats.add_stats(sequence_stats_by_range[range_name][sequence])
                scale_a = _fit_scale(train_stats)
                affine_a, affine_b = _fit_affine(train_stats)
                parameters[range_name][holdout] = {
                    "scale_a": scale_a, "affine_a": affine_a, "affine_b": affine_b,
                }
                range_folds.append({
                    "gt_range": range_name,
                    "fit_gt_depth_upper_bound_m": upper_bound,
                    "train_sequences": train_sequences,
                    "holdout_sequence": holdout,
                    "train_dense_pixel_count": train_stats.n,
                    "holdout_dense_pixel_count":
                        sequence_stats_by_range[range_name][holdout].n,
                    "scale_a": scale_a, "affine_a": affine_a, "affine_b": affine_b,
                    "train_sufficient_statistics": train_stats.as_dict(),
                })
            folds_by_range[range_name] = range_folds

        nonlinear_folds: dict[str, dict[str, list[dict[str, object]]]] = {
            range_name: {weighting: [] for weighting in NONLINEAR_WEIGHTINGS}
            for range_name, _ in NONLINEAR_RANGES
        }
        nonlinear_mappings: dict[
            str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
        ] = {
            range_name: {weighting: {} for weighting in NONLINEAR_WEIGHTINGS}
            for range_name, _ in NONLINEAR_RANGES
        }
        for range_name, upper_bound in NONLINEAR_RANGES:
            for weighting in NONLINEAR_WEIGHTINGS:
                for holdout in SEQUENCES:
                    train_sequences = [sequence for sequence in SEQUENCES if sequence != holdout]
                    x_knots, y_knots, diagnostics = _fit_binned_isotonic(
                        nonlinear_stats_by_range[range_name], train_sequences, weighting
                    )
                    nonlinear_mappings[range_name][weighting][holdout] = (x_knots, y_knots)
                    probes = np.asarray(RAW_PROBES_M, dtype=np.float64)
                    probe_values = np.interp(probes, x_knots, y_knots)
                    fold = {
                        "gt_range": range_name,
                        "fit_gt_depth_upper_bound_m": upper_bound,
                        "weighting": weighting,
                        "candidate": _nonlinear_name(range_name, weighting),
                        "holdout_sequence": holdout,
                        "train_sequences": train_sequences,
                        **diagnostics,
                        "number_of_knots": int(x_knots.size),
                        "x_knots": x_knots.tolist(),
                        "y_knots": y_knots.tolist(),
                        "fixed_raw_depth_probes_m": {
                            f"{probe:g}": float(value)
                            for probe, value in zip(probes, probe_values)
                        },
                    }
                    nonlinear_folds[range_name][weighting].append(fold)
                    print(
                        f"nonlinear_fold holdout={holdout} train={train_sequences} "
                        f"range={range_name} weighting={weighting} "
                        f"pixels={diagnostics['sampled_dense_pixel_count']} "
                        f"bins={diagnostics['occupied_bin_count']} knots={x_knots.size} "
                        f"probes={json.dumps(fold['fixed_raw_depth_probes_m'], sort_keys=True)}",
                        flush=True,
                    )

        predictions: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            raw = infer_da2(engine, _read_rgb(row.image_path, cv2))
            raw_features = _features(raw)
            output: dict[str, object] = {
                "holdout_sequence": row.sequence, "sequence": row.sequence,
                "source_index": row.source_index, "image_path": str(row.image_path),
                "gt_distance": row.gt_distance,
            }
            output.update({f"raw_{name}": value for name, value in raw_features.items()})
            for range_name, _ in GT_RANGES:
                fold_parameters = parameters[range_name][row.sequence]
                maps = {
                    "dense_scale": fold_parameters["scale_a"] * raw,
                    "dense_affine": (
                        fold_parameters["affine_a"] * raw + fold_parameters["affine_b"]
                    ),
                }
                for calibration, calibrated in maps.items():
                    calibrated_features = _features(calibrated)
                    output.update({
                        f"{range_name}:{calibration}:{name}": value
                        for name, value in calibrated_features.items()
                    })
            for range_name, _ in NONLINEAR_RANGES:
                for weighting in NONLINEAR_WEIGHTINGS:
                    x_knots, y_knots = nonlinear_mappings[range_name][weighting][row.sequence]
                    calibrated = np.interp(raw, x_knots, y_knots)
                    candidate = _nonlinear_name(range_name, weighting)
                    output.update({
                        f"{candidate}:{name}": value
                        for name, value in _features(calibrated).items()
                    })
            all_parameters = parameters["all"][row.sequence]
            output["scale_a"] = all_parameters["scale_a"]
            output.update({
                f"dense_scale_{feature}": output[f"all:dense_scale:{feature}"]
                for feature in FEATURES
            })
            predictions.append(output)
            print(f"holdout_eval [{index}/{len(rows)}] {row.sequence}:{row.source_index}",
                  flush=True)
    finally:
        engine.close()

    for range_name, _ in GT_RANGES:
        for fold in folds_by_range[range_name]:
            fold_rows = [row for row in predictions
                         if row["sequence"] == fold["holdout_sequence"]]
            fold["raw_metrics"] = {
                feature: _metric_pair(
                    np.asarray([row["gt_distance"] for row in fold_rows], dtype=np.float64),
                    np.asarray([row[f"raw_{feature}"] for row in fold_rows], dtype=np.float64),
                ) for feature in FEATURES
            }
            fold["calibrated_results"] = {
                calibration: {
                    feature: _candidate_result(
                        fold_rows, range_name, calibration, feature
                    ) for feature in FEATURES
                } for calibration in CALIBRATIONS
            }
            if range_name == "all":
                fold["scaled_metrics"] = {
                    feature: fold["calibrated_results"]["dense_scale"][feature]["metrics"]
                    for feature in FEATURES
                }

    for range_name, _ in NONLINEAR_RANGES:
        for weighting in NONLINEAR_WEIGHTINGS:
            candidate = _nonlinear_name(range_name, weighting)
            for fold in nonlinear_folds[range_name][weighting]:
                fold_rows = [row for row in predictions
                             if row["sequence"] == fold["holdout_sequence"]]
                fold["results"] = {
                    feature: _named_candidate_result(fold_rows, candidate, feature)
                    for feature in FEATURES
                }

    folds = folds_by_range["all"]

    prediction_path = output_dir / "da2_dense_scale_predictions.csv"
    fields = ["holdout_sequence", "sequence", "source_index", "image_path",
              "gt_distance", "scale_a",
              *[f"raw_{feature}" for feature in FEATURES],
              *[f"dense_scale_{feature}" for feature in FEATURES]]
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
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
            sequence: sequence_stats_by_range["all"][sequence].as_dict()
            for sequence in SEQUENCES
        },
        "folds": folds,
        "aggregate_out_of_fold_metrics": _calculate_metrics(predictions),
    }
    report_path = output_dir / "da2_dense_scale_report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")

    candidates = []
    aggregate_results: dict[str, dict[str, dict[str, object]]] = {}
    for range_name, upper_bound in GT_RANGES:
        aggregate_results[range_name] = {}
        for calibration in CALIBRATIONS:
            aggregate_results[range_name][calibration] = {}
            for feature in FEATURES:
                result = _candidate_result(predictions, range_name, calibration, feature)
                aggregate_results[range_name][calibration][feature] = result
                candidates.append({
                    "gt_range": range_name,
                    "fit_gt_depth_upper_bound_m": upper_bound,
                    "calibration": calibration,
                    "feature": feature,
                    "result": result,
                })
    best_overall = max(candidates, key=lambda candidate: _selection_key(candidate, False))
    best_boundary = max(candidates, key=lambda candidate: _selection_key(candidate, True))
    near_report = {
        "experiment_type": "da2_dense_near_scale_affine_strict_loso",
        "trt_engine_path": str(engine_path),
        "trt_engine_sha256": _sha256(engine_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "depth_scale": args.depth_scale,
        "max_timestamp_delta_sec": args.max_timestamp_delta,
        "roi_rows": list(ROI_ROWS), "roi_cols": list(ROI_COLS),
        "preprocessing": PREPROCESSING,
        "fit_gt_ranges": [
            {"name": range_name, "gt_depth_upper_bound_m": upper_bound}
            for range_name, upper_bound in GT_RANGES
        ],
        "fit_range_only_limits_calibration_pixels": True,
        "holdout_evaluation_scope": "all frames in each complete held-out sequence",
        "scale_formula": "a = sum(pred * gt) / sum(pred^2); D_cal = a * D_raw",
        "affine_formula": "least squares D_cal = a * D_raw + b; a > 0",
        "threshold_rule": "gt_distance < 2.0; pred_distance < 2.0",
        "boundary_subset": "1.5 <= gt_distance <= 2.5",
        "alignment": {
            "frames": len(alignments), "matched": len(matched),
            "unmatched": len(alignments) - len(matched),
            "max_matched_delta_sec": max(matched_deltas) if matched_deltas else None,
            "mean_matched_delta_sec": float(np.mean(matched_deltas)) if matched_deltas else None,
        },
        "sequence_dense_statistics_by_gt_range": {
            range_name: {
                sequence: sequence_stats_by_range[range_name][sequence].as_dict()
                for sequence in SEQUENCES
            } for range_name, _ in GT_RANGES
        },
        "folds_by_gt_range": folds_by_range,
        "aggregate_out_of_fold_results": aggregate_results,
        "change_definitions": ["fp_to_tn", "tp_to_fn", "fn_to_tp", "tn_to_fp"],
        "best_overall": best_overall,
        "best_boundary": best_boundary,
        "selection_policy": (
            "best_overall uses complete held-out aggregate F1, then MAE/RMSE; "
            "best_boundary is diagnostic only"
        ),
    }
    near_report_path = output_dir / "da2_dense_near_calibration_report.json"
    with near_report_path.open("w", encoding="utf-8") as stream:
        json.dump(near_report, stream, indent=2, allow_nan=False)
        stream.write("\n")

    nonlinear_candidates = []
    nonlinear_aggregate: dict[str, dict[str, object]] = {}
    gt_all = np.asarray([row["gt_distance"] for row in predictions], dtype=np.float64)
    boundary_all = (gt_all >= 1.5) & (gt_all <= 2.5)
    raw_results = {}
    global_scale_results = {}
    for feature in FEATURES:
        raw_values = np.asarray(
            [row[f"raw_{feature}"] for row in predictions], dtype=np.float64
        )
        raw_result = {
            "metrics": _metric_pair(gt_all, raw_values),
            "changes_from_raw": _transitions(gt_all, raw_values, raw_values),
            "boundary_changes_from_raw": _transitions(
                gt_all[boundary_all], raw_values[boundary_all], raw_values[boundary_all]
            ),
        }
        raw_results[feature] = raw_result
        nonlinear_candidates.append({
            "candidate": "raw", "feature": feature, "result": raw_result,
        })
        scale_result = _candidate_result(
            predictions, "all", "dense_scale", feature
        )
        global_scale_results[feature] = scale_result
        nonlinear_candidates.append({
            "candidate": "global_dense_scale", "feature": feature,
            "result": scale_result,
        })
    nonlinear_aggregate["raw"] = raw_results
    nonlinear_aggregate["global_dense_scale"] = global_scale_results
    for range_name, _ in NONLINEAR_RANGES:
        for weighting in NONLINEAR_WEIGHTINGS:
            candidate = _nonlinear_name(range_name, weighting)
            candidate_results = {
                feature: _named_candidate_result(predictions, candidate, feature)
                for feature in FEATURES
            }
            nonlinear_aggregate[candidate] = candidate_results
            for feature, result in candidate_results.items():
                nonlinear_candidates.append({
                    "candidate": candidate, "gt_range": range_name,
                    "weighting": weighting, "feature": feature, "result": result,
                })
    nonlinear_best_overall = max(
        nonlinear_candidates, key=lambda candidate: _selection_key(candidate, False)
    )
    nonlinear_best_boundary = max(
        nonlinear_candidates, key=lambda candidate: _selection_key(candidate, True)
    )
    nonlinear_report = {
        "experiment_type": "da2_dense_nonlinear_isotonic_strict_loso",
        "trt_engine_path": str(engine_path),
        "trt_engine_sha256": _sha256(engine_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "depth_scale": args.depth_scale,
        "max_timestamp_delta_sec": args.max_timestamp_delta,
        "roi_rows": list(ROI_ROWS), "roi_cols": list(ROI_COLS),
        "preprocessing": PREPROCESSING,
        "mapping": "weighted PAVA; D_cal = interp(D_raw, x_knots, y_knots)",
        "fixed_raw_bin_width_m": RAW_BIN_WIDTH_M,
        "fixed_raw_bin_max_m": RAW_BIN_MAX_M,
        "fixed_raw_depth_probes_m": list(RAW_PROBES_M),
        "fit_scopes": [
            {"name": range_name, "gt_depth_upper_bound_m": upper_bound}
            for range_name, upper_bound in NONLINEAR_RANGES
        ],
        "weightings": {
            "pixel_equal": (
                "pixels equal within each sequence; each train sequence normalized "
                "to total weight one"
            ),
            "frame_balanced": (
                "pixels normalized to total weight one per frame, then frames normalized "
                "within sequence and each train sequence to total weight one"
            ),
        },
        "holdout_evaluation_scope": "all frames in each complete held-out sequence",
        "threshold_rule": "gt_distance < 2.0; pred_distance < 2.0",
        "boundary_subset": "1.5 <= gt_distance <= 2.5",
        "alignment": near_report["alignment"],
        "sequence_binned_statistics": {
            range_name: {
                sequence: nonlinear_stats_by_range[range_name][sequence].summary()
                for sequence in SEQUENCES
            } for range_name, _ in NONLINEAR_RANGES
        },
        "folds": nonlinear_folds,
        "aggregate_out_of_fold_results": nonlinear_aggregate,
        "best_overall": nonlinear_best_overall,
        "best_boundary": nonlinear_best_boundary,
        "selection_policy": (
            "best_overall includes raw and global dense scale baselines and uses complete "
            "held-out aggregate F1, then MAE/RMSE; best_boundary is diagnostic only"
        ),
    }
    nonlinear_report_path = output_dir / "da2_dense_nonlinear_calibration_report.json"
    with nonlinear_report_path.open("w", encoding="utf-8") as stream:
        json.dump(nonlinear_report, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print(f"report={report_path}")
    print(f"near_report={near_report_path}")
    print(f"nonlinear_report={nonlinear_report_path}")
    print(f"predictions={prediction_path}")
    print(f"alignment={alignment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
