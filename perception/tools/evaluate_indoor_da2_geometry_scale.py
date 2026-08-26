#!/usr/bin/env python3
"""Compare frozen production Geometry on raw vs LOSO-scaled DA2 depth maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


EXPECTED_CALIBRATION_SHA256 = (
    "e553a61ff91138980b9b26496454205f4f807fa22267900014374b1ae9f66642"
)
SEQUENCES = ("slam1", "slam2", "slam3", "pioneer_360")
FEATURES = ("min", "p10")
DEFAULT_ENGINE = "/models/obstacle/depth_anything_v2_metric_hypersim_vits_int8.trt"
DEFAULT_CALIBRATION = "/models/obstacle/calib_tum_compliant_isotonic_rescue.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline strict-LOSO comparison of raw and scaled DA2 Geometry input."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--da2-engine", default=DEFAULT_ENGINE)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--max-timestamp-delta", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--expected-frame-count", type=int, default=1100)
    parser.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_self_test and (args.manifest is None or args.output_dir is None):
        parser.error("--manifest and --output-dir are required unless --synthetic-self-test is used")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_production_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    calibration_path = path.expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"production calibration not found: {calibration_path}")
    digest = _sha256(calibration_path)
    if digest != EXPECTED_CALIBRATION_SHA256:
        raise ValueError(
            f"production calibration SHA256 mismatch: expected "
            f"{EXPECTED_CALIBRATION_SHA256}, got {digest}: {calibration_path}"
        )
    with calibration_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    missing = {"x_knots", "y_knots"}.difference(payload)
    if missing:
        raise ValueError(f"production calibration missing keys: {sorted(missing)}")
    x_knots = np.asarray(payload["x_knots"], dtype=np.float64)
    y_knots = np.asarray(payload["y_knots"], dtype=np.float64)
    if x_knots.ndim != 1 or y_knots.ndim != 1 or x_knots.size != y_knots.size:
        raise ValueError("production x_knots/y_knots must be equal-length 1-D arrays")
    if x_knots.size < 2 or np.any(~np.isfinite(x_knots)) or np.any(~np.isfinite(y_knots)):
        raise ValueError("production x_knots/y_knots must contain at least two finite values")
    if np.any(np.diff(x_knots) <= 0.0):
        raise ValueError("production x_knots must be strictly increasing")
    return x_knots, y_knots, digest


def _load_helpers() -> tuple[object, object, object]:
    tools_dir = Path(__file__).resolve().parent
    perception_dir = tools_dir.parent
    for directory in (tools_dir, perception_dir):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    dense = importlib.import_module("fit_indoor_da2_dense_calibration")
    benchmark = importlib.import_module("benchmark_indoor_depth_models")
    obstacle = importlib.import_module("plugins.obstacle")
    if tuple(dense.SEQUENCES) != SEQUENCES:
        raise RuntimeError(f"LOSO sequences changed: {dense.SEQUENCES}")
    return dense, benchmark, obstacle


def _scalar_prediction(
    raw_depth: np.ndarray, x_knots: np.ndarray, y_knots: np.ndarray, obstacle: object,
) -> dict[str, float | bool]:
    roi = raw_depth[
        obstacle.INDOOR_ROI_ROWS[0]:obstacle.INDOOR_ROI_ROWS[1],
        obstacle.INDOOR_ROI_COLS[0]:obstacle.INDOOR_ROI_COLS[1],
    ]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if valid.size == 0:
        return {
            "fallback": True, "raw_min": float("nan"), "p10": float("nan"),
            "gap": float("nan"), "pred_iso": float(obstacle.INDOOR_CLIP[1]),
            "rescued": False, "prediction": float(obstacle.INDOOR_CLIP[1]),
        }
    raw_min = float(valid.min())
    p10 = float(np.percentile(valid, 10.0))
    pred_iso = float(np.interp(raw_min, x_knots, y_knots))
    pred_iso = float(np.clip(pred_iso, obstacle.INDOOR_CLIP[0], obstacle.INDOOR_CLIP[1]))
    rescued = (
        pred_iso >= obstacle.DECISION_THRESHOLD
        and raw_min >= obstacle.RAW_BASELINE_THRESHOLD
        and raw_min < obstacle.RESCUE_UPPER_BOUND
        and (p10 - raw_min) < obstacle.RESCUE_GAP_THRESHOLD
    )
    prediction = obstacle.RESCUE_DISTANCE if rescued else pred_iso
    prediction = float(np.clip(
        prediction, obstacle.INDOOR_CLIP[0], obstacle.INDOOR_CLIP[1]
    ))
    return {
        "fallback": False, "raw_min": raw_min, "p10": p10, "gap": p10 - raw_min,
        "pred_iso": pred_iso, "rescued": rescued, "prediction": prediction,
    }


def _geometry_decision(
    scalar_prediction: float, rescued: bool, gap: float,
    geometry: tuple[float, float] | None, obstacle: object,
) -> dict[str, float | bool | None]:
    if scalar_prediction >= obstacle.DECISION_THRESHOLD:
        return {
            "ran": False, "success": False, "vetoed": False,
            "geometry_p1": None, "floor_inlier_ratio": None,
            "prediction": scalar_prediction,
        }
    if geometry is None:
        return {
            "ran": True, "success": False, "vetoed": False,
            "geometry_p1": None, "floor_inlier_ratio": None,
            "prediction": scalar_prediction,
        }
    geometry_p1, floor_inlier_ratio = geometry
    vetoed = (
        geometry_p1 >= obstacle.GEOMETRY_VETO_P1
        and floor_inlier_ratio >= obstacle.GEOMETRY_VETO_FLOOR_INLIER_RATIO
    )
    high_veto = rescued and gap >= 0.25 and floor_inlier_ratio >= 0.985
    low_veto = (
        rescued and geometry_p1 >= 2.03
        and floor_inlier_ratio <= 0.80 and gap >= 0.26
    )
    vetoed = bool(vetoed or high_veto or low_veto)
    prediction = obstacle.GEOMETRY_VETO_DISTANCE if vetoed else scalar_prediction
    return {
        "ran": True, "success": True, "vetoed": vetoed,
        "geometry_p1": float(geometry_p1),
        "floor_inlier_ratio": float(floor_inlier_ratio),
        "prediction": float(prediction),
    }


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, int | float | None]:
    if gt.size == 0:
        return {
            "count": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "mae": None, "rmse": None,
        }
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


def _transitions(gt: np.ndarray, source: np.ndarray, target: np.ndarray) -> dict[str, int]:
    actual = gt < 2.0
    source_positive = source < 2.0
    target_positive = target < 2.0
    return {
        "fp_to_tn": int(np.sum(~actual & source_positive & ~target_positive)),
        "tp_to_fn": int(np.sum(actual & source_positive & ~target_positive)),
        "fn_to_tp": int(np.sum(actual & ~source_positive & target_positive)),
        "tn_to_fp": int(np.sum(~actual & ~source_positive & target_positive)),
    }


def _transition_pair(gt: np.ndarray, source: np.ndarray, target: np.ndarray) -> dict[str, object]:
    boundary = (gt >= 1.5) & (gt <= 2.5)
    return {
        "all": _transitions(gt, source, target),
        "boundary_1p5_to_2p5": _transitions(
            gt[boundary], source[boundary], target[boundary]
        ),
    }


def _run_production_geometry(
    depth: np.ndarray, obstacle: object,
) -> tuple[tuple[float, float] | None, str | None]:
    try:
        return obstacle.ObstaclePlugin._indoor_geometry(depth), None
    except Exception as exc:  # production is fail-open; retain the failure for diagnostics
        return None, f"{type(exc).__name__}: {exc}"


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "p1": None, "p5": None, "median": None,
                "mean": None, "p95": None, "max": None}
    percentiles = np.percentile(array, [1.0, 5.0, 50.0, 95.0])
    return {
        "count": int(array.size), "min": float(array.min()),
        "p1": float(percentiles[0]), "p5": float(percentiles[1]),
        "median": float(percentiles[2]), "mean": float(array.mean()),
        "p95": float(percentiles[3]), "max": float(array.max()),
    }


def _geometry_summary(rows: Sequence[dict[str, object]], prefix: str) -> dict[str, object]:
    ran = [row for row in rows if row[f"{prefix}_ran"]]
    successes = [row for row in ran if row[f"{prefix}_success"]]
    return {
        "run_count": len(ran),
        "success_count": len(successes),
        "failure_count": len(ran) - len(successes),
        "exception_count": sum(bool(row[f"{prefix}_error"]) for row in ran),
        "veto_count": sum(bool(row[f"{prefix}_vetoed"]) for row in ran),
        "geometry_p1_distribution": _distribution([
            float(row[f"{prefix}_geometry_p1"]) for row in successes
        ]),
        "floor_inlier_ratio_distribution": _distribution([
            float(row[f"{prefix}_floor_inlier_ratio"]) for row in successes
        ]),
        "camera_height_distribution": None,
        "camera_height_note": (
            "production ObstaclePlugin._indoor_geometry returns only geometry_p1 and "
            "floor_inlier_ratio; camera height is not exposed"
        ),
    }


def _engine_sha256_if_readable(path: Path) -> str | None:
    try:
        return _sha256(path)
    except OSError:
        return None


def _synthetic_self_test() -> int:
    class FakeObstacle:
        DECISION_THRESHOLD = 2.0
        GEOMETRY_VETO_P1 = 1.80
        GEOMETRY_VETO_FLOOR_INLIER_RATIO = 0.985
        GEOMETRY_VETO_DISTANCE = 2.01

    stats = importlib.import_module("fit_indoor_da2_dense_calibration").DenseStats()
    pred = np.asarray([0.5, 1.0, 2.0, 4.0])
    stats.add_arrays(pred, 1.25 * pred)
    scale = stats.sum_xy / stats.sum_x2
    if not math.isclose(scale, 1.25, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"expected LOSO scale 1.25, got {scale}")
    veto = _geometry_decision(1.9, False, 0.1, (1.9, 0.99), FakeObstacle)
    if not veto["vetoed"] or veto["prediction"] != 2.01:
        raise AssertionError(f"frozen base veto logic failed: {veto}")
    no_veto = _geometry_decision(1.9, False, 0.1, (1.7, 0.99), FakeObstacle)
    if no_veto["vetoed"] or no_veto["prediction"] != 1.9:
        raise AssertionError(f"frozen non-veto logic failed: {no_veto}")
    print("synthetic_self_test=PASS scale=1.250000000000 geometry_veto=PASS")
    return 0


def main() -> int:
    args = _parse_args()
    if args.synthetic_self_test:
        tools_dir = Path(__file__).resolve().parent
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        return _synthetic_self_test()

    calibration_path = Path(args.calibration).expanduser().resolve()
    x_knots, y_knots, calibration_sha256 = _load_production_calibration(calibration_path)
    dense, benchmark, obstacle = _load_helpers()
    manifest_path = args.manifest.expanduser().resolve()
    rows = dense._load_manifest(manifest_path, args.image_root)
    if len(rows) != args.expected_frame_count:
        raise ValueError(
            f"expected fixed TUM{args.expected_frame_count}, got {len(rows)} manifest frames"
        )
    alignments = dense._align_rows(rows, args.max_timestamp_delta)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_path = output_dir / "da2_geometry_scale_alignment_report.csv"
    dense._write_alignment_report(alignment_path, alignments)

    engine_path = Path(args.da2_engine).expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"DA2 TensorRT engine not found: {engine_path}")
    engine = benchmark._TrtEngine(str(engine_path))
    sequence_stats = {sequence: dense.DenseStats() for sequence in SEQUENCES}
    matched = [item for item in alignments if item.matched]
    try:
        first_bgr = dense._read_rgb(rows[0].image_path, benchmark.cv2)
        for _ in range(max(0, args.warmup)):
            benchmark.infer_da2(engine, first_bgr)
        for index, item in enumerate(matched, start=1):
            if item.depth_path is None:
                raise RuntimeError("matched alignment has no depth path")
            raw = benchmark.infer_da2(
                engine, dense._read_rgb(item.row.image_path, benchmark.cv2)
            )
            gt_depth = dense._read_tum_depth(
                item.depth_path, args.depth_scale, benchmark.cv2
            )
            pred_values, gt_values = dense._valid_dense_pairs(raw, gt_depth)
            sequence_stats[item.row.sequence].add_arrays(pred_values, gt_values)
            print(
                f"dense_scale_fit [{index}/{len(matched)}] {item.row.sequence}:"
                f"{item.row.source_index} pixels={pred_values.size}", flush=True,
            )

        folds = {}
        for holdout in SEQUENCES:
            train_sequences = [sequence for sequence in SEQUENCES if sequence != holdout]
            train_stats = dense.DenseStats()
            for sequence in train_sequences:
                train_stats.add_stats(sequence_stats[sequence])
            scale = dense._fit_scale(train_stats)
            folds[holdout] = {
                "holdout_sequence": holdout, "train_sequences": train_sequences,
                "train_dense_pixel_count": train_stats.n,
                "holdout_dense_pixel_count": sequence_stats[holdout].n,
                "scale_a": scale,
            }

        predictions: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            raw = benchmark.infer_da2(
                engine, dense._read_rgb(row.image_path, benchmark.cv2)
            )
            scalar = _scalar_prediction(raw, x_knots, y_knots, obstacle)
            scalar_prediction = float(scalar["prediction"])
            scale = float(folds[row.sequence]["scale_a"])
            if scalar_prediction < obstacle.DECISION_THRESHOLD:
                raw_geometry, raw_geometry_error = _run_production_geometry(raw, obstacle)
                scaled_geometry, scaled_geometry_error = _run_production_geometry(
                    scale * raw, obstacle
                )
            else:
                raw_geometry = None
                scaled_geometry = None
                raw_geometry_error = None
                scaled_geometry_error = None
            raw_decision = _geometry_decision(
                scalar_prediction, bool(scalar["rescued"]), float(scalar["gap"]),
                raw_geometry, obstacle,
            )
            scaled_decision = _geometry_decision(
                scalar_prediction, bool(scalar["rescued"]), float(scalar["gap"]),
                scaled_geometry, obstacle,
            )
            output: dict[str, object] = {
                "sequence": row.sequence, "source_index": row.source_index,
                "image_path": str(row.image_path), "gt_distance": row.gt_distance,
                "scale_a": scale, "raw_min": scalar["raw_min"], "p10": scalar["p10"],
                "pred_iso": scalar["pred_iso"], "rescued": scalar["rescued"],
                "scalar_only_pred": scalar_prediction,
                "raw_geometry_pred": raw_decision["prediction"],
                "scaled_geometry_pred": scaled_decision["prediction"],
            }
            for prefix, decision in (("raw_geometry", raw_decision),
                                     ("scaled_geometry", scaled_decision)):
                for key in ("ran", "success", "vetoed", "geometry_p1",
                            "floor_inlier_ratio"):
                    output[f"{prefix}_{key}"] = decision[key]
            output["raw_geometry_error"] = raw_geometry_error
            output["scaled_geometry_error"] = scaled_geometry_error
            predictions.append(output)
            print(
                f"geometry_eval [{index}/{len(rows)}] {row.sequence}:{row.source_index} "
                f"scalar={scalar_prediction:.6f} raw_veto={raw_decision['vetoed']} "
                f"scaled_veto={scaled_decision['vetoed']}", flush=True,
            )
    finally:
        engine.close()

    fields = [
        "sequence", "source_index", "image_path", "gt_distance", "scale_a",
        "raw_min", "p10", "pred_iso", "rescued", "scalar_only_pred",
        "raw_geometry_pred", "scaled_geometry_pred",
        *[f"{prefix}_{key}" for prefix in ("raw_geometry", "scaled_geometry")
          for key in ("ran", "success", "vetoed", "geometry_p1", "floor_inlier_ratio")],
        "raw_geometry_error", "scaled_geometry_error",
    ]
    prediction_path = output_dir / "da2_geometry_scale_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)

    gt = np.asarray([row["gt_distance"] for row in predictions], dtype=np.float64)
    scalar_pred = np.asarray([row["scalar_only_pred"] for row in predictions], dtype=np.float64)
    raw_pred = np.asarray([row["raw_geometry_pred"] for row in predictions], dtype=np.float64)
    scaled_pred = np.asarray(
        [row["scaled_geometry_pred"] for row in predictions], dtype=np.float64
    )
    variants = {
        "scalar_only": _metric_pair(gt, scalar_pred),
        "scalar_plus_raw_geometry": _metric_pair(gt, raw_pred),
        "scalar_plus_scaled_geometry": _metric_pair(gt, scaled_pred),
    }
    for name, target in (("scalar_plus_raw_geometry", raw_pred),
                         ("scalar_plus_scaled_geometry", scaled_pred)):
        variants[name]["changes_from_scalar_only"] = _transition_pair(
            gt, scalar_pred, target
        )
    variants["scalar_plus_scaled_geometry"]["changes_from_raw_geometry"] = _transition_pair(
        gt, raw_pred, scaled_pred
    )

    for holdout in SEQUENCES:
        fold_rows = [row for row in predictions if row["sequence"] == holdout]
        fold_gt = np.asarray([row["gt_distance"] for row in fold_rows], dtype=np.float64)
        fold_scalar = np.asarray(
            [row["scalar_only_pred"] for row in fold_rows], dtype=np.float64
        )
        fold_raw = np.asarray([row["raw_geometry_pred"] for row in fold_rows], dtype=np.float64)
        fold_scaled = np.asarray(
            [row["scaled_geometry_pred"] for row in fold_rows], dtype=np.float64
        )
        folds[holdout]["metrics"] = {
            "scalar_only": _metric_pair(fold_gt, fold_scalar),
            "scalar_plus_raw_geometry": _metric_pair(fold_gt, fold_raw),
            "scalar_plus_scaled_geometry": _metric_pair(fold_gt, fold_scaled),
        }
        folds[holdout]["changes"] = {
            "raw_geometry_from_scalar_only": _transition_pair(
                fold_gt, fold_scalar, fold_raw
            ),
            "scaled_geometry_from_scalar_only": _transition_pair(
                fold_gt, fold_scalar, fold_scaled
            ),
            "scaled_geometry_from_raw_geometry": _transition_pair(
                fold_gt, fold_raw, fold_scaled
            ),
        }
        folds[holdout]["geometry_diagnostics"] = {
            "raw_geometry": _geometry_summary(fold_rows, "raw_geometry"),
            "scaled_geometry": _geometry_summary(fold_rows, "scaled_geometry"),
        }

    raw_success = [row for row in predictions if row["raw_geometry_success"]]
    scaled_success = [row for row in predictions if row["scaled_geometry_success"]]
    both_success = [
        row for row in predictions
        if row["raw_geometry_success"] and row["scaled_geometry_success"]
    ]
    feature_diagnostic = {
        "both_success_count": len(both_success),
        "raw_success_scaled_failure_count": sum(
            bool(row["raw_geometry_success"] and not row["scaled_geometry_success"])
            for row in predictions
        ),
        "raw_failure_scaled_success_count": sum(
            bool(not row["raw_geometry_success"] and row["scaled_geometry_success"])
            for row in predictions
        ),
        "raw_geometry_p1_distribution": _distribution([
            float(row["raw_geometry_geometry_p1"]) for row in raw_success
        ]),
        "scaled_geometry_p1_distribution": _distribution([
            float(row["scaled_geometry_geometry_p1"]) for row in scaled_success
        ]),
        "scaled_minus_raw_geometry_p1_distribution": _distribution([
            float(row["scaled_geometry_geometry_p1"])
            - float(row["raw_geometry_geometry_p1"])
            for row in both_success
        ]),
        "scaled_minus_raw_floor_inlier_ratio_distribution": _distribution([
            float(row["scaled_geometry_floor_inlier_ratio"])
            - float(row["raw_geometry_floor_inlier_ratio"])
            for row in both_success
        ]),
    }
    report = {
        "experiment_type": "da2_loso_global_scale_geometry_only",
        "fixed_dataset": f"TUM{args.expected_frame_count}",
        "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "engine_path": str(engine_path),
        "engine_sha256": _engine_sha256_if_readable(engine_path),
        "production_calibration_path": str(calibration_path),
        "production_calibration_sha256": calibration_sha256,
        "expected_production_calibration_sha256": EXPECTED_CALIBRATION_SHA256,
        "production_calibration_knot_count": int(x_knots.size),
        "scalar_path": "raw_min -> production isotonic -> frozen production rescue",
        "geometry_parameters": "frozen by direct ObstaclePlugin._indoor_geometry reuse",
        "frozen_scalar_parameter_snapshot": {
            "raw_baseline_threshold_m": obstacle.RAW_BASELINE_THRESHOLD,
            "rescue_upper_bound_m": obstacle.RESCUE_UPPER_BOUND,
            "rescue_gap_threshold_m": obstacle.RESCUE_GAP_THRESHOLD,
            "rescue_distance_m": obstacle.RESCUE_DISTANCE,
            "decision_threshold_m": obstacle.DECISION_THRESHOLD,
        },
        "frozen_geometry_parameter_snapshot": {
            "geometry_min_depth_m": obstacle.GEOMETRY_MIN_DEPTH,
            "geometry_max_depth_m": obstacle.GEOMETRY_MAX_DEPTH,
            "floor_distance_threshold_m": obstacle.FLOOR_DISTANCE_THRESHOLD,
            "camera_height_min_m": obstacle.CAMERA_HEIGHT_MIN,
            "camera_height_max_m": obstacle.CAMERA_HEIGHT_MAX,
            "obstacle_height_min_m": obstacle.OBSTACLE_HEIGHT_MIN,
            "obstacle_height_max_m": obstacle.OBSTACLE_HEIGHT_MAX,
            "geometry_veto_p1_m": obstacle.GEOMETRY_VETO_P1,
            "geometry_veto_floor_inlier_ratio":
                obstacle.GEOMETRY_VETO_FLOOR_INLIER_RATIO,
            "geometry_veto_distance_m": obstacle.GEOMETRY_VETO_DISTANCE,
            "high_veto_gap_m": 0.25,
            "high_veto_floor_inlier_ratio": 0.985,
            "low_veto_geometry_p1_m": 2.03,
            "low_veto_floor_inlier_ratio": 0.80,
            "low_veto_gap_m": 0.26,
        },
        "geometry_input_variants": ["raw_depth", "loso_global_scale_a_times_raw_depth"],
        "threshold_rule": "gt_distance < 2.0; pred_distance < 2.0",
        "boundary_subset": "1.5 <= gt_distance <= 2.5",
        "depth_scale": args.depth_scale,
        "max_timestamp_delta_sec": args.max_timestamp_delta,
        "alignment": {
            "frames": len(alignments), "matched": len(matched),
            "unmatched": len(alignments) - len(matched),
        },
        "folds": [folds[sequence] for sequence in SEQUENCES],
        "aggregate_out_of_fold": variants,
        "aggregate_geometry_diagnostics": {
            "raw_geometry": _geometry_summary(predictions, "raw_geometry"),
            "scaled_geometry": _geometry_summary(predictions, "scaled_geometry"),
        },
        "raw_vs_scaled_geometry_feature_diagnostic": feature_diagnostic,
    }
    report_path = output_dir / "da2_geometry_scale_report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"report={report_path}")
    print(f"predictions={prediction_path}")
    print(f"alignment={alignment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
