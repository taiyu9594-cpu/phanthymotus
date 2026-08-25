#!/usr/bin/env python3
"""Fit and compare Indoor raw-depth calibrations with sequence holdouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


DEFAULT_SEQUENCES = ("slam1", "slam2", "slam3", "pioneer_360")
MODEL_FEATURES = {
    "da2": ("da2_min", "da2_p1", "da2_p5", "da2_p10"),
    "yolo": ("yolo_min", "yolo_p1", "yolo_p5", "yolo_p10"),
}
CALIBRATIONS = ("raw", "scale_only", "affine", "isotonic")


@dataclass(frozen=True)
class Row:
    sequence: str
    source_index: str
    image_path: str
    gt: float
    values: dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DA2 and YOLO raw-depth calibration using leave-one-sequence-out "
            "validation. No production calibration is read or modified."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=list(DEFAULT_SEQUENCES),
        help="Sequences participating in leave-one-sequence-out validation.",
    )
    return parser.parse_args()


def _read_rows(path: Path, sequences: Sequence[str]) -> list[Row]:
    feature_names = tuple(name for names in MODEL_FEATURES.values() for name in names)
    required = {"sequence", "source_index", "image_path", "gt_distance", *feature_names}
    allowed = set(sequences)
    rows: list[Row] = []
    seen: set[tuple[str, str]] = set()

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"input CSV is missing columns: {sorted(missing)}")
        for line_no, item in enumerate(reader, start=2):
            sequence = item["sequence"].strip()
            if sequence not in allowed:
                continue
            source_index = item["source_index"].strip()
            key = (sequence, source_index)
            if key in seen:
                raise ValueError(f"duplicate sequence/source_index at line {line_no}: {key}")
            seen.add(key)
            try:
                gt = float(item["gt_distance"])
                values = {name: float(item[name]) for name in feature_names}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric value at line {line_no}") from exc
            if not math.isfinite(gt) or any(not math.isfinite(v) for v in values.values()):
                raise ValueError(f"non-finite value at line {line_no}")
            rows.append(
                Row(
                    sequence=sequence,
                    source_index=source_index,
                    image_path=item["image_path"],
                    gt=gt,
                    values=values,
                )
            )

    counts = {sequence: sum(row.sequence == sequence for row in rows) for sequence in sequences}
    empty = [sequence for sequence, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"no rows found for sequences: {empty}; counts={counts}")
    print(f"loaded_rows={len(rows)} sequence_counts={json.dumps(counts, sort_keys=True)}")
    return rows


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    if gt.size == 0:
        return {
            "count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mae": None,
            "rmse": None,
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
        "count": int(gt.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _evaluate(
    gt: np.ndarray, pred: np.ndarray
) -> dict[str, dict[str, float | int | None]]:
    boundary = (gt >= 1.5) & (gt <= 2.5)
    return {
        "all": _metrics(gt, pred),
        "boundary_1p5_to_2p5": _metrics(gt[boundary], pred[boundary]),
    }


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    sorted_y = y[order]
    unique_x, starts, counts = np.unique(sorted_x, return_index=True, return_counts=True)
    means = np.add.reduceat(sorted_y, starts) / counts

    block_start: list[int] = []
    block_end: list[int] = []
    block_weight: list[float] = []
    block_value: list[float] = []
    for index, (value, weight) in enumerate(zip(means, counts)):
        block_start.append(index)
        block_end.append(index)
        block_weight.append(float(weight))
        block_value.append(float(value))
        while len(block_value) >= 2 and block_value[-2] > block_value[-1]:
            total_weight = block_weight[-2] + block_weight[-1]
            pooled = (
                block_value[-2] * block_weight[-2]
                + block_value[-1] * block_weight[-1]
            ) / total_weight
            block_end[-2] = block_end[-1]
            block_weight[-2] = total_weight
            block_value[-2] = pooled
            block_start.pop()
            block_end.pop()
            block_weight.pop()
            block_value.pop()

    fitted = np.empty(unique_x.size, dtype=np.float64)
    for start, end, value in zip(block_start, block_end, block_value):
        fitted[start : end + 1] = value
    return unique_x, fitted


def _fit_calibration(
    kind: str, x: np.ndarray, y: np.ndarray
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, float | int]]:
    if kind == "raw":
        return lambda values: values.copy(), {}
    if kind == "scale_only":
        denominator = float(np.dot(x, x))
        if denominator <= 0.0:
            raise ValueError("cannot fit scale-only calibration with zero denominator")
        scale = float(np.dot(x, y) / denominator)
        return lambda values: scale * values, {"scale": scale}
    if kind == "affine":
        design = np.column_stack((x, np.ones_like(x)))
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        slope = float(slope)
        intercept = float(intercept)
        return (
            lambda values: slope * values + intercept,
            {"slope": slope, "intercept": intercept},
        )
    if kind == "isotonic":
        knots, fitted = _fit_isotonic(x, y)
        return (
            lambda values: np.interp(values, knots, fitted),
            {
                "knot_count": int(knots.size),
                "x_min": float(knots[0]),
                "x_max": float(knots[-1]),
            },
        )
    raise ValueError(f"unsupported calibration: {kind}")


def _candidate_name(model: str, feature: str, calibration: str) -> str:
    percentile = feature.removeprefix(f"{model}_")
    return f"{model}:{percentile}:{calibration}"


def _run_candidate(
    rows: Sequence[Row], sequences: Sequence[str], model: str, feature: str, calibration: str
) -> tuple[dict[str, object], np.ndarray]:
    gt_all = np.asarray([row.gt for row in rows], dtype=np.float64)
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    folds: list[dict[str, object]] = []

    for holdout in sequences:
        train_indices = [i for i, row in enumerate(rows) if row.sequence != holdout]
        test_indices = [i for i, row in enumerate(rows) if row.sequence == holdout]
        train_x = np.asarray([rows[i].values[feature] for i in train_indices], dtype=np.float64)
        train_y = gt_all[train_indices]
        test_x = np.asarray([rows[i].values[feature] for i in test_indices], dtype=np.float64)
        predictor, parameters = _fit_calibration(calibration, train_x, train_y)
        test_pred = predictor(test_x)
        predictions[test_indices] = test_pred
        folds.append(
            {
                "holdout_sequence": holdout,
                "train_count": len(train_indices),
                "test_count": len(test_indices),
                "fit_parameters": parameters,
                "metrics": _evaluate(gt_all[test_indices], test_pred),
            }
        )

    if np.any(~np.isfinite(predictions)):
        raise RuntimeError("holdout prediction coverage is incomplete or non-finite")
    report = {
        "candidate": _candidate_name(model, feature, calibration),
        "model": model,
        "feature": feature,
        "calibration": calibration,
        "folds": folds,
        "aggregate_holdout": _evaluate(gt_all, predictions),
    }
    return report, predictions


def _selection_key(report: dict[str, object]) -> tuple[float, float, float]:
    metrics = report["aggregate_holdout"]["all"]  # type: ignore[index]
    return (float(metrics["f1"]), -float(metrics["mae"]), -float(metrics["rmse"]))


def _classification(gt: float, pred: float) -> str:
    actual = gt < 2.0
    predicted = pred < 2.0
    if actual:
        return "TP" if predicted else "FN"
    return "FP" if predicted else "TN"


def _write_best_predictions(
    path: Path,
    rows: Sequence[Row],
    best_reports: dict[str, dict[str, object]],
    best_predictions: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="") as stream:
        fields = [
            "sequence",
            "source_index",
            "image_path",
            "gt_distance",
            "da2_candidate",
            "da2_holdout_pred",
            "da2_classification",
            "yolo_candidate",
            "yolo_holdout_pred",
            "yolo_classification",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            da2_pred = float(best_predictions["da2"][index])
            yolo_pred = float(best_predictions["yolo"][index])
            writer.writerow(
                {
                    "sequence": row.sequence,
                    "source_index": row.source_index,
                    "image_path": row.image_path,
                    "gt_distance": row.gt,
                    "da2_candidate": best_reports["da2"]["candidate"],
                    "da2_holdout_pred": da2_pred,
                    "da2_classification": _classification(row.gt, da2_pred),
                    "yolo_candidate": best_reports["yolo"]["candidate"],
                    "yolo_holdout_pred": yolo_pred,
                    "yolo_classification": _classification(row.gt, yolo_pred),
                }
            )


def _write_error_overlap(
    path: Path, rows: Sequence[Row], best_predictions: dict[str, np.ndarray]
) -> dict[str, int]:
    categories = {
        ("FN", "TP"): "DA2_FN_to_YOLO_TP",
        ("FP", "TN"): "DA2_FP_to_YOLO_TN",
        ("TP", "FN"): "YOLO_FN_to_DA2_TP",
        ("TN", "FP"): "YOLO_FP_to_DA2_TN",
    }
    counts = {label: 0 for label in categories.values()}
    with path.open("w", newline="") as stream:
        fields = [
            "category",
            "sequence",
            "source_index",
            "image_path",
            "gt_distance",
            "da2_holdout_pred",
            "da2_classification",
            "yolo_holdout_pred",
            "yolo_classification",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            da2_pred = float(best_predictions["da2"][index])
            yolo_pred = float(best_predictions["yolo"][index])
            da2_class = _classification(row.gt, da2_pred)
            yolo_class = _classification(row.gt, yolo_pred)
            category = categories.get((da2_class, yolo_class))
            if category is None:
                continue
            counts[category] += 1
            writer.writerow(
                {
                    "category": category,
                    "sequence": row.sequence,
                    "source_index": row.source_index,
                    "image_path": row.image_path,
                    "gt_distance": row.gt,
                    "da2_holdout_pred": da2_pred,
                    "da2_classification": da2_class,
                    "yolo_holdout_pred": yolo_pred,
                    "yolo_classification": yolo_class,
                }
            )
    return counts


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _print_report(report: dict[str, object]) -> None:
    print(f"\ncandidate={report['candidate']}")
    for fold in report["folds"]:  # type: ignore[union-attr]
        metrics = fold["metrics"]["all"]
        boundary = fold["metrics"]["boundary_1p5_to_2p5"]
        print(
            f"  holdout={fold['holdout_sequence']} n={metrics['count']} "
            f"F1={metrics['f1']:.6f} P={metrics['precision']:.6f} "
            f"R={metrics['recall']:.6f} TP={metrics['tp']} FP={metrics['fp']} "
            f"FN={metrics['fn']} TN={metrics['tn']} MAE={metrics['mae']:.6f} "
            f"RMSE={metrics['rmse']:.6f} boundary_F1={boundary['f1']:.6f} "
            f"boundary_MAE={_format_optional(boundary['mae'])}"
        )
    metrics = report["aggregate_holdout"]["all"]  # type: ignore[index]
    boundary = report["aggregate_holdout"]["boundary_1p5_to_2p5"]  # type: ignore[index]
    print(
        f"  aggregate n={metrics['count']} F1={metrics['f1']:.6f} "
        f"P={metrics['precision']:.6f} R={metrics['recall']:.6f} "
        f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']} "
        f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} "
        f"boundary_F1={boundary['f1']:.6f} "
        f"boundary_MAE={_format_optional(boundary['mae'])}"
    )


def main() -> None:
    args = _parse_args()
    sequences = tuple(dict.fromkeys(args.sequences))
    if len(sequences) < 2:
        raise ValueError("leave-one-sequence-out requires at least two sequences")
    rows = _read_rows(args.input_csv, sequences)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    predictions_by_candidate: dict[str, np.ndarray] = {}
    for model, features in MODEL_FEATURES.items():
        for feature in features:
            for calibration in CALIBRATIONS:
                report, predictions = _run_candidate(
                    rows, sequences, model, feature, calibration
                )
                reports.append(report)
                predictions_by_candidate[str(report["candidate"])] = predictions
                _print_report(report)

    best_reports: dict[str, dict[str, object]] = {}
    best_predictions: dict[str, np.ndarray] = {}
    for model in MODEL_FEATURES:
        candidates = [report for report in reports if report["model"] == model]
        best = max(candidates, key=_selection_key)
        best_reports[model] = best
        best_predictions[model] = predictions_by_candidate[str(best["candidate"])]
        print(f"best_{model}={best['candidate']} selection=cross_sequence_holdout")

    prediction_path = args.output_dir / "best_holdout_predictions.csv"
    overlap_path = args.output_dir / "error_overlap.csv"
    _write_best_predictions(prediction_path, rows, best_reports, best_predictions)
    overlap_counts = _write_error_overlap(overlap_path, rows, best_predictions)

    report_path = args.output_dir / "calibration_report.json"
    payload = {
        "input_csv": str(args.input_csv),
        "sequences": list(sequences),
        "threshold_rule": "gt < 2.0; pred < 2.0",
        "boundary_subset": "1.5 <= gt <= 2.5",
        "selection_rule": "highest aggregate holdout F1, then lowest MAE, then lowest RMSE",
        "candidates": reports,
        "best": {model: report["candidate"] for model, report in best_reports.items()},
        "error_overlap_counts": overlap_counts,
    }
    with report_path.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print(f"error_overlap={json.dumps(overlap_counts, sort_keys=True)}")
    print(f"report={report_path}")
    print(f"best_holdout_predictions={prediction_path}")
    print(f"error_overlap_csv={overlap_path}")


if __name__ == "__main__":
    main()
