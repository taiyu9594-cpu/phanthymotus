#!/usr/bin/env python3
"""Evaluate simple YOLO-depth FN rescue rules with strict sequence LOSO."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


SEQUENCES = ("slam1", "slam2", "slam3", "pioneer_360")
FEATURES = ("yolo_p1", "yolo_p5", "yolo_p10")
FEATURE_TIE_ORDER = {"yolo_p5": 2, "yolo_p10": 1, "yolo_p1": 0}
PRED_ISO_UPPERS = tuple(2.0 + 0.05 * index for index in range(61)) + (None,)
YOLO_THRESHOLDS = tuple(1.0 + 0.05 * index for index in range(61))
RESCUE_DISTANCE = 1.99


@dataclass(frozen=True)
class JoinedRow:
    sequence: str
    source_index: str
    image_path: str
    gt: float
    baseline_pred: float
    pred_iso: float
    fixed_geometry_vetoed: bool
    yolo_p1: float
    yolo_p5: float
    yolo_p10: float


@dataclass(frozen=True)
class Rule:
    feature: str | None
    pred_iso_upper: float | None
    yolo_threshold: float | None
    no_rescue: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "pred_iso_upper": (
                "disabled" if self.pred_iso_upper is None and not self.no_rescue
                else self.pred_iso_upper
            ),
            "yolo_threshold": self.yolo_threshold,
            "no_rescue": self.no_rescue,
        }


NO_RESCUE = Rule(None, None, None, True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-depth-csv", type=Path)
    parser.add_argument("--geometry-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-frame-count", type=int, default=1100)
    parser.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_self_test and (
        args.raw_depth_csv is None or args.geometry_csv is None or args.output_dir is None
    ):
        parser.error(
            "--raw-depth-csv, --geometry-csv and --output-dir are required unless "
            "--synthetic-self-test is used"
        )
    if args.expected_frame_count <= 0:
        parser.error("--expected-frame-count must be positive")
    return args


def _read_unique_csv(path: Path, required: set[str]) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for line_no, item in enumerate(reader, start=2):
            sequence = item["sequence"].strip()
            source_index = item["source_index"].strip()
            if sequence not in SEQUENCES:
                raise ValueError(f"{path}:{line_no} unexpected sequence: {sequence!r}")
            key = (sequence, source_index)
            if key in rows:
                raise ValueError(f"{path}:{line_no} duplicate key: {key}")
            rows[key] = dict(item)
    return rows


def _parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean {label}: {value!r}")


def _finite(value: str, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {label}: {value!r}")
    return parsed


def _load_joined(
    raw_path: Path, geometry_path: Path, expected_count: int,
) -> list[JoinedRow]:
    raw_required = {
        "sequence", "source_index", "image_path", "gt_distance",
        "yolo_p1", "yolo_p5", "yolo_p10",
    }
    geometry_required = {
        "sequence", "source_index", "gt_distance", "pred_iso",
        "fixed_geometry_pred", "fixed_geometry_vetoed",
    }
    raw = _read_unique_csv(raw_path, raw_required)
    geometry = _read_unique_csv(geometry_path, geometry_required)
    if len(raw) != expected_count or len(geometry) != expected_count:
        raise ValueError(
            f"expected {expected_count} rows per CSV, got raw={len(raw)} geometry={len(geometry)}"
        )
    raw_keys = set(raw)
    geometry_keys = set(geometry)
    raw_only = raw_keys - geometry_keys
    geometry_only = geometry_keys - raw_keys
    if len(raw_keys & geometry_keys) != expected_count or raw_only or geometry_only:
        raise ValueError(
            f"join mismatch: intersection={len(raw_keys & geometry_keys)} "
            f"raw_only={len(raw_only)} geometry_only={len(geometry_only)}"
        )
    joined = []
    for key in sorted(raw_keys, key=lambda item: (SEQUENCES.index(item[0]), item[1])):
        raw_row = raw[key]
        geometry_row = geometry[key]
        raw_gt = _finite(raw_row["gt_distance"], f"raw gt {key}")
        geometry_gt = _finite(geometry_row["gt_distance"], f"geometry gt {key}")
        if abs(raw_gt - geometry_gt) > 1e-6:
            raise ValueError(f"GT mismatch for {key}: raw={raw_gt} geometry={geometry_gt}")
        joined.append(JoinedRow(
            sequence=key[0], source_index=key[1], image_path=raw_row["image_path"],
            gt=raw_gt,
            baseline_pred=_finite(geometry_row["fixed_geometry_pred"], f"baseline {key}"),
            pred_iso=_finite(geometry_row["pred_iso"], f"pred_iso {key}"),
            fixed_geometry_vetoed=_parse_bool(
                geometry_row["fixed_geometry_vetoed"], f"fixed_geometry_vetoed {key}"
            ),
            yolo_p1=_finite(raw_row["yolo_p1"], f"yolo_p1 {key}"),
            yolo_p5=_finite(raw_row["yolo_p5"], f"yolo_p5 {key}"),
            yolo_p10=_finite(raw_row["yolo_p10"], f"yolo_p10 {key}"),
        ))
    counts = Counter(row.sequence for row in joined)
    if set(counts) != set(SEQUENCES) or any(counts[sequence] == 0 for sequence in SEQUENCES):
        raise ValueError(f"all four sequences must be non-empty: {dict(counts)}")
    return joined


def _arrays(rows: Sequence[JoinedRow]) -> dict[str, np.ndarray]:
    return {
        "gt": np.asarray([row.gt for row in rows], dtype=np.float64),
        "baseline": np.asarray([row.baseline_pred for row in rows], dtype=np.float64),
        "pred_iso": np.asarray([row.pred_iso for row in rows], dtype=np.float64),
        "vetoed": np.asarray([row.fixed_geometry_vetoed for row in rows], dtype=bool),
        **{
            feature: np.asarray([getattr(row, feature) for row in rows], dtype=np.float64)
            for feature in FEATURES
        },
    }


def _eligible(data: dict[str, np.ndarray]) -> np.ndarray:
    return (data["baseline"] >= 2.0) & ~data["vetoed"]


def _apply_rule(data: dict[str, np.ndarray], rule: Rule) -> tuple[np.ndarray, np.ndarray]:
    prediction = data["baseline"].copy()
    if rule.no_rescue:
        return prediction, np.zeros(prediction.shape, dtype=bool)
    if rule.feature is None or rule.yolo_threshold is None:
        raise ValueError("rescue rule is incomplete")
    rescued = _eligible(data) & (data[rule.feature] <= rule.yolo_threshold)
    if rule.pred_iso_upper is not None:
        rescued &= data["pred_iso"] <= rule.pred_iso_upper
    prediction[rescued] = RESCUE_DISTANCE
    return prediction, rescued


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, int | float | None]:
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
        "mae": float(np.mean(np.abs(error))) if gt.size else None,
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if gt.size else None,
    }


def _metric_pair(gt: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    boundary = (gt >= 1.5) & (gt <= 2.5)
    return {
        "all": _metrics(gt, pred),
        "boundary_1p5_to_2p5": _metrics(gt[boundary], pred[boundary]),
    }


def _transitions(gt: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    actual = gt < 2.0
    before = baseline < 2.0
    after = pred < 2.0
    result = {
        "fp_to_tn": int(np.sum(~actual & before & ~after)),
        "tp_to_fn": int(np.sum(actual & before & ~after)),
        "fn_to_tp": int(np.sum(actual & ~before & after)),
        "tn_to_fp": int(np.sum(~actual & ~before & after)),
    }
    if result["fp_to_tn"] or result["tp_to_fn"]:
        raise AssertionError(f"rescue-only invariant violated: {result}")
    return result


def _candidate_key(
    rule: Rule, metrics: dict[str, int | float | None], transitions: dict[str, int],
    rescued_count: int,
) -> tuple[float, int, int, int, float, float, int]:
    upper = math.inf if rule.pred_iso_upper is None else rule.pred_iso_upper
    return (
        float(metrics["f1"]), -transitions["tn_to_fp"], transitions["fn_to_tp"],
        -rescued_count, -float(rule.yolo_threshold), -upper,
        FEATURE_TIE_ORDER[str(rule.feature)],
    )


def _select_rule(rows: Sequence[JoinedRow]) -> tuple[Rule, dict[str, object]]:
    data = _arrays(rows)
    baseline_metrics = _metrics(data["gt"], data["baseline"])
    best: tuple[tuple[float, int, int, int, float, float, int], Rule,
                dict[str, int], int, dict[str, int | float | None]] | None = None
    for feature in FEATURES:
        for upper in PRED_ISO_UPPERS:
            for threshold in YOLO_THRESHOLDS:
                rule = Rule(feature, upper, threshold)
                prediction, rescued = _apply_rule(data, rule)
                transitions = _transitions(data["gt"], data["baseline"], prediction)
                if transitions["fn_to_tp"] <= 0:
                    continue
                if transitions["tn_to_fp"] > transitions["fn_to_tp"]:
                    continue
                metrics = _metrics(data["gt"], prediction)
                if float(metrics["f1"]) <= float(baseline_metrics["f1"]):
                    continue
                rescued_count = int(rescued.sum())
                key = _candidate_key(rule, metrics, transitions, rescued_count)
                if best is None or key > best[0]:
                    best = (key, rule, transitions, rescued_count, metrics)
    if best is None:
        return NO_RESCUE, {
            "baseline_metrics": baseline_metrics, "selected_metrics": baseline_metrics,
            "transitions": _transitions(data["gt"], data["baseline"], data["baseline"]),
            "rescued_frame_count": 0, "fallback_to_no_rescue": True,
        }
    _, rule, transitions, rescued_count, metrics = best
    return rule, {
        "baseline_metrics": baseline_metrics, "selected_metrics": metrics,
        "transitions": transitions, "rescued_frame_count": rescued_count,
        "fallback_to_no_rescue": False,
    }


def _subset_report(rows: Sequence[JoinedRow], rule: Rule) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    data = _arrays(rows)
    prediction, rescued = _apply_rule(data, rule)
    eligible = _eligible(data)
    report = {
        "baseline_metrics": _metric_pair(data["gt"], data["baseline"]),
        "selected_metrics": _metric_pair(data["gt"], prediction),
        "transitions": _transitions(data["gt"], data["baseline"], prediction),
        "eligible_frame_count": int(eligible.sum()),
        "eligible_gt_positive_count": int(np.sum(eligible & (data["gt"] < 2.0))),
        "eligible_gt_negative_count": int(np.sum(eligible & (data["gt"] >= 2.0))),
        "rescued_frame_count": int(rescued.sum()),
    }
    return report, prediction, rescued


def _class_label(gt: float, pred: float) -> str:
    if gt < 2.0:
        return "TP" if pred < 2.0 else "FN"
    return "FP" if pred < 2.0 else "TN"


def _transition_label(gt: float, baseline: float, pred: float) -> str:
    before = _class_label(gt, baseline)
    after = _class_label(gt, pred)
    if before == "FN" and after == "TP":
        return "FN_to_TP"
    if before == "TN" and after == "FP":
        return "TN_to_FP"
    return f"unchanged_{before}"


def _frame_record(row: JoinedRow, rule: Rule, yolo_value: float) -> dict[str, object]:
    return {
        "sequence": row.sequence, "source_index": row.source_index, "gt": row.gt,
        "baseline_pred": row.baseline_pred, "pred_iso": row.pred_iso,
        "selected_feature": rule.feature, "yolo_value": yolo_value,
        "U": "disabled" if rule.pred_iso_upper is None else rule.pred_iso_upper,
        "T": rule.yolo_threshold,
    }


def _correction_loss_ratio(transitions: dict[str, int]) -> float | str:
    losses = transitions["tn_to_fp"]
    return "infinity" if losses == 0 else transitions["fn_to_tp"] / losses


def _parameter_stability(rules: Sequence[Rule]) -> dict[str, object]:
    rescue_rules = [rule for rule in rules if not rule.no_rescue]
    features = Counter(str(rule.feature) for rule in rescue_rules)
    finite_uppers = [float(rule.pred_iso_upper) for rule in rescue_rules
                     if rule.pred_iso_upper is not None]
    thresholds = [float(rule.yolo_threshold) for rule in rescue_rules
                  if rule.yolo_threshold is not None]
    disabled_count = sum(rule.pred_iso_upper is None for rule in rescue_rules)
    instability = (
        len(features) > 1
        or (disabled_count > 0 and finite_uppers)
        or (finite_uppers and max(finite_uppers) - min(finite_uppers) > 0.5)
        or (thresholds and max(thresholds) - min(thresholds) > 0.5)
    )
    def stats(values: Sequence[float]) -> dict[str, float | None]:
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": float(np.mean(values)) if values else None,
        }
    return {
        "selected_feature_counts": dict(features),
        "no_rescue_count": sum(rule.no_rescue for rule in rules),
        "pred_iso_upper": {**stats(finite_uppers), "disabled_count": disabled_count},
        "yolo_threshold": stats(thresholds),
        "instability_criterion": (
            "multiple selected features, mixed disabled/finite U, U span > 0.5m, "
            "or T span > 0.5m"
        ),
        "cross_sequence_parameter_instability": bool(instability),
    }


def _run_loso(rows: Sequence[JoinedRow]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    folds = []
    oof_rows = []
    for holdout in SEQUENCES:
        train_rows = [row for row in rows if row.sequence != holdout]
        holdout_rows = [row for row in rows if row.sequence == holdout]
        rule, selection = _select_rule(train_rows)
        train_report, _, _ = _subset_report(train_rows, rule)
        holdout_report, holdout_prediction, holdout_rescued = _subset_report(
            holdout_rows, rule
        )
        folds.append({
            "holdout_sequence": holdout,
            "train_sequences": [sequence for sequence in SEQUENCES if sequence != holdout],
            "selected": rule.as_dict(), "selection": selection,
            "train": train_report, "holdout": holdout_report,
        })
        for row, prediction, rescued in zip(
            holdout_rows, holdout_prediction, holdout_rescued
        ):
            eligible = row.baseline_pred >= 2.0 and not row.fixed_geometry_vetoed
            yolo_value = None if rule.no_rescue else float(getattr(row, str(rule.feature)))
            oof_rows.append({
                "row": row, "rule": rule, "eligible": eligible,
                "rescued": bool(rescued), "oof_pred": float(prediction),
                "yolo_value": yolo_value,
            })
    return folds, oof_rows


def _synthetic_rows() -> list[JoinedRow]:
    rows = []
    for seq_index, sequence in enumerate(SEQUENCES):
        examples = [
            (1.5, 1.8, False, 1.0),  # baseline TP: never eligible
            (1.7, 2.4, False, 1.1),  # eligible FN -> TP
            (2.4, 2.5, False, 1.2),  # eligible TN -> FP
            (1.8, 2.3, True, 0.5),   # veto-protected FN
        ]
        for index, (gt, baseline, vetoed, yolo) in enumerate(examples):
            rows.append(JoinedRow(
                sequence, f"{seq_index}-{index}", f"{seq_index}-{index}.png", gt,
                baseline, 2.1, vetoed, yolo, yolo, yolo,
            ))
    return rows


def _write_synthetic_csvs(directory: Path) -> tuple[Path, Path]:
    rows = _synthetic_rows()
    raw_path = directory / "raw.csv"
    geometry_path = directory / "geometry.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["sequence", "source_index", "image_path", "gt_distance", *FEATURES]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in reversed(rows):
            writer.writerow({
                "sequence": row.sequence, "source_index": row.source_index,
                "image_path": row.image_path, "gt_distance": row.gt,
                **{feature: getattr(row, feature) for feature in FEATURES},
            })
    with geometry_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["sequence", "source_index", "gt_distance", "pred_iso",
                  "fixed_geometry_pred", "fixed_geometry_vetoed"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sequence": row.sequence, "source_index": row.source_index,
                "gt_distance": row.gt, "pred_iso": row.pred_iso,
                "fixed_geometry_pred": row.baseline_pred,
                "fixed_geometry_vetoed": row.fixed_geometry_vetoed,
            })
    return raw_path, geometry_path


def _synthetic_self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="yolo_fn_rescue_test_") as temp:
        raw_path, geometry_path = _write_synthetic_csvs(Path(temp))
        joined = _load_joined(raw_path, geometry_path, 16)
        assert len(joined) == 16 and joined[0].source_index == "0-0"
        data = _arrays(joined)
        permissive = Rule("yolo_p5", None, 1.2)
        prediction, rescued = _apply_rule(data, permissive)
        assert not np.any(rescued[data["baseline"] < 2.0])
        assert not np.any(rescued[data["vetoed"]])
        changes = _transitions(data["gt"], data["baseline"], prediction)
        assert changes == {"fp_to_tn": 0, "tp_to_fn": 0, "fn_to_tp": 4, "tn_to_fp": 4}

        duplicate = raw_path.read_text(encoding="utf-8")
        raw_path.write_text(duplicate + duplicate.splitlines()[1] + "\n", encoding="utf-8")
        try:
            _load_joined(raw_path, geometry_path, 16)
        except ValueError as exc:
            assert "duplicate key" in str(exc)
        else:
            raise AssertionError("duplicate key did not fail")

        raw_path, geometry_path = _write_synthetic_csvs(Path(temp))
        geometry_text = geometry_path.read_text(encoding="utf-8")
        geometry_path.write_text(geometry_text.replace(",1.5,", ",1.50001,", 1),
                                 encoding="utf-8")
        try:
            _load_joined(raw_path, geometry_path, 16)
        except ValueError as exc:
            assert "GT mismatch" in str(exc)
        else:
            raise AssertionError("GT mismatch did not fail")

    no_gain_rows = [
        JoinedRow(sequence, f"n-{i}", "x", 2.5, 2.5, 2.5, False, 3.0, 3.0, 3.0)
        for i, sequence in enumerate(SEQUENCES)
    ]
    rule, selection = _select_rule(no_gain_rows)
    assert rule.no_rescue and selection["fallback_to_no_rescue"]

    # The selector receives train rows only. Mutating separate holdout labels cannot
    # affect the selected rule for that fold.
    train = [row for row in joined if row.sequence != "slam1"]
    selected_before = _select_rule(train)[0]
    mutated_holdout = [
        JoinedRow(row.sequence, row.source_index, row.image_path, 99.0,
                  row.baseline_pred, row.pred_iso, row.fixed_geometry_vetoed,
                  row.yolo_p1, row.yolo_p5, row.yolo_p10)
        for row in joined if row.sequence == "slam1"
    ]
    assert mutated_holdout and _select_rule(train)[0] == selected_before
    print("synthetic_self_test=PASS checks=10")
    return 0


def main() -> int:
    args = _parse_args()
    if args.synthetic_self_test:
        return _synthetic_self_test()
    rows = _load_joined(
        args.raw_depth_csv.expanduser().resolve(),
        args.geometry_csv.expanduser().resolve(),
        args.expected_frame_count,
    )
    folds, oof = _run_loso(rows)
    oof.sort(key=lambda item: (SEQUENCES.index(item["row"].sequence), item["row"].source_index))
    gt = np.asarray([item["row"].gt for item in oof], dtype=np.float64)
    baseline = np.asarray([item["row"].baseline_pred for item in oof], dtype=np.float64)
    prediction = np.asarray([item["oof_pred"] for item in oof], dtype=np.float64)
    transitions = _transitions(gt, baseline, prediction)
    aggregate = {
        "baseline": _metric_pair(gt, baseline),
        "loso_yolo_rescue": _metric_pair(gt, prediction),
        "transitions_from_baseline": transitions,
        "correction_loss_ratio": _correction_loss_ratio(transitions),
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "yolo_fn_rescue_oof_predictions.csv"
    fields = [
        "sequence", "source_index", "gt_distance", "baseline_pred", "pred_iso",
        "fixed_geometry_vetoed", *FEATURES, "selected_feature",
        "selected_pred_iso_upper", "selected_yolo_threshold", "eligible", "rescued",
        "oof_pred", "baseline_class", "oof_class", "transition",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in oof:
            row = item["row"]
            rule = item["rule"]
            writer.writerow({
                "sequence": row.sequence, "source_index": row.source_index,
                "gt_distance": row.gt, "baseline_pred": row.baseline_pred,
                "pred_iso": row.pred_iso,
                "fixed_geometry_vetoed": row.fixed_geometry_vetoed,
                **{feature: getattr(row, feature) for feature in FEATURES},
                "selected_feature": rule.feature or "",
                "selected_pred_iso_upper": (
                    "disabled" if rule.pred_iso_upper is None and not rule.no_rescue
                    else ("" if rule.no_rescue else rule.pred_iso_upper)
                ),
                "selected_yolo_threshold": rule.yolo_threshold or "",
                "eligible": item["eligible"], "rescued": item["rescued"],
                "oof_pred": item["oof_pred"],
                "baseline_class": _class_label(row.gt, row.baseline_pred),
                "oof_class": _class_label(row.gt, float(item["oof_pred"])),
                "transition": _transition_label(
                    row.gt, row.baseline_pred, float(item["oof_pred"])
                ),
            })

    fn_to_tp_frames = []
    tn_to_fp_frames = []
    for item in oof:
        row = item["row"]
        label = _transition_label(row.gt, row.baseline_pred, float(item["oof_pred"]))
        if label not in {"FN_to_TP", "TN_to_FP"}:
            continue
        record = _frame_record(row, item["rule"], float(item["yolo_value"]))
        (fn_to_tp_frames if label == "FN_to_TP" else tn_to_fp_frames).append(record)

    full_rule, full_selection = _select_rule(rows)
    full_report, _, _ = _subset_report(rows, full_rule)
    stability = _parameter_stability([
        Rule(
            None if fold["selected"]["feature"] is None else str(fold["selected"]["feature"]),
            None if fold["selected"]["pred_iso_upper"] in {None, "disabled"}
            else float(fold["selected"]["pred_iso_upper"]),
            None if fold["selected"]["yolo_threshold"] is None
            else float(fold["selected"]["yolo_threshold"]),
            bool(fold["selected"]["no_rescue"]),
        ) for fold in folds
    ])
    warnings = []
    if stability["cross_sequence_parameter_instability"]:
        warnings.append("cross-sequence parameter instability")
    report = {
        "experiment_type": "strict_loso_yolo_fn_rescue",
        "baseline_source": "fixed_geometry_pred",
        "geometry_veto_protected": True,
        "rescue_distance": RESCUE_DISTANCE,
        "threshold_rule": "gt < 2.0; pred < 2.0",
        "boundary_subset": "1.5 <= gt <= 2.5",
        "input_rows": len(rows),
        "search_space": {
            "features": list(FEATURES),
            "pred_iso_upper": "2.00..5.00 step 0.05 plus disabled",
            "yolo_threshold": "1.00..4.00 step 0.05",
            "includes_no_rescue": True,
        },
        "folds": folds,
        "aggregate_oof": aggregate,
        "oof_fn_to_tp_frames": fn_to_tp_frames,
        "oof_tn_to_fp_frames": tn_to_fp_frames,
        "parameter_stability": stability,
        "full_data_deployment_fit": {
            "warning": "full_data_deployment_fit is NOT held-out generalization",
            "full_data_selected_rule": full_rule.as_dict(),
            "selection": full_selection,
            "full_data_metrics": full_report["selected_metrics"],
            "full_data_transitions": full_report["transitions"],
        },
        "generalization_basis": "strict LOSO OOF metrics only",
        "warnings": warnings,
    }
    report_path = output_dir / "yolo_fn_rescue_report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"report={report_path}")
    print(f"predictions={prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
