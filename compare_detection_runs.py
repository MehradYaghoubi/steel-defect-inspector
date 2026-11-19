"""Estimate paired image bootstrap intervals for the difference in COCO AP."""

import argparse
import contextlib
import io
import hashlib
import json
from pathlib import Path

import numpy as np

from steel_inspector.data import CLASSES, write_json
from steel_inspector.evaluation import coco_match, coco_report


METRICS = ("coco_ap_50_95", "coco_ap_50")
MATCHED_FIELDS = ("split", "split_sha256", "manifest_sha256", "ap_score_floor", "nms_iou", "max_detections")


def cache_coco_matches(images, annotations, predictions):
    if not predictions:
        return None
    # The requested AP metrics use the all-area range and 100-box cap.
    evaluator = coco_match(images, annotations, predictions, all_area_only=True)
    positions = {key: index for index, key in enumerate(evaluator.params.imgIds)}
    matches = np.array(evaluator.evalImgs, dtype=object).reshape(len(CLASSES), len(images))
    return evaluator, matches[:, [positions[image["id"]] for image in images]]


def resampled_ap(cache, indices):
    if cache is None:
        return np.zeros(len(METRICS))
    evaluator, matches = cache
    # Each draw keeps one evaluation slot per image. Repeating a slot copies
    # both the matched detections and the ground truth contribution.
    evaluator.evalImgs = matches[:, indices].ravel().tolist()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()
    all_iou = evaluator.eval["precision"][:, :, :, 0, 0]
    return np.asarray([subset[subset >= 0].mean() if np.any(subset >= 0) else 0.0
                       for subset in (all_iou, all_iou[0])])


def paired_ap_difference(baseline, candidate, *, repetitions, seed):
    if repetitions < 2:
        raise ValueError("At least two repetitions are required")
    images, annotations = baseline["images"], baseline["annotations"]
    for field in MATCHED_FIELDS:
        if baseline["report"][field] != candidate["report"][field]:
            raise ValueError(f"Prediction bundles differ in {field}")
    if not images or images != candidate["images"] or annotations != candidate["annotations"]:
        raise ValueError("Prediction bundles must contain the same ordered images and annotations")

    runs = baseline, candidate
    image_ids = {image["id"] for image in images}
    referenced = {row["image_id"] for rows in (annotations, *(run["predictions"] for run in runs)) for row in rows}
    if len(image_ids) != len(images) or not referenced <= image_ids:
        raise ValueError("Duplicate or unknown image IDs in prediction bundles")

    observed = [coco_report(images, annotations, run["predictions"]) for run in runs]
    caches = [cache_coco_matches(images, annotations, run["predictions"]) for run in runs]
    draws = np.empty((repetitions, len(METRICS)))
    rng = np.random.default_rng(seed)
    for repetition in range(repetitions):
        indices = rng.integers(0, len(images), size=len(images))
        draws[repetition] = resampled_ap(caches[1], indices) - resampled_ap(caches[0], indices)

    return {
        "split": baseline["report"]["split"], "split_sha256": baseline["report"]["split_sha256"],
        "image_count": len(images), "repetitions": repetitions, "seed": seed,
        "baseline": baseline["report"], "candidate": candidate["report"],
        "metric_differences": {
            metric: {
                "candidate_minus_baseline": observed[1][metric] - observed[0][metric],
                "paired_bootstrap_95_interval": np.quantile(draws[:, index], [0.025, 0.975]).tolist(),
            }
            for index, metric in enumerate(METRICS)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_paths = (args.baseline, args.candidate)
    if any(path.resolve() == args.output.resolve() for path in input_paths):
        parser.error("The output must be different from both prediction bundles")
    input_bytes = [path.read_bytes() for path in input_paths]
    baseline, candidate = map(json.loads, input_bytes)
    result = paired_ap_difference(baseline, candidate, repetitions=args.repetitions, seed=args.seed)
    for name, payload in zip(("baseline", "candidate"), input_bytes, strict=True):
        result[name + "_bundle_sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(args.output, result)
