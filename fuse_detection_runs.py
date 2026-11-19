"""Fuse scored detection bundles with confidence-weighted box coordinates."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from compare_detection_runs import MATCHED_FIELDS
from steel_inspector.data import write_json
from steel_inspector.evaluation import coco_report, xywh
from steel_inspector.inference import iou


def fuse(bundles, overlap=0.55):
    if len(bundles) < 2 or not 0 < overlap <= 1:
        raise ValueError("Fusion requires at least two bundles and IoU in (0, 1]")
    lead = bundles[0]
    report = {field: lead["report"][field] for field in MATCHED_FIELDS}
    for bundle in bundles[1:]:
        if bundle["images"] != lead["images"] or bundle["annotations"] != lead["annotations"]:
            raise ValueError("Fusion requires identical images and annotations")
        if any(bundle["report"][field] != report[field] for field in MATCHED_FIELDS):
            raise ValueError("Fusion requires equivalent detector evaluation settings")

    detections = defaultdict(list)
    for member, bundle in enumerate(bundles):
        for row in bundle["predictions"]:
            x, y, width, height = row["bbox"]
            detections[row["image_id"], row["category_id"]].append(
                (row["score"], member, np.array([x, y, x + width, y + height]))
            )
    outputs = defaultdict(list)
    for (image_id, category_id), rows in detections.items():
        clusters = []
        for score, member, box in sorted(rows, key=lambda row: row[0], reverse=True):
            available = [group for group in clusters if member not in group["members"]]
            if available:
                overlaps = iou(box, np.array([group["box"] for group in available]))
                best = int(overlaps.argmax())
                if overlaps[best] >= overlap:
                    group = available[best]
                    group["weighted_box"] += box * score
                    group["score"] += score
                    group["box"] = group["weighted_box"] / group["score"]
                    group["members"].add(member)
                    continue
            clusters.append({"weighted_box": box * score, "score": score, "box": box, "members": {member}})
        for group in clusters:
            score = group["score"] / len(bundles)
            if score >= report["ap_score_floor"]:
                outputs[image_id].append({
                    "image_id": image_id, "category_id": category_id,
                    "bbox": xywh(group["box"]), "score": score,
                })

    predictions = [
        row for per_image in outputs.values()
        for row in sorted(per_image, key=lambda row: row["score"], reverse=True)[:report["max_detections"]]
    ]
    report.update(
        image_count=len(lead["images"]), annotation_count=len(lead["annotations"]),
        fusion_iou=overlap, members=[bundle["report"] for bundle in bundles],
        **coco_report(lead["images"], lead["annotations"], predictions),
    )
    return dict(images=lead["images"], annotations=lead["annotations"], predictions=predictions, report=report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, nargs="+", required=True)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.members]
    if len(set(paths)) != len(paths) or args.output.resolve() in paths:
        parser.error("Use distinct member paths and a separate output path")
    payloads = [path.read_bytes() for path in paths]
    result = fuse([json.loads(payload) for payload in payloads], args.iou)
    result["report"]["member_bundle_sha256"] = [hashlib.sha256(payload).hexdigest() for payload in payloads]
    write_json(args.output, result)
