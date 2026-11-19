import contextlib
import hashlib
import io
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .data import CLASSES, preprocess, read_split, write_json
from .inference import annotate, decode, iou, load_onnx, load_torch

AP_THRESHOLD = 0.01
MATCH_IOU = 0.5


def xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def split_fingerprint(rows):
    return hashlib.sha256("".join(row["sha256"] for row in rows).encode()).hexdigest()


def coco_ap(images, annotations, predictions):
    result = coco_report(images, annotations, predictions)
    return result["coco_ap_50_95"], result["coco_ap_50"]


def coco_match(images, annotations, predictions, *, all_area_only=False):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dataset = COCO()
    dataset.dataset = {
        "info": {},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES, 1)],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        dataset.createIndex()
        evaluator = COCOeval(dataset, dataset.loadRes(predictions), "bbox")
        if all_area_only:
            evaluator.params.areaRng = [evaluator.params.areaRng[0]]
            evaluator.params.areaRngLbl = ["all"]
            evaluator.params.maxDets = [100]
        evaluator.evaluate()
    return evaluator


def coco_report(images, annotations, predictions):
    present_classes = {annotation["category_id"] for annotation in annotations if not annotation.get("iscrowd", 0)}
    per_class = {
        name: {"ap50": 0.0 if index in present_classes else None,
               "ap50_95": 0.0 if index in present_classes else None}
        for index, name in enumerate(CLASSES, 1)
    }
    if not predictions:
        return {"coco_ap_50_95": 0.0, "coco_ap_50": 0.0, "coco_per_class": per_class}
    evaluator = coco_match(images, annotations, predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()
        evaluator.summarize()
    # COCO dimensions are IoU, recall, category, area range, and detection cap.
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    for index, name in enumerate(CLASSES):
        class_precision = precision[:, :, index]
        if per_class[name]["ap50"] is None:
            continue
        at_50 = class_precision[0]
        valid_at_50 = at_50[at_50 >= 0]
        valid = class_precision[class_precision >= 0]
        per_class[name]["ap50"] = float(valid_at_50.mean()) if valid_at_50.size else None
        per_class[name]["ap50_95"] = float(valid.mean()) if valid.size else None
    return {
        "coco_ap_50_95": float(evaluator.stats[0]),
        "coco_ap_50": float(evaluator.stats[1]),
        "coco_per_class": per_class,
    }


def match_detections(boxes, labels, true_boxes, true_labels):
    matched, false_positives = set(), []
    for index, (box, label) in enumerate(zip(boxes, labels)):
        candidates = [i for i, actual in enumerate(true_labels) if actual == label and i not in matched]
        if candidates:
            overlaps = iou(box, true_boxes[candidates])
            best = int(overlaps.argmax())
            if overlaps[best] >= MATCH_IOU:
                matched.add(candidates[best])
                continue
        false_positives.append(index)
    return matched, false_positives


def operating_point(threshold, true_positives, false_positives, ground_truth_count):
    predictions = true_positives + false_positives
    return {
        "threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": ground_truth_count - true_positives,
        "precision": true_positives / predictions if predictions else 0.0,
        "recall": true_positives / ground_truth_count if ground_truth_count else 0.0,
        "f1": 2 * true_positives / (predictions + ground_truth_count) if predictions + ground_truth_count else 0.0,
    }


def precision_recall_curve(scores, matches, ground_truth_count):
    scores = np.asarray(scores)
    matches = np.asarray(matches, dtype=bool)
    points = []
    for percent in range(1, 101):
        threshold = percent / 100
        accepted = scores >= threshold
        tp = int(np.count_nonzero(accepted & matches))
        fp = int(np.count_nonzero(accepted & ~matches))
        points.append(operating_point(threshold, tp, fp, ground_truth_count))
    return points


def save_examples(output, root, examples, key):
    selected, classes, seen = [], set(), set()
    for example in examples:
        file = example["file"]
        label = CLASSES.index(example["class"])
        if file in seen or label in classes:
            continue
        classes.add(label)
        seen.add(file)
        count = len(selected) + 1
        directory = Path(output).with_suffix("")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}-{count:02}.png"
        caption = f'{key}: {example["class"].replace("_", " ")}'
        if "score" in example:
            caption += f' {example["score"]:.2f}'
        with Image.open(root / file) as image:
            annotate(image, [example["box"]], [label], [caption]).save(path)
        selected.append({**example, "overlay": f"{directory.name}/{path.name}"})
    return selected


def evaluate(manifest, checkpoint, output, *, split, threshold=0.25, device="cpu", tune_threshold=False, flip_ensemble=False):
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be in (0, 1]")
    if tune_threshold and split != "val":
        raise ValueError("Threshold tuning requires the validation split")
    root, rows = read_split(manifest, split)
    runtime = load_torch(checkpoint, device=device, flip_ensemble=flip_ensemble)
    images, annotations, predictions, false_examples, missed_examples = [], [], [], [], []
    observations, tuning_scores, tuning_matches = [], [], []
    counts = {label: {"ground_truth": 0, "matched": 0, "false_positives": 0} for label in CLASSES}
    for image_id, row in enumerate(rows, 1):
        width, height = row["size"]
        images.append({"id": image_id, "width": width, "height": height})
        for box, label in zip(row["boxes"], row["labels"]):
            coordinates = xywh(box)
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": label + 1,
                "bbox": coordinates, "area": coordinates[2] * coordinates[3], "iscrowd": 0,
            })
            counts[CLASSES[label]]["ground_truth"] += 1
        with Image.open(root / row["file"]) as image:
            raw = runtime.run(preprocess(image, runtime.size))
        boxes, scores, labels = decode(raw, (width, height), threshold=min(AP_THRESHOLD, threshold))
        for box, score, label in zip(boxes, scores, labels):
            if score >= AP_THRESHOLD:
                predictions.append({
                    "image_id": image_id, "category_id": int(label) + 1, "bbox": xywh(box), "score": float(score),
                })
        observations.append((row, boxes, scores, labels))
        if tune_threshold:
            # Matching visits highest scores first, so dropping low scores keeps the earlier matches.
            _, rejected = match_detections(boxes, labels, np.asarray(row["boxes"]), row["labels"])
            accepted = np.ones(len(scores), dtype=bool)
            accepted[rejected] = False
            tuning_scores.extend(scores)
            tuning_matches.extend(accepted)

    if tune_threshold:
        sweep = precision_recall_curve(tuning_scores, tuning_matches, len(annotations))
        selected = max(sweep, key=lambda point: (point["f1"], point["precision"], point["threshold"]))
        threshold = selected["threshold"] if selected["f1"] else threshold

    for row, boxes, scores, labels in observations:
        keep = scores >= threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        true_boxes = np.asarray(row["boxes"])
        matched, false_positives = match_detections(boxes, labels, true_boxes, row["labels"])
        for index, label in enumerate(row["labels"]):
            defect = CLASSES[label]
            if index in matched:
                counts[defect]["matched"] += 1
            else:
                missed_examples.append({"file": row["file"], "class": defect, "box": row["boxes"][index]})
        for index in false_positives:
            defect = CLASSES[labels[index]]
            counts[defect]["false_positives"] += 1
            false_examples.append({
                "file": row["file"], "class": defect, "box": boxes[index].tolist(), "score": float(scores[index]),
            })
    coco = coco_report(images, annotations, predictions)
    for row in counts.values():
        row["recall"] = row["matched"] / row["ground_truth"] if row["ground_truth"] else None
        row["misses"] = row["ground_truth"] - row["matched"]
    false_examples.sort(key=lambda row: row["score"], reverse=True)
    results = {
        "split": split,
        "image_count": len(images),
        "split_sha256": split_fingerprint(rows),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "flip_ensemble": flip_ensemble,
        **coco,
        "ap_score_floor": AP_THRESHOLD,
        "error_score_threshold": threshold,
        "error_match_iou": MATCH_IOU,
        "micro_at_threshold": operating_point(
            threshold, sum(row["matched"] for row in counts.values()),
            sum(row["false_positives"] for row in counts.values()), len(annotations),
        ),
        "per_class": counts,
        "false_positive_examples": save_examples(output, root, false_examples, "false-positive"),
        "miss_examples": save_examples(output, root, missed_examples, "miss"),
    }
    if tune_threshold:
        results["threshold_sweep"] = sweep
    write_json(output, results)
    print(f"{split}: {len(images)} images, mAP50:95={coco['coco_ap_50_95']:.4f}, "
          f"AP50={coco['coco_ap_50']:.4f}")
    return results


def compare(manifest, checkpoint, onnx, output, *, samples=60, threads=1, threshold=0.25, flip_ensemble=False):
    import onnxruntime
    import torch

    if samples < 1 or threads < 1 or not 0 < threshold <= 1:
        raise ValueError("Positive sample/thread counts and a threshold in (0, 1] are required")
    root, rows = read_split(manifest, "test")
    # Spread the fixed sample over the class-sorted manifest.
    sampled = np.linspace(0, len(rows) - 1, min(samples, len(rows)), dtype=int)
    runtimes = [load_torch(checkpoint, threads=threads, flip_ensemble=flip_ensemble), load_onnx(onnx, threads=threads)]
    if runtimes[0].size != runtimes[1].size:
        raise ValueError("Checkpoint and ONNX model expect different input sizes")
    size = runtimes[0].size
    with Image.open(root / rows[0]["file"]) as image:
        for runtime in runtimes:
            for _ in range(5):
                decode(runtime.run(preprocess(image, size)), image.size, threshold=threshold)
    times = [[], []]
    raw_error = 0.0
    decoded_box_error = 0.0
    decoded_score_error = 0.0
    nonempty_images = 0
    failures = []
    for index, sample in enumerate(sampled):
        row = rows[sample]
        raw, decoded = [None, None], [None, None]
        with Image.open(root / row["file"]) as opened:
            image = opened.convert("L")
        for runtime_index in (index % 2, 1 - index % 2):
            started = perf_counter()
            raw[runtime_index] = runtimes[runtime_index].run(preprocess(image, size))
            decoded[runtime_index] = decode(raw[runtime_index], image.size, threshold=threshold)
            times[runtime_index].append(1000 * (perf_counter() - started))
        raw_error = max(raw_error, float(np.max(np.abs(raw[0] - raw[1]))))
        raw_matches = np.allclose(raw[0], raw[1], atol=1e-4, rtol=1e-4)
        boxes0, scores0, labels0 = decoded[0]
        boxes1, scores1, labels1 = decoded[1]
        nonempty_images += bool(len(labels0))
        labels_match = np.array_equal(labels0, labels1)
        if labels_match and len(labels0):
            decoded_box_error = max(decoded_box_error, float(np.max(np.abs(boxes0 - boxes1))))
            decoded_score_error = max(decoded_score_error, float(np.max(np.abs(scores0 - scores1))))
        if not raw_matches or not labels_match or not np.allclose(boxes0, boxes1, atol=0.01, rtol=0) or not np.allclose(scores0, scores1, atol=1e-5, rtol=0):
            failures.append(row["file"])
    report = {
        "passed": not failures,
        "sample_count": len(sampled),
        "images_with_detections": nonempty_images,
        "failures": failures,
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "threads_per_runtime": threads,
        "torch_version": torch.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "flip_ensemble": flip_ensemble,
        "input_size": [size, size],
        "threshold": threshold,
        "timed_steps": ["preprocess decoded image", "inference", "decode and NMS"],
        "max_raw_absolute_error": raw_error,
        "max_matching_box_absolute_error_px": decoded_box_error,
        "max_matching_score_absolute_error": decoded_score_error,
        "latency_ms": {
            name: {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
            for name, values in zip(("pytorch", "onnx"), times)
        },
    }
    write_json(output, report)
    print(f"Runtime parity {'passed' if report['passed'] else 'failed'} on {len(sampled)} images")
    return report
