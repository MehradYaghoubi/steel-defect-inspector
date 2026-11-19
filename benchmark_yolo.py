"""Score a trained Ultralytics detector on the project's audited NEU-DET split."""

import argparse
import hashlib
import shutil
from pathlib import Path

from steel_inspector.data import CLASSES, read_split, write_json
from steel_inspector.evaluation import AP_THRESHOLD, coco_report, split_fingerprint, xywh
from steel_inspector.inference import MAX_DETECTIONS, NMS_IOU


def texture_image(path):
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot decode {path}")
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    residual = cv2.addWeighted(gray, 2.0, blur, -2.0, 128)
    # Ultralytics converts OpenCV BGR inputs to RGB before inference.
    return cv2.merge((residual, contrast, gray))


def prepare(manifest, destination, representation):
    splits = {split: read_split(manifest, split) for split in ("train", "val")}
    destination = destination.resolve()
    destination.mkdir(parents=True)
    for split, (root, rows) in splits.items():
        images = destination / "images" / split
        labels = destination / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for row in rows:
            source = root / row["file"]
            suffix = ".png" if representation == "texture" else source.suffix
            image = images / f"{source.stem}{suffix}"
            label_file = labels / f"{image.stem}.txt"
            if image.exists() or label_file.exists():
                raise ValueError(f"Duplicate filename in {split}: {source.name}")
            if representation == "texture":
                import cv2

                encoded, payload = cv2.imencode(".png", texture_image(source))
                if not encoded:
                    raise ValueError(f"Cannot encode {source}")
                image.write_bytes(payload.tobytes())
            else:
                shutil.copyfile(source, image)
            width, height = row["size"]
            lines = []
            for (x1, y1, x2, y2), label in zip(row["boxes"], row["labels"], strict=True):
                box = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                       (x2 - x1) / width, (y2 - y1) / height)
                lines.append(f"{label} " + " ".join(f"{value:.9f}" for value in box))
            label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = {
        "path": destination.as_posix(), "train": "images/train",
        "val": "images/val", "names": CLASSES,
    }
    write_json(destination / "dataset.yaml", config)


def evaluate(
    manifest, checkpoint, split, imgsz, device, representation, *,
    tta=False, flip_horizontal=False, predictions_output=None,
):
    from ultralytics import YOLO, __version__
    from ultralytics.utils.patches import imread

    root, rows = read_split(manifest, split)
    model = YOLO(str(checkpoint))
    if model.task != "detect" or tuple(model.names.values()) != CLASSES:
        raise ValueError("Checkpoint must be a detector with the NEU-DET class order")
    if tta and getattr(model.model, "end2end", False):
        raise ValueError("Ultralytics TTA requires a detector with an NMS-based head")

    images, annotations, predictions = [], [], []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        paths = [root / row["file"] for row in batch]
        sources = [texture_image(path) for path in paths] if representation == "texture" else list(map(str, paths))
        if flip_horizontal:
            if representation == "raw":
                sources = [imread(str(path)) for path in paths]
            for index, (path, source) in enumerate(zip(paths, sources, strict=True)):
                if source is None:
                    raise ValueError(f"Cannot decode {path}")
                sources[index] = source[:, ::-1].copy()
        results = model.predict(
            sources,
            imgsz=imgsz, conf=AP_THRESHOLD, iou=NMS_IOU,
            max_det=MAX_DETECTIONS, agnostic_nms=False, device=device, verbose=False, augment=tta,
        )
        for image_id, (row, result) in enumerate(zip(batch, results, strict=True), start + 1):
            width, height = row["size"]
            if result.orig_shape != (height, width):
                raise ValueError(f"Unexpected image dimensions for {row['file']}")
            images.append({"id": image_id, "width": width, "height": height})
            for box, label in zip(row["boxes"], row["labels"], strict=True):
                coordinates = xywh(box)
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": label + 1, "bbox": coordinates,
                    "area": coordinates[2] * coordinates[3], "iscrowd": 0,
                })
            boxes = result.boxes.cpu()
            for box, score, label in zip(boxes.xyxy, boxes.conf, boxes.cls, strict=True):
                x1, y1, x2, y2 = box.tolist()
                if flip_horizontal:
                    x1, x2 = width - x2, width - x1
                coordinates = xywh([x1, y1, x2, y2])
                if min(coordinates[2:]) > 0:
                    predictions.append({
                        "image_id": image_id, "category_id": int(label) + 1,
                        "bbox": coordinates, "score": float(score),
                    })

    report = {
        "split": split, "image_count": len(images), "annotation_count": len(annotations),
        "split_sha256": split_fingerprint(rows),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "ultralytics_version": __version__, "input_size": imgsz, "device": device,
        "representation": representation, "tta": tta, "flip_horizontal": flip_horizontal,
        "ap_score_floor": AP_THRESHOLD, "nms_iou": NMS_IOU, "max_detections": MAX_DETECTIONS,
        **coco_report(images, annotations, predictions),
    }
    if predictions_output is not None:
        write_json(predictions_output, {
            "report": report, "images": images, "annotations": annotations, "predictions": predictions,
        })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--representation", choices=("raw", "texture"), default="raw")
    commands = parser.add_subparsers(dest="command", required=True)
    training_data = commands.add_parser("prepare")
    training_data.add_argument("--destination", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=("val", "test"), required=True)
    evaluation.add_argument("--imgsz", type=int, required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--tta", action="store_true", help="Enable Ultralytics test-time augmentation")
    evaluation.add_argument("--flip-horizontal", action="store_true", help="Score a horizontally reflected image view")
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--predictions-output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.destination, args.representation)
    else:
        inputs = {args.manifest.resolve(), args.checkpoint.resolve()}
        outputs = [path.resolve() for path in (args.output, args.predictions_output) if path is not None]
        if inputs.intersection(outputs) or len(set(outputs)) != len(outputs):
            parser.error("Output paths must be distinct and must not replace the manifest or checkpoint")
        report = evaluate(
            args.manifest, args.checkpoint, args.split, args.imgsz, args.device, args.representation,
            tta=args.tta, flip_horizontal=args.flip_horizontal, predictions_output=args.predictions_output,
        )
        write_json(args.output, report)
        print(f"{args.split}: mAP50:95={report['coco_ap_50_95']:.4f}, AP50={report['coco_ap_50']:.4f}")
