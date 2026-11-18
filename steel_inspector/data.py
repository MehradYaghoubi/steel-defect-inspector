import hashlib
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
STRIDE = 8
CENTER_RANGE = 1.5


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(source, manifest, *, seed=42):
    source, manifest = Path(source).resolve(strict=True), Path(manifest).resolve()
    paths = defaultdict(list)
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            paths[path.stem.lower()].append(path)

    by_digest = defaultdict(dict)
    collapsed_images = duplicate_boxes = 0
    for annotation in sorted(source.rglob("*.xml")):
        xml = ET.parse(annotation).getroot()
        stem = Path((xml.findtext("filename") or annotation.name).replace("\\", "/")).stem.lower()
        matches = paths.get(stem) or paths.get(annotation.stem.lower(), [])
        if len(matches) != 1:
            raise ValueError(f"{annotation.name} matches {len(matches)} images")
        path = matches[0]
        with Image.open(path) as image:
            image = image.convert("L")
            width, height = image.size
            digest = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()

        objects = set()
        for obj in xml.findall("object"):
            name = (obj.findtext("name") or "").strip().lower().replace(" ", "_")
            name = "rolled-in_scale" if name == "rolled_in_scale" else name
            if name not in CLASSES:
                raise ValueError(f"Unknown label {name!r} in {annotation}")
            x1, y1, x2, y2 = (int(obj.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax"))
            # VOC uses one-based inclusive coordinates. Store zero-based, half-open boxes.
            box = [max(0, x1 - 1), max(0, y1 - 1), min(width, x2), min(height, y2)]
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError(f"Empty box {box} in {annotation}")
            key = (CLASSES.index(name), tuple(box))
            duplicate_boxes += key in objects
            objects.add(key)
        if not objects:
            raise ValueError(f"No objects in {annotation}")

        signature = tuple(sorted(objects))
        if signature in by_digest[digest]:
            collapsed_images += 1
            continue
        by_digest[digest][signature] = {
            "file": path.relative_to(source).as_posix(),
            "sha256": digest,
            "size": [width, height],
            "boxes": [list(box) for _, box in signature],
            "labels": [label for label, _ in signature],
        }

    if not by_digest:
        raise ValueError(f"No VOC annotations found under {source}")
    groups = defaultdict(list)
    conflicting_digests = [digest for digest, variants in sorted(by_digest.items()) if len(variants) > 1]
    for _, variants in sorted(by_digest.items()):
        if len(variants) == 1:
            variant = next(iter(variants.values()))
            stem = Path(variant["file"]).stem.lower()
            group = next((label for label, name in enumerate(CLASSES) if stem.startswith(f"{name}_")), variant["labels"][0])
            groups[group].append(variant)
    if not groups:
        raise ValueError(f"No consistently annotated images found under {source}")
    rng = random.Random(seed)
    records = []
    for label, group in sorted(groups.items()):
        if len(group) < 3:
            raise ValueError(f"At least three unique images are required for the {CLASSES[label]} label group")
        rng.shuffle(group)
        held_out = max(1, round(len(group) * 0.15))
        for index, record in enumerate(group):
            split = "test" if index < held_out else "val" if index < 2 * held_out else "train"
            records.append({**record, "split": split})
    try:
        relative_source = Path(os.path.relpath(source, manifest.parent)).as_posix()
    except ValueError:  # Windows paths on separate drives
        relative_source = source.as_posix()
    payload = {
        "version": 1, "seed": seed, "classes": list(CLASSES), "source": relative_source, "images": records,
        "annotation_audit": {
            "collapsed_identical_images": collapsed_images,
            "removed_duplicate_boxes": duplicate_boxes,
            "excluded_conflicting_digests": conflicting_digests,
            "excluded_distinct_annotation_records": sum(len(by_digest[digest]) for digest in conflicting_digests),
        },
    }
    write_json(manifest, payload)
    return payload


def read_split(manifest, split):
    manifest = Path(manifest).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload["version"] != 1 or payload["classes"] != list(CLASSES):
        raise ValueError("Unsupported split manifest")
    rows = [row for row in payload["images"] if row["split"] == split]
    if not rows:
        raise ValueError(f"No images in '{split}' split")
    return (manifest.parent / payload["source"]).resolve(), rows


def preprocess(image, size):
    grayscale = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(grayscale, dtype=np.float32)[None, None] / 255.0


def horizontal_flip(image, boxes):
    flipped = boxes.copy()
    flipped[:, [0, 2]] = image.width - boxes[:, [2, 0]]
    return ImageOps.mirror(image), flipped
