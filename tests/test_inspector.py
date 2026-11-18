from base64 import b64decode
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app import inspect_image
from steel_inspector.data import CLASSES, horizontal_flip, prepare
from steel_inspector.evaluation import coco_ap, compare, evaluate, match_detections, precision_recall_curve
from steel_inspector.inference import decode, load_onnx, load_torch, predict
from steel_inspector.model import export_model, train


@pytest.fixture
def fixture_dataset(tmp_path):
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "annotations").mkdir()
    for class_index, defect in enumerate(CLASSES):
        for item in range(3):
            stem = f"{defect}_{item}"
            Image.new("L", (64, 64), 10 * class_index + item).save(source / "images" / f"{stem}.png")
            (source / "annotations" / f"{stem}.xml").write_text(
                f"<annotation><filename>{stem}.png</filename>"
                "<size><width>64</width><height>64</height></size>"
                f"<object><name>{defect}</name><bndbox>"
                "<xmin>9</xmin><ymin>13</ymin><xmax>32</xmax><ymax>44</ymax>"
                "</bndbox></object></annotation>",
                encoding="utf-8",
            )
    return source


def test_prepare_has_stable_disjoint_stratified_splits(fixture_dataset):
    path = fixture_dataset.parent / "split.json"
    first = prepare(fixture_dataset, path, seed=17)
    assert first == prepare(fixture_dataset, path, seed=17)
    for split in ("train", "val", "test"):
        records = [row for row in first["images"] if row["split"] == split]
        assert {row["labels"][0] for row in records} == set(range(6))
        assert all(row["boxes"] == [[8, 12, 32, 44]] for row in records)
    assert len({row["sha256"] for row in first["images"]}) == len(first["images"]) == 18


def test_prepare_excludes_conflicting_identical_pixels_and_audits_boxes(fixture_dataset):
    images = fixture_dataset / "images"
    annotations = fixture_dataset / "annotations"
    Image.new("L", (64, 64), 250).save(images / "conflict_a.png")
    (images / "conflict_b.png").write_bytes((images / "conflict_a.png").read_bytes())
    original = (annotations / "crazing_0.xml").read_text()
    (annotations / "conflict_a.xml").write_text(original.replace("crazing_0.png", "conflict_a.png"))
    (annotations / "conflict_b.xml").write_text(
        original.replace("crazing_0.png", "conflict_b.png").replace("<xmin>9", "<xmin>11")
    )
    distinct_box = original[original.index("<object>"):original.index("</annotation>")]
    (annotations / "crazing_0.xml").write_text(original.replace("</annotation>", distinct_box + "</annotation>"))
    results = prepare(fixture_dataset, fixture_dataset.parent / "split.json")
    assert len(results["images"]) == 18
    audit = results["annotation_audit"]
    assert audit["removed_duplicate_boxes"] == 1
    assert len(audit["excluded_conflicting_digests"]) == 1
    assert audit["excluded_distinct_annotation_records"] == 2


def test_decode_removes_same_class_duplicate_only():
    outputs = np.full((1, 10, 2, 2), -12, dtype=np.float32)
    outputs[:, 6:] = 0
    outputs[0, 6, 0, 0] = np.arctanh(1 / 3)
    outputs[0, 6, 0, 1] = np.arctanh(-1 / 3)
    outputs[0, 0, 0, 0] = 6
    outputs[0, 0, 0, 1] = 4
    outputs[0, 1, 0, 1] = 5
    boxes, scores, labels = decode(outputs, (80, 60), threshold=0.5)
    assert labels.tolist() == [0, 1]
    np.testing.assert_allclose(boxes, [[20, 0, 60, 30]] * 2, atol=0.001)
    assert scores[0] > scores[1] > 0.5


def test_flip_preserves_class_box_with_mirrored_pixels():
    source = Image.new("L", (64, 32), 0)
    source.putpixel((4, 10), 255)
    flipped, boxes = horizontal_flip(source, np.asarray([[4, 10, 8, 12]]))
    assert flipped.getpixel((59, 10)) == 255
    assert boxes.tolist() == [[56, 10, 60, 12]]


def test_duplicate_and_wrong_class_predictions_are_errors():
    boxes = np.asarray([[0, 0, 10, 10]] * 3)
    matched, errors = match_detections(boxes, [1, 0, 0], boxes[:1], [0])
    assert matched == {0}
    assert errors == [0, 2]


def test_threshold_sweep_counts_false_alarms_and_misses():
    points = precision_recall_curve([0.80, 0.60, 0.50, 0.30], [True, False, True, False], 3)
    assert points[49] == {
        "threshold": 0.50, "true_positives": 2, "false_positives": 1, "false_negatives": 1,
        "precision": pytest.approx(2 / 3), "recall": pytest.approx(2 / 3), "f1": pytest.approx(2 / 3),
    }
    assert points[99]["precision"] == 0
    assert points[99]["recall"] == 0


def test_perfect_prediction_reaches_full_coco_ap():
    ap, ap50 = coco_ap(
        [{"id": 1, "width": 20, "height": 20}],
        [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
        [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.99}],
    )
    assert ap == pytest.approx(1)
    assert ap50 == pytest.approx(1)


@pytest.mark.parametrize("flip_ensemble", [False, True])
def test_cpu_train_export_and_inference_agree(fixture_dataset, flip_ensemble):
    manifest = fixture_dataset.parent / "split.json"
    checkpoint = fixture_dataset.parent / "detector.pt"
    onnx = fixture_dataset.parent / "detector.onnx"
    prepare(fixture_dataset, manifest, seed=17)
    train(manifest, checkpoint, epochs=1, batch_size=3, augment=True, size=64, seed=17, device="cpu")
    export_model(checkpoint, onnx, flip_ensemble=flip_ensemble)
    path = next((fixture_dataset / "images").glob("*.png"))
    with Image.open(path) as sample:
        torch_output = predict(load_torch(checkpoint, flip_ensemble=flip_ensemble), sample, threshold=0.01)
        onnx_output = predict(load_onnx(onnx), sample, threshold=0.01)
    assert len(torch_output[0]) > 0
    for first, second in zip(torch_output, onnx_output):
        np.testing.assert_allclose(first, second, atol=0.001)

    preview = inspect_image(load_onnx(onnx), path.read_bytes(), threshold=0.01)
    assert len(preview["detections"]) == len(onnx_output[0])
    with Image.open(BytesIO(b64decode(preview["overlay"].split(",", 1)[1]))) as rendered:
        assert rendered.mode == "RGB"
        assert rendered.size == (192, 192)

    report = evaluate(manifest, checkpoint, fixture_dataset.parent / "eval.json", split="test")
    assert report["image_count"] == 6
    assert len(report["miss_examples"]) == 6
    assert 0 <= report["coco_ap_50_95"] <= 1
    assert report["micro_at_threshold"]["false_negatives"] == 6
    with pytest.raises(ValueError, match="validation"):
        evaluate(manifest, checkpoint, fixture_dataset.parent / "tuned.json", split="test", tune_threshold=True)
    tuned = evaluate(manifest, checkpoint, fixture_dataset.parent / "tuned.json", split="val", tune_threshold=True)
    assert len(tuned["threshold_sweep"]) == 100
    assert tuned["micro_at_threshold"]["threshold"] == tuned["error_score_threshold"]

    parity = compare(manifest, checkpoint, onnx, fixture_dataset.parent / "compare.json", threshold=0.01, flip_ensemble=flip_ensemble)
    assert parity["passed"]
    assert parity["images_with_detections"] == 6
