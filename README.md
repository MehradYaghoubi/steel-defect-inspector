# Steel Defect Inspector

Upload a grayscale or color image of a steel strip to see the detector's bounding boxes, defect names, and confidence scores. The browser demo runs on a local CPU using ONNX Runtime.

On the same **270 NEU-DET test images**, the updated detector improved **mAP@50:95 from 18.12% to 19.51%** and **AP@50 from 46.77% to 52.13%**. With each confidence cutoff selected on validation, test **F1 rose from 45.79% to 55.04%**, with 56.50% precision and 53.66% recall. [Experiment report](EXPERIMENTS.md)

The six classes are crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches. The benchmark uses the bounding-box annotations from the NEU-DET release. The maintainers describe 1,800 grayscale images at 200 × 200 pixels. [NEU surface defect database](https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm)

## Run the demo

If you have the local `runs/detector.onnx` file from training, start with the demo command below. In a fresh checkout, train and export a checkpoint first. The demo needs only the ONNX model and the demo dependencies.

```bash
python -m pip install -r requirements-demo.txt
python app.py --model runs/detector.onnx --threshold 34
```

Open the local URL, upload a JPG, PNG, BMP, or TIFF, and select **Inspect image**. Adjust **Minimum confidence** and run the inspection again to see how the detections change. The model sees a resized grayscale image. The demo draws predictions in the original image's orientation and aspect ratio.

## Reproduce training

Use Python 3.14 in a virtual environment. For CPU training, install the PyTorch CPU wheel first.

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For NVIDIA training, install the corresponding CUDA PyTorch wheel and add `--device cuda` to the training command. Download **NEU-DET** from the maintainers and extract the original images and Pascal VOC XML under `data/NEU-DET`. [Official download links](https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm)

```bash
python -m steel_inspector prepare --source data/NEU-DET --manifest data/split.json --seed 42
python -m steel_inspector train --manifest data/split.json --output runs/resize128-e24.pt --size 128 --epochs 24 --batch-size 32 --seed 42
python -m steel_inspector evaluate --manifest data/split.json --checkpoint runs/resize128-e24.pt --split val --output runs/ensemble-val.json --tune-threshold --flip-ensemble
```

Preparation pairs XML to photos by filename and converts VOC boxes to zero-based half-open coordinates. It removes repeated boxes, collapses pixel-identical images with equivalent annotations, and excludes images whose identical pixels have conflicting annotations. The seeded 70/15/15 train, validation, and test assignment is stratified by NEU-DET filename class and stored in `data/split.json` with image hashes, labels, geometry, and annotation audit counts.

The model trains from scratch using AdamW and a cosine schedule. It predicts class logits and box offsets on a stride-8 grid, combines two spatial feature scales, and uses focal loss plus a box regression loss. `--flip-ensemble` averages predictions from the original and horizontally mirrored inputs after mapping them to the original coordinates. Export embeds both passes inside the ONNX model.

## Evaluate and export

See [the experiment report](EXPERIMENTS.md) for the measured run and the exact training settings.

Use the checkpoint with the highest validation mAP@50:95. Read `error_score_threshold` in its validation report and carry that cutoff into test and runtime comparison. The documented run selected `0.34`.

```bash
python -m steel_inspector evaluate --manifest data/split.json --checkpoint runs/resize128-e24.pt --split test --output runs/ensemble-test.json --threshold 0.34 --flip-ensemble
python -m steel_inspector export --checkpoint runs/resize128-e24.pt --output runs/detector.onnx --flip-ensemble
python -m steel_inspector compare --manifest data/split.json --checkpoint runs/resize128-e24.pt --onnx runs/detector.onnx --output runs/ensemble-runtime.json --threshold 0.34 --flip-ensemble
```

- Evaluation uses COCO mAP@50:95 and AP@50, a 0.01 candidate score floor, 0.50 class-wise NMS, and at most 100 output boxes per image. For operating-point precision and recall, predictions match ground-truth boxes one-to-one within class at IoU ≥ 0.50. `--tune-threshold` sweeps confidence cutoffs on validation and selects the cutoff with highest micro-F1. Reports include false-positive and missed-box overlays when errors occur. [COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- The CPU comparison checks raw output, final box coordinates, classes, and confidence scores on the same fixed test images. It reports p50 and p95 batch-one latency, including preprocessing, inference, and NMS, after five warm-up passes per runtime. Image file decoding and drawing occur outside the measured interval. Comparison exits with a nonzero status if runtime parity fails.
- The default confidence score is a model output rather than a calibrated probability. NEU-DET is a benchmark for images containing defects. Deployment on a line would also require measurement on defect-free strips, the installed camera, and the production lighting.
- The official dataset page requests citations for use. Check redistribution terms for dataset images, overlays, and derived weights before publishing them. Local model and data artifacts are ignored by Git. [Dataset citation notice](https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm)

## Tests

```bash
python -m pip install pytest==9.1.1
python -m pytest -q
```

The CPU tests use generated grayscale fixtures. They cover reproducible splits, box mirroring, NMS, validation-only threshold selection, a one-epoch training run, and PyTorch–ONNX parity for both single-pass and mirrored inference.
