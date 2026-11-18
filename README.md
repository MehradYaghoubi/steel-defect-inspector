# Steel Defect Inspector

Upload a grayscale or color image of a steel strip to see the detector's bounding boxes, defect names, and confidence scores. The browser demo runs on a local CPU using ONNX Runtime.

The six classes are crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches. The benchmark uses the bounding-box annotations from the NEU-DET release. The maintainers describe 1,800 grayscale images at 200 × 200 pixels. [NEU surface defect database](https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm)

## Run the demo

If you have the local `runs/detector.onnx` file from training, start with the demo command below. In a fresh checkout, train and export a checkpoint first. The demo needs only the ONNX model and the demo dependencies.

```bash
python -m pip install -r requirements-demo.txt
python app.py --model runs/detector.onnx
```

Open the local URL, upload a JPG, PNG, BMP, or TIFF, and select **Inspect image**. Adjust **Minimum confidence** and run the inspection again to see how the detections change. The model sees a resized grayscale image. The demo draws predictions in the original image's orientation and aspect ratio.

## Reproduce training

Use Python 3.14 in a virtual environment. For CPU training, install the PyTorch CPU wheel first.

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For NVIDIA training, install the corresponding CUDA PyTorch wheel and add `--device cuda` to the two training commands. Download **NEU-DET** from the maintainers and extract the original images and Pascal VOC XML under `data/NEU-DET`. [Official download links](https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm)

```bash
python -m steel_inspector prepare --source data/NEU-DET --manifest data/split.json --seed 42
python -m steel_inspector train --manifest data/split.json --output runs/baseline.pt
python -m steel_inspector train --manifest data/split.json --output runs/flip.pt --augment
python -m steel_inspector evaluate --manifest data/split.json --checkpoint runs/baseline.pt --split val --output runs/baseline-val.json
python -m steel_inspector evaluate --manifest data/split.json --checkpoint runs/flip.pt --split val --output runs/flip-val.json
```

Preparation pairs XML to photos by filename and converts VOC boxes to zero-based half-open coordinates. It removes repeated boxes, collapses pixel-identical images with equivalent annotations, and excludes images whose identical pixels have conflicting annotations. The seeded 70/15/15 train, validation, and test assignment is stratified by NEU-DET filename class and stored in `data/split.json` with image hashes, labels, geometry, and annotation audit counts.

The model trains from scratch on 192 × 192 grayscale inputs with AdamW and a cosine schedule for 30 epochs by default. It predicts class logits and box offsets on a stride-8 grid, combines two spatial feature scales, and uses focal loss plus a box regression loss. The augmentation run adds 50% probability of a horizontal flip and mirrors the boxes along with the pixels. Set `--size`, `--epochs`, `--batch-size`, or `--seed` to rerun a documented experiment with its exact settings.

## Evaluate and export

See [the experiment report](EXPERIMENTS.md) for the measured run and the exact training settings.

Use the checkpoint with the higher validation mAP50:95 for the test, export, and CPU comparison. Substitute `runs/flip.pt` in these commands if it won the validation comparison.

```bash
python -m steel_inspector evaluate --manifest data/split.json --checkpoint runs/baseline.pt --split test --output runs/test.json
python -m steel_inspector export --checkpoint runs/baseline.pt --output runs/detector.onnx
python -m steel_inspector compare --manifest data/split.json --checkpoint runs/baseline.pt --onnx runs/detector.onnx --output runs/runtime.json
```

- Evaluation uses COCO mAP50:95 and AP50, a 0.01 candidate score floor, 0.50 class-wise NMS, and at most 100 output boxes per image. The default threshold for per-class recall and error counts is 0.25. Detections match ground-truth boxes one-to-one within class at IoU ≥ 0.50. Each report includes paths to local false-positive and missed-box overlays when errors occur. [COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- The CPU comparison checks raw output, final box coordinates, classes, and confidence scores on the same fixed test images. It reports p50 and p95 batch-one latency, including preprocessing, inference, and NMS, after five warm-up passes per runtime. Image file decoding and drawing occur outside the measured interval. Comparison exits with a nonzero status if runtime parity fails.
- The default confidence score is a model output rather than a calibrated probability. NEU-DET is a benchmark for images containing defects. Deployment on a line would also require measurement on defect-free strips, the installed camera, and the production lighting.
- The official dataset page requests citations for use. Check redistribution terms for dataset images, overlays, and derived weights before publishing them. Local model and data artifacts are ignored by Git. [Dataset citation notice](https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm)

## Tests

```bash
python -m pip install pytest==9.1.1
python -m pytest -q
```

The CPU tests use generated grayscale fixtures. They cover split reproducibility, box mirroring, NMS, metric matching, a one-epoch training run, ONNX export, and PyTorch–ONNX inference parity without downloading NEU-DET.
