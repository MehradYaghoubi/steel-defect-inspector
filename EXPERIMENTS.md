# NEU-DET experiment

The official archive supplied 1,800 image and annotation pairs. The preparation audit removed three repeated boxes and excluded two records with identical decoded pixels and conflicting annotations. The seed-42 split contained 1,258 training images, 270 validation images, and 270 test images. The source archive SHA-256 was `cf04bb0b05d364f23e05031b191b20b908a0e5eb11466df74ee7fba98ec835d5`. [NEU-DET maintainers](https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm)

## 192 × 192 run

The 192 × 192 model trained for 35 epochs with batch size 24, seed 42 and horizontal flip augmentation. Evaluation used mirrored inference. Validation mAP@50:95 was 22.61% and AP@50 was 59.77%, up from 21.84% and 54.20% for the 128 × 128 model.

The confidence sweep used 0.01 increments on the 270 validation images. For the balanced cutoff, the sweep maximized micro-F1 and selected 0.34. For the high-precision cutoff, it maximized recall subject to the lower endpoint of a two-sided 95% Wilson interval for validation precision being at least 85%, using matched and unmatched prediction counts with `z = 1.96`. The sweep selected 0.58 with 86 matches and 5 unmatched predictions (94.51% precision, 13.78% recall, Wilson lower endpoint 87.78%). This rule adds a margin during threshold selection. Boxes from one image can be correlated.

Both cutoffs were fixed before the 192 × 192 evaluation on the existing 270-image test split. Matching uses class agreement and IoU ≥ 0.50.

| Test cutoff | Precision | Recall | F1 | Matched boxes | Unmatched predictions | Missed boxes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.34 | 62.10% | 56.71% | 59.28% | 372 | 227 | 284 |
| 0.58 | **88.99%** | **14.79%** | 25.36% | 97 | 12 | 559 |

Test AP@50 was 57.74% and mAP@50:95 was 21.67% for the same ranked predictions. At 0.34 the model matched 20 more boxes and returned 44 fewer unmatched predictions than the previous model at its balanced cutoff. The 0.58 cutoff suits prioritizing high-confidence alerts for review. It missed 559 of 656 annotated defects. The 85% target refers to observed micro-precision on the benchmark.

PyTorch and ONNX Runtime passed parity on 60 fixed test images at 0.58. On one CPU thread, ONNX Runtime took 8.11 ms p50 and 9.43 ms p95, including preprocessing and NMS. PyTorch took 15.38 ms p50 and 18.61 ms p95. Reproduction commands are in the [README](README.md).

## Previous runs

### Validation and model selection

Training ran on an Intel Core i5-9400F with Windows, CPython 3.14.7 and PyTorch 2.14.0+cpu. The 96 × 96 and 128 × 128 runs used batch size 32 and seed 42. Use `--size 96 --epochs 8` to reproduce the original configuration and add `--augment` for its training-time flip variant. The updated run used `--size 128 --epochs 24`, without training-time augmentation. Its two-pass variant averaged the original and mirrored grid predictions during inference.

| Training variant | Validation mAP@50:95 | Validation AP@50 |
| --- | ---: | ---: |
| Original 96 × 96 | 18.13% | 45.23% |
| Original with training-time flips | 17.72% | 46.58% |
| Updated 128 × 128, single pass | 20.41% | 53.26% |
| Updated 128 × 128, mirrored inference | 21.84% | 54.20% |

The updated model and inference mode were selected by validation mAP@50:95. For each model, a sweep of whole-percent cutoffs on validation maximized micro-F1, breaking ties by precision and then the higher cutoff. Test used the resulting fixed cutoffs.

### Test results

All three rows use the same 270 test images and 656 reference boxes. AP uses ranked predictions with a 0.01 score floor. The operating-point metrics use the cutoff shown, class agreement, and IoU ≥ 0.50.

| Model and cutoff | mAP@50:95 | AP@50 | Precision | Recall | F1 | Unmatched predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, initial cutoff 0.25 | 18.12% | 46.77% | 22.93% | 65.55% | 33.98% | 1,445 |
| Original, validation cutoff 0.37 | 18.12% | 46.77% | 47.84% | 43.90% | 45.79% | 314 |
| Updated mirrored model, validation cutoff 0.34 | **19.51%** | **52.13%** | **56.50%** | **53.66%** | **55.04%** | **271** |

Against the validation-tuned baseline, the update matched 64 additional boxes and produced 43 fewer unmatched predictions. The initial 0.25 cutoff remains useful as a reference for a higher-recall, lower-precision operating point.

### Test errors

The table below describes the updated model at the chosen 0.34 cutoff.

| Defect | Annotated boxes | Recall | Unmatched predictions | Missed boxes |
| --- | ---: | ---: | ---: | ---: |
| Crazing | 104 | 16.35% | 43 | 87 |
| Inclusion | 170 | 60.59% | 104 | 67 |
| Patches | 134 | 74.63% | 28 | 34 |
| Pitted surface | 61 | 36.07% | 2 | 39 |
| Rolled-in scale | 93 | 49.46% | 56 | 47 |
| Scratches | 94 | 68.09% | 38 | 30 |

Crazing had the lowest recall and the most missed boxes at the global cutoff. Inclusion produced the most unmatched predictions. `evaluate` generates marked examples alongside the local JSON report. NEU-DET measures performance on annotated defect images. A production assessment needs representative defect-free strips and the installed camera and lighting.

### CPU runtime

PyTorch and ONNX agreed within tolerance on 60 fixed test images for the updated two-pass model. The largest difference between matched box coordinates was 0.000069 pixels. Timing includes preprocessing, both passes, and decoding with NMS on one CPU thread after warm-up. Runtime versions were PyTorch 2.14.0+cpu and ONNX Runtime 1.32.0. The original single-pass ONNX run reported a 1.96 ms p95 on the same machine.

| Runtime | p50 | p95 |
| --- | ---: | ---: |
| PyTorch | 7.64 ms | 8.62 ms |
| ONNX Runtime | 3.57 ms | 3.90 ms |

The local ONNX model is `runs/detector.onnx`. Aggregate results and annotated samples are regenerated by the CLI, with the dataset, trained weights, and image overlays kept in Git-ignored folders.

## Research target

Match or exceed the strongest reproducible published detector on NEU-DET. Track COCO mAP@50:95 as the primary metric and mAP@50, AP by defect category, and CPU latency as secondary results. The published MBY result (85.8% mAP@50 and 56.2% mAP@50:95) gives a numerical target. Its scores and our current test scores are from separate evaluations. [MBY, Table 7](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349437/)

- Use the existing audited split for initial model development. Its manifest SHA-256 is 3e278f74bf136cbcd149a0281ed8e3b661df24f8ad9c0ec3c1d1819ebd4dcead. Select checkpoints and settings with the 270 validation images.
- Establish a pretrained YOLO11n baseline. Compare it with the current detector using the same annotations, COCO evaluator, confidence floor, NMS IoU and maximum detections per image.
- Use the released DFE-Net split, weights and evaluation script as a reproduction check. The authors also used the 360-image evaluation split for checkpoint selection. Record that result separately, audit for duplicate pixels across splits and seek reproducible releases from stronger published methods. [DFE-Net release](https://github.com/hoid1/DFE-Net/tree/936067614ef39325c91b85f77995f648305e08e9)
- Freeze settings before final test runs. For any claim of beating a named method, publish matched-protocol results, checkpoint and split identifiers, per-seed scores and uncertainty for the score difference. A benchmark-wide claim requires checking the strongest comparable work available at the time.

Initial split audit of the DFE-Net release matched all 1,800 filenames to the local NEU-DET archive. There was no filename overlap. Decoded grayscale pixel hashing found one cross-split duplicate pair, patches_105 in train and patches_101 in validation. Retain the published split for exact reproduction and report a separate duplicate-free evaluation for any held-out claim.

### YOLO11n development run

Run the YOLO11n development baseline in a separate environment with the project requirements and Ultralytics 8.4.11. Prepare the training and validation directories from the audited manifest, then train with the official COCO-pretrained YOLO11n weights. The preparation command requires a new destination and copies only train and validation. The evaluator uses the same COCO scoring code, 0.01 confidence floor, 0.50 class-wise NMS and 100 detections per image as the current detector.

~~~bash
python benchmark_yolo.py --manifest data/split.json prepare --destination runs/yolo-neudet
yolo detect train model=yolo11n.pt data=runs/yolo-neudet/dataset.yaml imgsz=256 epochs=16 batch=16 workers=0 seed=42 device=cpu amp=False plots=False project=runs name=yolo11n-seed42
python benchmark_yolo.py --manifest data/split.json evaluate --checkpoint runs/detect/runs/yolo11n-seed42/weights/best.pt --split val --imgsz 256 --output runs/yolo11n-seed42-val.json
~~~

The 16-epoch CPU run is an initial development comparison. Under the shared COCO protocol, the epoch-1 checkpoint scored 18.41% mAP@50 and 8.06% mAP@50:95 on validation. The epoch-3 checkpoint reached 36.12% and 14.98%, respectively. These interim scores track the run in progress. Checkpoints and settings for the final runs are chosen using validation results.

### Matched validation diagnostics

The epoch-7 YOLO11n snapshot reached 65.63% mAP@50 and 30.00% mAP@50:95, gains of 5.86 and 7.39 percentage points over the current 192 px detector on the same validation images. The reports share the same ordered split hash and COCO evaluator. The release test results above are from the existing detector.

| Validation AP@50 | Current 192 px detector | YOLO11n, epoch 5 of 16 | YOLO11n, epoch 7 of 16 |
| --- | ---: | ---: | ---: |
| Crazing | 27.0% | 6.1% | 27.8% |
| Inclusion | 62.7% | 55.1% | 67.6% |
| Patches | 86.7% | 68.1% | 90.0% |
| Pitted surface | 53.4% | 63.1% | 82.3% |
| Rolled-in scale | 58.5% | 41.3% | 53.5% |
| Scratches | 70.2% | 67.4% | 72.6% |

To test whether texture information improves detection of crazing, the texture pilot replaces replicated grayscale channels with grayscale, CLAHE and a Gaussian high-pass residual. It starts from the same YOLO11n weights and keeps the raw pilot's train/validation assignments, normalized annotations, epochs, resolution and seed. CPU thread counts differed between the two pilots. Inference applies the same deterministic transform as dataset preparation. Use the texture representation for both preparation and evaluation.

~~~bash
python benchmark_yolo.py --manifest data/split.json --representation texture prepare --destination runs/yolo-neudet-texture
yolo detect train model=yolo11n.pt data=runs/yolo-neudet-texture/dataset.yaml imgsz=256 epochs=16 batch=16 workers=0 seed=42 device=cpu amp=False plots=False project=runs name=yolo11n-texture-seed42
python benchmark_yolo.py --manifest data/split.json --representation texture evaluate --checkpoint runs/detect/runs/yolo11n-texture-seed42/weights/best.pt --split val --imgsz 256 --output runs/yolo11n-texture-val.json
~~~

The DFE-Net training split contains 217 of our validation images and 223 of our test images by filename. Direct comparisons on the project split require training DFE-Net from a checkpoint that has not previously seen NEU-DET.

### External DFE-Net reproduction

The published DFE-Net study evaluates 360 validation images and 882 boxes at input size 640, confidence floor 0.001, class-wise NMS IoU 0.7 and batch size 32. We reran the authors' validation script using the released images, annotations and FP32 weights from commit 936067614ef39325c91b85f77995f648305e08e9. The SHA-256 hashes of both checkpoints matched the release manifest. [Published paper](https://doi.org/10.3390/electronics15215196) · [Pinned reproduction repository](https://github.com/hoid1/DFE-Net/tree/936067614ef39325c91b85f77995f648305e08e9)

| Released checkpoint | Reference mAP@50 | Reproduced mAP@50 | Reference mAP@50:95 | Reproduced mAP@50:95 |
| --- | ---: | ---: | ---: | ---: |
| DFE-Net | 75.40% | 75.39% | 42.60% | 42.59% |
| YOLO11n | 72.60% | 72.67% | 41.10% | 41.12% |

The command was `python tools/val.py --model all --device cpu` on Windows with Python 3.14.4 and PyTorch 2.14.0+cpu. In the local Windows checkout we deferred imports of MMCV CARAFE and skipped imports of the legacy Mamba and DyHead variants, which neither checkpoint uses. The forward pass, annotations, thresholds and metric computation were unchanged. The source validation split is used for checkpoint selection and contains the duplicate pair noted above. These results are reported on the authors' split and scored by their evaluator. The project uses a different held-out split and COCO scoring.

A fresh model built from the released DFE-Net YAML had 2,246,720 trainable parameters. It completed an SGD update with finite gradients on four images and ten boxes from the project's training split. This check establishes that the architecture can be trained locally from fresh weights. Full matched-split training is pending.

### Follow-up validation snapshots

At epoch 9, raw YOLO11n reached 70.39% mAP@50 and 33.83% mAP@50:95 under the shared COCO protocol.

| Validation metric at epoch 3 | Raw YOLO11n | Texture YOLO11n |
| --- | ---: | ---: |
| mAP@50 | 36.12% | 40.83% |
| mAP@50:95 | 14.98% | 16.89% |
| Crazing AP@50 | 4.95% | 16.02% |
| Inclusion AP@50 | 47.38% | 43.93% |

Both runs used the same training and validation images, normalized boxes, initial checkpoint and seed. The raw run used four CPU threads and texture used two. These are interim results from one seed, collected before the two runs finished.

DFE-Net also completed a one-epoch engineering run from fresh weights, using a 25-image subset of project training and all 270 validation images. The run saved a checkpoint that the shared COCO evaluator could load and score. This run used 128 px input and only checks the training-to-evaluation path.

The next full GPU candidate is COCO-pretrained YOLO11s at 448 px, seed 42, up to 80 epochs, batch 8, AdamW with a 0.001 initial learning rate and cosine decay. The planned run was updated to use the texture representation before launch based on the final YOLO11n validation comparison below. The split and COCO evaluator stay fixed. The GPU recipe uses no hue or saturation shifts. CUDA is installed and training has started, as recorded below.

### Inference settings and uncertainty

On the raw epoch-9 checkpoint, default 256 px inference reached 70.39% mAP@50 and 33.83% mAP@50:95. Ultralytics test-time augmentation at 256 px reached 71.03% and 34.75%. Single-pass 320 px inference reached 63.15% and 25.03%. All three results use validation only and the same score floor, NMS IoU and maximum detections. [Ultralytics prediction settings](https://docs.ultralytics.com/modes/predict/)

The evaluator can save a prediction bundle with the boxes, image IDs, annotations, checkpoint and evaluation settings. To estimate uncertainty for a matched comparison, pass both bundles to the comparison script. It resamples whole images with replacement and scores both models on the same draw. The interval estimates variation across images for two fixed runs. Report variation across training seeds separately.

~~~bash
python benchmark_yolo.py --manifest data/split.json evaluate --checkpoint runs/yolo11n-seed42-e9.pt --split val --imgsz 256 --output runs/e9-raw-val.json --predictions-output runs/e9-raw-predictions.json
python benchmark_yolo.py --manifest data/split.json evaluate --checkpoint runs/yolo11n-seed42-e9.pt --split val --imgsz 256 --tta --output runs/e9-tta-val.json --predictions-output runs/e9-tta-predictions.json
python compare_detection_runs.py --baseline runs/e9-raw-predictions.json --candidate runs/e9-tta-predictions.json --repetitions 200 --seed 42 --output runs/e9-tta-paired-val.json
~~~

Across 200 paired image draws (seed 42), TTA gained 0.92 percentage points in mAP@50:95 with a 95% percentile interval of [-0.40, +2.09] points. For mAP@50 the gain was 0.65 points with an interval of [-1.34, +2.47] points. These intervals include zero, so single-pass inference remains the development default. The cached evaluator and full COCO rematching produced identical intervals on these two prediction bundles.

### Independent multi-seed reference

Jeong and Song released their benchmark manifests, training arguments and per-run evaluation records. On their NEU-DET test split, YOLO12s had the highest mean of the four evaluated architectures at 74.4% mAP@50 and 42.4% mAP@50:95. The standard deviations across three training seeds were 1.1 and 0.6 percentage points. Their YOLO11s averaged 73.0% and 41.1%. Training used up to 200 epochs at 640 px. [Benchmark archive and result tables](https://doi.org/10.5281/zenodo.23073155)

The downloaded archive matched its published MD5. Its NEU-DET manifest has 1,259 train, 269 validation and 271 test images. After removing the Roboflow filename suffix, 185 of our test filenames matched the authors' training split and 41 matched their test split. Use their results as an external reference. A direct test comparison requires a clean retraining and the same images and annotations.

At epoch 7 the texture advantage seen at epoch 3 did not persist. Raw YOLO11n reached 65.63% mAP@50 and 30.00% mAP@50:95, compared with 60.04% and 28.08% for texture. Crazing AP@50 was 27.82% raw and 26.53% texture. The raw run reached 73.18% mAP@50 and 38.81% mAP@50:95 at epoch 13. These scores use project validation and the shared COCO evaluator.

### Validation-only checkpoint fusion pilot

This pilot grouped same-class detections across checkpoints at IoU 0.55, allowing at most one box per checkpoint in each group. It averaged coordinates weighted by confidence and divided the sum of confidence scores by the number of checkpoints. All input and output bundles use the audited 270-image validation split and the shared COCO scorer. The fusion is inspired by weighted boxes fusion. [Method paper](https://arxiv.org/abs/1910.13302)

| Validation predictions | mAP@50 | mAP@50:95 |
| --- | ---: | ---: |
| Raw epoch 13, single checkpoint | 73.18% | 38.81% |
| Raw epochs 9 and 13 | 74.81% | 39.55% |
| Raw epoch 13 and texture epoch 7 | 71.76% | 37.68% |
| All three checkpoints | 73.85% | 38.83% |

Against raw epoch 13, the two-raw-checkpoint fusion gained 0.74 percentage points in mAP@50:95 (95% paired image bootstrap interval [-0.28, +1.68]) and 1.64 points in mAP@50 (interval [-0.13, +2.93]). Both intervals span zero. Keep single-checkpoint inference as the default and report the fusion as a development experiment.

### Completed raw YOLO11n development run

The 16-epoch raw run is complete. Both the best and last checkpoints scored 73.79% mAP@50 and 39.54% mAP@50:95 on the audited 270-image validation split. Compared with the 192 px incumbent, the gains were 14.03 and 16.94 percentage points, respectively. The checkpoints had different file hashes and identical aggregate and class AP on validation. The best checkpoint SHA-256 is d19cd01f7dc4c49853810629b50bc23f055ae6d394bb8c5a922db0e52c444633.

The incumbent evaluator now exports the same prediction-bundle format as the YOLO evaluator. This allows paired COCO intervals between architectures on an identical split without running inference again.

~~~bash
python -m steel_inspector evaluate --manifest data/split.json --checkpoint runs/precision192-e35.pt --split val --output runs/incumbent-matched-val.json --predictions-output runs/incumbent-matched-predictions.json --threshold 0.34 --flip-ensemble
python benchmark_yolo.py --manifest data/split.json evaluate --checkpoint runs/detect/runs/yolo11n-seed42/weights/best.pt --split val --imgsz 256 --output runs/yolo11n-seed42-final-best-val.json --predictions-output runs/yolo11n-seed42-final-best-predictions.json
python compare_detection_runs.py --baseline runs/incumbent-matched-predictions.json --candidate runs/yolo11n-seed42-final-best-predictions.json --repetitions 1000 --seed 42 --output runs/incumbent-vs-yolo11n-final-paired-val.json
~~~

The paired comparison used 1,000 draws with seed 42. The mAP@50 gain was 14.03 points (95% interval [9.61, 17.28]). The mAP@50:95 gain was 16.94 points (95% interval [14.33, 19.29]). This estimates the variation across these validation images for two fixed checkpoints.

### Fusion with the later texture checkpoint

The final raw checkpoint and the texture checkpoint after epoch 11 reached 75.93% mAP@50 and 41.12% mAP@50:95 when fused at IoU 0.55 on the audited validation split. The texture checkpoint alone scored 72.63% and 37.08%. Compared with the final raw checkpoint alone, fusion gained 2.14 and 1.57 percentage points. The 95% paired image intervals were [0.14, 3.97] and [0.14, 2.64] points across 500 draws with seed 42. This comparison is exploratory because the checkpoint combination was selected during development. Fusion requires two model passes per image.

The initial hypothesis of a general box-size bias was unsupported. In exploratory three-fold cross-validation over the 270 validation images, each fold selected no rescaling by mAP@50:95 from factors 0.90, 0.95, 1.00, 1.05 and 1.10. The held-out predictions therefore matched the original boxes.

~~~bash
python benchmark_yolo.py --manifest data/split.json --representation texture evaluate --checkpoint runs/yolo11n-texture-seed42-e11.pt --split val --imgsz 256 --output runs/yolo11n-texture-seed42-e11-coco.json --predictions-output runs/e11-texture-predictions.json
python fuse_detection_runs.py --members runs/yolo11n-seed42-final-best-predictions.json runs/e11-texture-predictions.json --iou 0.55 --output runs/fusion-final-raw-e11-texture-repro.json
python compare_detection_runs.py --baseline runs/yolo11n-seed42-final-best-predictions.json --candidate runs/fusion-final-raw-e11-texture-repro.json --repetitions 500 --seed 42 --output runs/fusion-final-raw-e11-texture-paired-repro.json
~~~

### Completed texture run and final-checkpoint fusion

The texture run completed all 16 epochs. The raw and texture checkpoints both used seed 42. All rows were evaluated on the same 270 images and 624 boxes with the shared COCO scorer.

| Input and checkpoint | mAP@50 | mAP@75 | mAP@50:95 |
| --- | ---: | ---: | ---: |
| Raw YOLO11n | 73.79% | 35.77% | 39.54% |
| Texture YOLO11n | 76.16% | 39.79% | 42.94% |
| Raw and texture fusion | 76.75% | 39.87% | 43.39% |

The texture model improved mAP@50:95 by 3.40 points over raw (95% paired image interval [1.53, 4.96]). Fusion added 0.45 points over texture alone (interval [-0.45, 1.50]). Both intervals use 1,000 paired draws with seed 42 and characterize image variation for the fitted checkpoints. Crazing AP@75 improved from 4.64% with raw input to 9.43% with texture input and remained the weakest class. Prioritize the texture model for the next single-checkpoint run. A CPU repeat with seed 43 was started to check sensitivity to training seed.

The texture checkpoint SHA-256 is d200d37dbb191bb73c35f10a1f887e68ca82b83b6a03db69336dd692644c1996. To reproduce the paired comparisons after training both development runs:

~~~bash
python benchmark_yolo.py --manifest data/split.json --representation texture evaluate --checkpoint runs/detect/runs/yolo11n-texture-seed42/weights/best.pt --split val --imgsz 256 --output runs/yolo11n-texture-final-val.json --predictions-output runs/yolo11n-texture-final-predictions.json
python fuse_detection_runs.py --members runs/yolo11n-seed42-final-best-predictions.json runs/yolo11n-texture-final-predictions.json --output runs/fusion-final-raw-final-texture.json
python compare_detection_runs.py --baseline runs/yolo11n-seed42-final-best-predictions.json --candidate runs/yolo11n-texture-final-predictions.json --repetitions 1000 --seed 42 --output runs/raw-vs-texture-final-paired-val.json
python compare_detection_runs.py --baseline runs/yolo11n-texture-final-predictions.json --candidate runs/fusion-final-raw-final-texture.json --repetitions 1000 --seed 42 --output runs/texture-vs-fusion-final-paired-val.json
~~~

These results are validation-only. The reproduced DFE-Net metrics are from its authors' split and evaluator, so they are kept in a separate table.

### Texture resolution adaptation

The final 256 px texture checkpoint scored 70.75% mAP@50 and 33.04% mAP@50:95 when run at 320 px without training at that resolution. At 256 px, the same weights scored 76.16% and 42.94% on the same 270 validation images. The 9.90-point mAP@50:95 decrease motivated a separate 320 px fine-tuning run from the texture checkpoint. The run uses 24 epochs, AdamW with a 0.0005 initial learning rate, cosine decay, and weaker scale and mosaic augmentation. It uses the original audited train and validation partitions. Evaluate its trained 320 px checkpoint against both the unchanged 320 px control and the 256 px starting result. The first warm-up epoch reached 63.41% mAP@50 and 30.14% mAP@50:95 with the shared COCO evaluator. The fine-tuning run completed all 24 epochs. Final scores appear below.

### Frozen split verification

Rebuilding the manifest from the source images and Pascal VOC annotations with seed 42 reproduced the same ordered image digests, classes, boxes, split assignments and annotation-audit counts. The existing manifest SHA-256 remained 3e278f74bf136cbcd149a0281ed8e3b661df24f8ad9c0ec3c1d1819ebd4dcead. Both YOLO dataset exports matched all 1,258 training and 270 validation records. Raw copies matched source bytes, texture images matched the deterministic transform, and normalized labels matched the manifest within nine-decimal rounding. Both exports contained train and validation images.

As an additional screen for cross-split near duplicates, we compared the signs relative to the median of 63 low-frequency DCT coefficients from 32 px grayscale images. We checked original and horizontally flipped versions and flagged Hamming distance at most four. This screen flagged no pair. Crops, stronger transformations and shared acquisition conditions remain outside this check.

### Completed resolution adaptation and second seed

On the same 270 validation images, the fine-tuned 320 px checkpoint reached 78.84% mAP@50 and 44.17% mAP@50:95. Against the 256 px starting checkpoint, the gains were 2.68 points (95% paired image interval [0.50, 4.76]) and 1.23 points (interval [-0.66, 3.02]), respectively. The 320 px run changed the learning rate schedule and augmentation as well as the resolution. Its result measures the combined recipe.

| 256 px development model | Seed | mAP@50 | mAP@50:95 |
| --- | ---: | ---: | ---: |
| Raw YOLO11n | 42 | 73.79% | 39.54% |
| Texture YOLO11n | 42 | 76.16% | 42.94% |
| Raw YOLO11n | 43 | 75.94% | 40.89% |
| Texture YOLO11n | 43 | 74.22% | 40.36% |
| Texture fusion | 42 and 43 | 77.54% | 43.89% |

The texture effect was +3.40 mAP@50:95 points for seed 42 and -0.53 points for seed 43 (paired image interval [-2.27, 1.11] for seed 43). The two arms used two CPU threads each in the seed-43 comparison. Built-in TTA with the final seed-42 texture weights scored 75.26% mAP@50 and 40.89% mAP@50:95. Its mAP@50:95 difference from single-pass inference was -2.06 points (interval [-3.24, -0.71], 500 paired draws). Single-pass inference remains the default.

### GPU training started

The CUDA environment uses CPython 3.11.16, PyTorch 2.8.0+cu126, torchvision 0.23.0+cu126 and Ultralytics 8.4.11 on an RTX 2060 SUPER. The first YOLO11s training attempt ran through validation but failed before its first checkpoint while Polars 2.1.0 checked CPU flags. Pinning Polars 1.31.0 resolved the import failure. The restarted run saved its first checkpoint and reached epoch 2. It uses the prepared texture dataset at 448 px, seed 42, batch size 8, AdamW, and up to 80 epochs. The test split has been reserved for final comparison.

### SLF-YOLO artifact audit

SLF-YOLO reports 80.0% mAP@50 and 49.0% mAP@50:95 under its own 8:1:1 split. The paper describes 300 epochs, 640 px inputs, batch size 32, SGD and FIMetal-IoU. The authors' released architecture built locally at scale s with six classes and 9,666,942 parameters, and produced finite detections in a forward pass. [Paper, Tables 1 and 4](https://www.nature.com/articles/s41598-025-94936-9.pdf) · [Pinned source](https://github.com/zacianfans/SLF-YOLO/tree/da113f16d8a46b0543f13366b0d3c1f2ae445d24)

At that revision the released training loss calls CIoU, and the repository tree contains neither evaluation weights nor a NEU-DET split manifest. A retrained result from the available code would therefore be an architecture comparison using CIoU. The reported 49.0% stays an external reference under a separate protocol. [Released box loss](https://github.com/zacianfans/SLF-YOLO/blob/da113f16d8a46b0543f13366b0d3c1f2ae445d24/SLF-YOLO/ultralytics/utils/loss.py#L102)

In the 448 px YOLO11s run, the shared COCO evaluator scored the epoch-10 checkpoint at 60.58% mAP@50 and 26.77% mAP@50:95. At epoch 20 it reached 67.14% and 34.77%, respectively, on the same validation split. These are development checkpoints from the ongoing 80-epoch run. The 640 px DFE-Net GPU preflight also completed training on a small subset, saved a checkpoint and evaluated it on the 270 validation images with the shared evaluator. Full DFE-Net training remains queued.

The archived epoch-30 checkpoint reached 75.92% mAP@50 and 39.45% mAP@50:95 on the same 270 images. Its SHA-256 is `6427c1eca0a250ea1395dd09b618bc761b5ecb4be79cdb19870026a9a250a7fb`.

### Validation operating points and CPU latency

For each model, we swept confidence in increments of 0.01 and selected the threshold with the highest recall among points with at least 85% micro precision. Matching was one-to-one within the same class and used IoU of at least 0.50. The sweep used 624 annotated boxes across 270 validation images.

| Development model | Confidence threshold | Micro precision | Micro recall | Matched boxes |
| --- | ---: | ---: | ---: | ---: |
| YOLO11n texture, 256 px | 0.46 | 85.30% | 52.08% | 325 |
| YOLO11n texture fine-tuned at 320 px | 0.34 | 85.71% | 66.35% | 414 |
| Raw and texture fusion, 256 px | 0.39 | 85.23% | 56.41% | 352 |

We timed every third validation image in manifest order, for 90 images total. Each measurement includes image decoding, texture preprocessing, resizing, the forward pass and NMS at batch size 1. We used a Windows i5-9400F, two PyTorch CPU threads and one OpenCV thread. We warmed up each model on eight images and rotated the model order across measured images while GPU training ran in a separate process.

| Development model | Median latency | 95th percentile latency |
| --- | ---: | ---: |
| YOLO11n texture, 256 px | 32.7 ms | 42.6 ms |
| YOLO11n texture fine-tuned at 320 px | 42.3 ms | 54.6 ms |
| YOLO11s texture, 448 px, epoch 20 | 131.9 ms | 147.6 ms |

### Image-level score adjustment

We pooled the frozen 320 px detector's features with global means and maxima. Using the 1,258 training images, we standardized the pooled features and fitted six binary logistic regressions with `C=0.1` to predict whether each defect class was present. We multiplied each retained box score by `(0.1 + 0.9 * class_probability) ** alpha` after NMS, then applied the shared score floor. Across validation trials with `alpha` of 0.25, 0.50, 0.75 and 1.00, mAP@50:95 ranged from 43.98% to 44.13%. The original scores reached 44.17%, so we retained them.
