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

The MBY article's availability statement points to the article and the corresponding author. We found no author-provided training code, weights, or exact split manifest for a direct comparison. [MBY availability statement](https://www.mdpi.com/1424-8220/25/15/4817)

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

We implemented the piecewise overlap term from equations 14–20 locally and used it in the released architecture's box regression loss. Distribution focal loss and assignment stayed as released. We expanded both box dimensions by a factor of 1.20, clipped negative overlap lengths to zero, and mapped the resulting IoU into [0, 1] using lower and upper cutoffs of 0.00 and 0.95. The ratio is a local choice and the cutoffs come from the Focaler-IoU reference. The SLF-YOLO paper describes admissible ranges, so these settings are an independent implementation choice. Identical, overlapping and disjoint boxes passed numerical checks, including finite gradients. A 128 px CPU preflight trained on 25 project training images, saved and reloaded a checkpoint, and ran the shared evaluator on all 270 validation images. The smoke checkpoint produced zero AP at the shared confidence floor. Full training with this loss is pending. [SLF-YOLO loss equations](https://www.nature.com/articles/s41598-025-94936-9) · [Focaler-IoU reference](https://github.com/malagoutou/Focaler-IoU/blob/main/focaleriou.py)

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

### Fusion across model sizes and resolutions

We combined the 320 px YOLO11n and the 448 px YOLO11s texture checkpoints using `fuse_detection_runs.py`. The sweep changes the IoU used to cluster same-class boxes across models. Each model's predictions use the shared COCO evaluation settings before fusion.

| Model on the frozen validation split | Fusion IoU | mAP@50 | mAP@50:95 |
| --- | ---: | ---: | ---: |
| YOLO11n, fine-tuned at 320 px | — | 78.84% | 44.17% |
| YOLO11s, 448 px, epoch 30 | — | 75.92% | 39.45% |
| YOLO11n and epoch-30 YOLO11s | 0.45 | 81.04% | 45.77% |
| YOLO11n and epoch-30 YOLO11s | 0.55 | 80.57% | 45.89% |
| YOLO11n and epoch-30 YOLO11s | 0.65 | 79.47% | **45.96%** |
| YOLO11s, 448 px, epoch 40 | — | 74.42% | 38.90% |
| YOLO11n and epoch-40 YOLO11s | 0.65 | 79.67% | 45.56% |
| YOLO11s, 448 px, epoch 50 | — | 76.27% | 43.03% |
| YOLO11n and epoch-50 YOLO11s | 0.65 | 79.13% | 47.27% |
| YOLO11n original and reflected views, plus epoch-50 YOLO11s | 0.65 | 79.83% | **47.78%** |
| YOLO11s, 448 px, epoch 60 | — | 77.89% | 43.43% |
| YOLO11n and epoch-60 YOLO11s | 0.65 | 79.82% | 46.96% |
| YOLO11n original and reflected views, plus epoch-60 YOLO11s | 0.65 | 79.34% | 47.15% |
| YOLO11s, 448 px, epoch 70 | — | 77.18% | 43.89% |
| YOLO11n and epoch-70 YOLO11s | 0.65 | 79.41% | 47.03% |
| YOLO11n original and reflected views, plus epoch-70 YOLO11s | 0.65 | 79.64% | 47.77% |
| YOLO11s, 448 px, best checkpoint from completed run | — | 78.49% | 44.79% |
| YOLO11n and best YOLO11s | 0.65 | 80.30% | 47.87% |
| YOLO11n original and reflected views, plus best YOLO11s | 0.65 | 79.89% | **47.96%** |
| YOLO11s, 448 px, last checkpoint from completed run | — | 76.80% | 44.25% |
| YOLO11n and last YOLO11s | 0.65 | 79.32% | 47.45% |
| YOLO11n original and reflected views, plus last YOLO11s | 0.65 | 79.85% | 47.89% |

YOLO11s completed 80 epochs. The best and last checkpoint SHA-256 hashes are `03637fba8ca00a03826b06d9504d4cecfccec31ce3968e264564ce4db1b0ec39` and `61831a89e28b0191571a5a704eb5b95925461c0c510ed50bb9188a9d5f78cec7`. The three-pass ensemble using the completed run's best checkpoint gained 0.18 strict mAP points over its epoch-50 counterpart (95% interval [-0.49, 0.95], 1,000 paired image draws). The matched DFE-Net run then started from fresh weights with 640 px inputs, batch 16, seed 42, and the 300-epoch released training recipe. It saved its first checkpoint. Its held-out evaluation is pending.

We rechecked the prepared raw and texture datasets used by these runs. Each contained the expected 1,258 training images with 2,900 boxes and 270 validation images with 624 boxes. Source pixel fingerprints matched the frozen manifest. Raw copies matched source bytes and decoded texture images matched the configured transform. Exported class IDs and normalized boxes agreed with the manifest within the nine-decimal rounding tolerance. Both dataset configurations reference only train and validation, and both export directories contain only those partitions.

At fusion IoU 0.65, the epoch-30 pair improved mAP@50:95 by 1.78 points over YOLO11n alone (95% interval [0.65, 2.80], 1,000 paired image draws). All six class APs improved at this setting. Crazing remained the weakest class, at 17.82% AP@50:95. The validation set was also used to choose the fusion IoU, so the interval describes that fixed pair of predictions on validation. The epoch-40 checkpoint SHA-256 is `e6d6696127a9839c77a63c3d293992438c72b36c2aedfd074166f9098cdeb687`.

Keeping fusion IoU at 0.65, the epoch-50 pair gained 1.32 strict mAP points over the epoch-30 pair (95% interval [0.29, 2.37], 1,000 paired image draws). The epoch-60 checkpoint improved its standalone AP but scored lower in both fusion settings. The checkpoint SHA-256 hashes are `b3d7795e234716c325c883cca1b33be2a32ead6dda03e5a4ec5061ba63c1ce97` (epoch 50) and `17e15ea865daa541b8f260f76b7fa9a582874ebedb9150569a44bfc8f0bfc6a0` (epoch 60).

We repeated the 90-image batch-1 CPU measurement with the 320 px model and the epoch-30 pair, alternating the measurement order. The 320 px model took 43.0 ms at the median and 53.1 ms at the 95th percentile. The pair took 170.8 ms and 197.4 ms, respectively, after sharing image decode and texture preprocessing across both detectors and including box fusion. The evaluator was excluded. This measurement used the same CPU and thread counts as the earlier single-model measurements.

At the validation operating point with at least 85% micro precision, the epoch-30 pair selected threshold 0.30 and reached 85.80% precision and 65.87% recall. The 320 px model reached 85.71% and 66.35% at 0.34. For crazing, the pair matched 23 of 100 annotated boxes. Of the 77 misses, 69 had at least one same-class box with IoU at least 0.50 below the confidence threshold, one had only looser same-class overlap among accepted detections, and seven had no eligible nearby detection under those rules. This diagnostic checks available candidates per annotation, so the same box can explain more than one miss. Neither operating point has been evaluated on the held-out set.

### Precision-target validation for the final checkpoints

We checked threshold selection on the 270 validation images using five image folds. For each fold, thresholds were selected on the other four folds in 0.01 increments to maximize recall. We pooled detection counts from the excluded images. The empirical rule required observed micro-precision of at least 85%. The conservative rule used the lower endpoint of a two-sided 95% Wilson interval, as in the demo benchmark. Matching required class agreement and IoU of at least 0.50.

| Configuration | Empirical precision | Empirical recall | Conservative precision | Conservative recall |
| --- | ---: | ---: | ---: | ---: |
| Texture 320 px alone | 85.39% | 66.51% | 89.33% | 61.70% |
| Three-pass fusion | 84.92% | 68.59% | 88.97% | 62.02% |
| Four passes with raw seed 42 | 85.23% | 68.43% | 88.44% | 63.78% |
| Three passes with raw seed 43 replacing the reflected view | 85.37% | 67.31% | 89.11% | 64.26% |

The conservative rule selects threshold 0.38 for the three-pass candidate when fitted on all validation images, with 88.47% precision and 63.94% recall at that threshold. Across the five excluded folds, thresholds ranged from 0.36 to 0.43. Resampling the 270 images with their fold-specific predictions gave a 95% precision interval of 85.99% to 91.79% and a recall interval of 57.45% to 66.93% using 1,000 draws. These intervals condition on the fitted rules and the current checkpoints. The detector checkpoints were previously selected on validation, and the results are operating-point measurements on validation.

### Reflected image views

Using the 320 px model, we predicted on horizontally reflected, vertically reflected, and 180-degree rotated versions of the preprocessed images. Boxes were mapped back to original image coordinates before fusion. At IoU 0.65, combining the original and horizontal views scored 79.30% mAP@50 and 45.83% mAP@50:95. The original and vertical views reached 43.43% strict mAP. Fusing all four views reached 44.75%. The horizontal pair gained 1.66 strict mAP points over the original view (95% interval [0.82, 2.65], 1,000 paired image draws). Adding the horizontal view to the epoch-50 pair gave the three-pass result above. All view choices used validation data.

The three-pass ensemble gained 0.51 strict mAP points over the two-detector epoch-50 pair (95% interval [-0.24, 1.29], 1,000 paired image draws). On the same 90 CPU timing images, the original and horizontal pair took 87.9 ms at the median and 118.2 ms at the 95th percentile. The single view took 41.6 ms and 60.4 ms in that timing run. In a separate alternating-order run with the epoch-50 checkpoint, the two-detector pair took 178.2 ms at the median and 267.3 ms at the 95th percentile. The three-pass ensemble took 220.8 ms and 294.4 ms. Each measurement shared decoding and texture preprocessing across passes and included fusion.

Of seven score-weight settings on the three-pass epoch-50 ensemble, the highest validation strict mAP used score multipliers of 1.0, 0.5 and 1.0 for the original 320 px view, reflected 320 px view, and 448 px view, respectively. The 47.98% strict mAP was 0.19 points above equal weights (95% interval [-0.23, 0.61], 1,000 paired image draws). We retain equal weights as the primary comparison because the gain is uncertain and the validation set selected the multipliers.

To reproduce the pair, score the same 320 px checkpoint twice with `benchmark_yolo.py --manifest data/split.json --representation texture evaluate --checkpoint PATH --split val --imgsz 320 --device cpu`. Supply a distinct `--output` and `--predictions-output` for each run, and include `--flip-horizontal` on the second run. Pass both prediction files to `fuse_detection_runs.py --members ORIGINAL REFLECTED --iou 0.65 --output FUSION`. We reran this workflow and reproduced the pilot predictions exactly.

### Localization and ranking diagnostics

We froze the 2,567 boxes from the equal-weight three-pass ensemble using the completed YOLO11s run. For the ranking diagnostic, each box's score was replaced with its best IoU against a same-class annotation in the same image, or zero when no such annotation existed. For the coordinate diagnostic, each box with best IoU at least 0.50 was replaced with that annotation's coordinates. This changed 782 boxes and retained their scores. Multiple predictions could map to the same annotation, and the evaluator still enforced one-to-one matching. The diagnostics use validation labels and describe the potential in the retained candidates.

| Diagnostic on frozen candidates | mAP@50 | mAP@50:95 |
| --- | ---: | ---: |
| Original | 79.89% | 47.96% |
| Annotation-based ranking | 92.39% | 60.34% |
| Annotation-based coordinates | 79.89% | 79.89% |
| Both | 92.39% | 92.39% |

We fitted simple corrections in five image-grouped validation folds. A ridge model predicted bounded coordinate offsets and size changes from class, box geometry, and agreement across views. A shallow gradient-boosted regressor estimated the fraction of COCO IoU thresholds each box would pass. We scored each fold with correctors fitted on the other four folds. The original scored 47.96% strict mAP, the coordinate correction 46.23%, the score adjustment 47.99%, and both corrections 46.31%. The score adjustment changed strict mAP by 0.03 points. We retained the uncalibrated predictions. The underlying detector checkpoints were already selected using the validation set.

We started a controlled training experiment from the existing 320 px checkpoint. Both arms use the same training images, ten CPU epochs, batch eight, AdamW at an initial learning rate of 0.00025, and seed 42. The intervention multiplies box and distribution focal loss weights by 1.5, from 7.5 and 1.5 to 11.25 and 2.25. The classification weight remains 0.5. Both arms will be scored with the same single-view and fusion protocols.

The epoch-six control snapshot scored 77.93% mAP@50 and 43.47% mAP@50:95 as a single-view detector. Replacing the original 320 px checkpoint in both views of the three-pass ensemble gave 80.95% mAP@50 and 47.14% mAP@50:95. Against the original fusion, the differences were +1.05 points at IoU 0.50 (95% interval [-0.59, 2.76]) and -0.82 points across IoU 0.50:0.95 (interval [-1.77, 0.14]), from 1,000 paired image draws. The snapshot SHA-256 is `f33ddc07bb8039163e9995056706971717535cfdf78a0905392f1d0e1cf4f1f0`. This comparison checks the interim control checkpoint against its initialization. The comparison between training arms will use the completed runs.

We also tested a second round of same-class suppression on the fused predictions. Hard suppression at IoU 0.45, 0.55, and 0.65 produced 46.78%, 47.36%, and 47.88% mAP@50:95. Gaussian score decay using `exp(-IoU**2 / sigma)` at sigma 0.25, 0.50, and 0.75 produced 47.07%, 47.72%, and 47.84%. All six settings scored below the original 47.96% and were dropped. This experiment operated on the already-fused boxes and used the same validation images and confidence floor.

### Box voting and ensemble diversity

We tried a second coordinate pass on the frozen fused predictions. Each source view contributed its nearest same-class box by IoU. If at least two boxes met the overlap cutoff, we moved the fused box halfway toward their coordinate average, weighting each vote by its confidence times squared overlap. Scores and the set of predictions stayed fixed. At cutoffs 0.45, 0.55, and 0.65, strict mAP on the completed YOLO11s ensemble was 48.02%, 47.99%, and 47.91%. At 0.45, the epoch-50 ensemble moved from 47.78% to 47.80%. The changes were small, so we retained the original fusion.

We also tested a 256 px YOLO11n member in place of or in addition to the reflected view, using the same equal-weight fusion and IoU 0.65 cutoff. All rows use the same 270 validation images.

| Ensemble | Inference passes | mAP@50 | mAP@50:95 |
| --- | ---: | ---: | ---: |
| Original three-pass ensemble | 3 | 79.89% | 47.96% |
| Add raw seed 42 | 4 | 81.08% | 48.22% |
| Add raw seed 43 | 4 | 80.87% | 48.19% |
| Add texture seed 43 | 4 | 80.84% | 47.82% |
| Replace reflected view with raw seed 43 | 3 | 81.13% | 48.09% |
| Replace reflected view with texture seed 43 | 3 | 80.10% | 47.72% |

Against the original, adding raw seed 42 gained 0.26 mAP@50:95 points (95% paired image interval [-0.33, 0.77]). Replacing the reflected view with raw seed 43 gained 0.12 points (interval [-0.75, 0.79]). Both intervals use 1,000 paired image draws and describe settings explored on the same validation data. The original remains the primary candidate because the observed strict mAP changes are small and the raw variants add a separate checkpoint.

### Grouping the three views

We tested hierarchical fusion on the same saved predictions. One order fused the two YOLO11n views first and then combined that output with YOLO11s. The other order fused the two original-orientation models first and then included the reflected view. Every fusion stage used the same IoU 0.65 clustering, score floor, and detection cap.

| Fusion configuration | Final YOLO11s checkpoint mAP@50:95 | Epoch-50 YOLO11s checkpoint mAP@50:95 |
| --- | ---: | ---: |
| Direct three-pass fusion | 47.96% | 47.78% |
| Direct two-pass fusion | 47.87% | 47.27% |
| Group the two YOLO11n views first | 48.21% | 47.67% |
| Group the two original-orientation models first | 47.43% | 47.42% |

For the final checkpoint, grouping both YOLO11n views gained 0.25 strict mAP points over direct three-pass fusion (95% interval [-0.27, 0.66], 1,000 paired image draws). Dropping the reflected pass changed strict mAP by -0.09 points (interval [-0.78, 0.50]). The grouping result varied across checkpoints, and the direct three-pass configuration remains the primary research candidate.

### Accuracy by box size and overlap threshold

The validation annotations contain 62 boxes below 32² original-image pixels, 423 from 32² up to 96², and 139 of at least 96². The same COCO evaluator reports the following size and overlap breakdown.

| Configuration | Small AP | Medium AP | Large AP | AP@75 | AP@90 | AP@95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Texture YOLO11n, 320 px | 30.72% | 40.21% | 55.03% | 41.12% | 10.87% | 4.14% |
| Texture YOLO11s, 448 px | 28.73% | 39.93% | 57.07% | 43.32% | 11.13% | 4.56% |
| Three passes | 28.87% | 43.68% | 60.11% | 48.26% | 13.04% | 4.91% |
| Four passes with raw seed 42 | 32.22% | 43.83% | 60.42% | 48.79% | 12.99% | 5.75% |

We tested two adjustments based on predicted area below 32² pixels. Halving the 448 px member's scores for those predictions changed the three-pass strict mAP from 47.96% to 48.00%. Including the raw fourth member only for small predicted boxes scored 47.42%. We retained the original three-pass setup. The size breakdown comes from one validation split and has relatively few small annotations.

### Overlapping crops

We ran the same 320 px detector on four corner crops, each spanning 70% of the source width and height. We preprocessed texture on the full image before cropping. We mapped boxes to full-image coordinates, dropped predictions within three pixels of a cut edge, and applied the shared class-wise NMS and detection cap after merging crops. The experiment used all 270 validation images.

| Configuration | Small AP | mAP@50:95 |
| --- | ---: | ---: |
| Three full-image passes | 28.87% | 47.96% |
| Four crops alone | 30.78% | 19.12% |
| Add all crop detections to the three-pass fusion | 34.74% | 46.26% |
| Add only small crop detections to the three-pass fusion | 33.60% | 47.30% |

As a separate check, we appended small crop detections whose best overlap with a same-class full-image prediction was below 0.50. We left existing coordinates and scores in place and multiplied the appended scores by 0.25 or 0.50. Both settings scored 48.00% strict mAP when rounded to two decimals. The largest small-AP change in this check was 0.27 points. We retained full-image inference because adding four crop passes gave little overall benefit.

### Reproducing the candidate training runs

Generate the texture dataset with `python benchmark_yolo.py --manifest data/split.json --representation texture prepare --destination runs/yolo-neudet-texture` in a fresh output directory. The three stages use this dataset with Ultralytics 8.4.11, seed 42, zero data-loader workers, and two PyTorch CPU threads. The two CPU runs used Python 3.14.4 and PyTorch 2.14.0+cpu. The CUDA run used Python 3.11.16, PyTorch 2.8.0+cu126, and an RTX 2060 SUPER. Run the two CPU stages in sequence and the CUDA stage in its own environment.

| Stage | Initialization | Size, epochs, batch | Settings |
| --- | --- | --- | --- |
| YOLO11n texture | COCO `yolo11n.pt` | 256, 16, 16 | `optimizer=auto`, selected AdamW at 0.001 with momentum 0.9, `amp=False` |
| YOLO11n resolution adaptation | `best.pt` from the 256 px stage | 320, 24, 16 | AdamW, `lr0=0.0005`, `lrf=0.10`, `cos_lr=True`, `hsv_h=0`, `hsv_s=0`, `hsv_v=0.20`, `scale=0.25`, `mosaic=0.25`, `close_mosaic=6`, `amp=False` |
| YOLO11s texture | COCO `yolo11s.pt` | 448, 80, 8 | AdamW, `lr0=0.001`, `lrf=0.05`, `cos_lr=True`, `patience=25`, `hsv_h=0`, `hsv_s=0`, `hsv_v=0.20`, `scale=0.25`, `mosaic=0.50`, `close_mosaic=10`, `amp=True` |

Use `YOLO(start_checkpoint).train(data=dataset_yaml, device=device, imgsz=size, epochs=epochs, batch=batch, workers=0, seed=42, deterministic=True, plots=False, **stage_settings)` with the settings above and the Ultralytics 8.4.11 defaults for unspecified arguments. Use the trained `best.pt` from each branch. The three-pass validation ensemble uses the 320 px checkpoint in both orientations and the 448 px checkpoint in the original orientation, with the same scoring commands and IoU 0.65 fusion described above.

| Artifact | SHA-256 |
| --- | --- |
| COCO `yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| COCO `yolo11s.pt` | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| Trained YOLO11n 256 px | `d200d37dbb191bb73c35f10a1f887e68ca82b83b6a03db69336dd692644c1996` |
| Trained YOLO11n 320 px | `a67d5a9d72aee7609cbccf8a9222486f47ade9a7eab19a059ec6d718ac60b64c` |
| Trained YOLO11s 448 px | `03637fba8ca00a03826b06d9504d4cecfccec31ce3968e264564ce4db1b0ec39` |

The archived 448 px checkpoint produced 2,042 validation detections in the same order on CPU and CUDA. The maximum matched coordinate and score differences were 0.00077 px and 0.00000346, respectively. Both COCO AP scores were identical, including after replacing the CPU YOLO11s predictions with CUDA predictions in the three-pass ensemble. This check covers those devices, the frozen validation split, and this checkpoint.

### Held-out comparison procedure

Select the candidate pipeline and the matched reference checkpoint using validation, and record their checkpoint hashes, inference settings, and selection rule before scoring the held-out split. The benchmark compares complete detection systems. Our candidate uses COCO-pretrained texture models and multiple inference passes, while DFE-Net uses its released fresh-initialization recipe and one pass on raw images. Report these differences alongside accuracy and latency.

Use the project manifest and `benchmark_yolo.py` for each detector or view, then `fuse_detection_runs.py` for the selected ensemble. Run `compare_detection_runs.py` on the resulting reference and candidate prediction bundles to verify identical ordered images, annotations and evaluation settings, and to compute the candidate-minus-reference AP difference. Report the paired image bootstrap interval and the two absolute scores. The interval quantifies variation across sampled test images for the frozen pair of runs. It does not include variation from training seeds.

We rehearsed the shared scoring path with the archived epoch-11 checkpoint from the live 640 px DFE-Net run. On 270 validation images with 624 boxes, it scored 18.76% mAP@50 and 5.98% mAP@50:95. The checkpoint SHA-256 is `350abbb94800694de928df1fe9e8c75db43f05ea160e657e4887b54f1b518bfc`. The comparison tool verified the snapshot and ensemble bundles on the same ordered validation inputs and completed 1,000 paired draws. This early snapshot exercises the reference checkpoint and evaluation path while training continues for 300 epochs.

At epoch 30, DFE-Net reached 47.84% mAP@50 and 20.99% mAP@50:95 on the same images and shared evaluator. The archived checkpoint SHA-256 is `f39ee54dde910b3c2340197d3f8a3a42a204d2ff0e9e80d82c6c7961d7843205`. The training logger showed 20.06% strict mAP at this epoch with its native evaluator. The authors' seed-42 log showed 20.76% at the same epoch on their validation split. These checkpoints describe early convergence under their respective protocols. [Released seed-42 training log](https://github.com/hoid1/DFE-Net/blob/936067614ef39325c91b85f77995f648305e08e9/runs/detect/runs/ours-neudet-seed42/results.csv)

The interim training-selected checkpoint from epoch 46 reached 61.34% mAP@50 and 29.41% mAP@50:95 with the shared evaluator, up 13.50 and 8.41 points from epoch 30 on the same images. The training logger recorded 29.98% strict mAP for epoch 46 using its native evaluator. The archived checkpoint SHA-256 is `ec5304c0445433ecdc72506ac3f487a27d8cfc450e3417d5ed80215b005650f3`. This is a development snapshot while the reference run continues.

For the matched DFE-Net run, prepare the raw dataset with `python benchmark_yolo.py --manifest data/split.json prepare --destination runs/yolo-neudet`. Use the released `yolo11n-dfe-net.yaml` architecture and `train_args_dfe-net.yaml` from the pinned checkout. In addition to dataset and output paths, the archived effective settings differ from that recipe only in seed (42 instead of 0), data-loader workers (0 instead of 4), disabled plots, and `pretrained=False`. The authors' own training script also sets `pretrained=False` for its fresh-initialization experiments. [Pinned DFE-Net training script](https://github.com/hoid1/DFE-Net/blob/936067614ef39325c91b85f77995f648305e08e9/tools/train.py)

The replay command below prepares the raw dataset in a new run directory and trains the pinned model in the DFE-Net environment. It exports only training and validation, records effective arguments and manifest and architecture hashes in `recipe.json`, and writes the trained weights under `dfenet/weights/`.

```bash
git clone https://github.com/hoid1/DFE-Net.git runs/dfenet-reference
git -C runs/dfenet-reference checkout --detach 936067614ef39325c91b85f77995f648305e08e9
python train_dfenet_baseline.py --reference runs/dfenet-reference --manifest data/split.json --output runs/dfenet-replay --device 0
```

Install the released environment's dependencies before training and expose the checkout on `PYTHONPATH` when scoring a DFE-Net checkpoint with `benchmark_yolo.py`. The local Windows checkout uses the optional-import changes described in the external reproduction section. The replay command restores names for omitted legacy Mamba and DyHead layers required by the upstream parser. Construction raises an error if a selected layer requires an omitted dependency. [Released environment instructions](https://github.com/hoid1/DFE-Net/tree/936067614ef39325c91b85f77995f648305e08e9#1-environment)

The replay command completed a CPU smoke run with `--device cpu --epochs 1 --imgsz 128 --batch 4 --fraction 0.02`. It exported 1,258 training and 270 validation images, trained on the selected 25-image subset, and saved a checkpoint that the shared evaluator scored on all 270 validation images. The smoke checkpoint's AP was zero. The 300-epoch matched result comes from the separate live GPU run.

UC-DetNet reports 82.8% mAP@50 and 50.1% mAP@50:95. We found no author-provided evaluation weights or exact split manifest for a direct comparison. The continuity regularization study reports a 5.03-point strict mAP improvement from Multi-Continuity on a YOLOv12 experiment at 1280 px with a 6:2:2 split. The authors publish training code, which makes the loss a candidate for a matched ablation. [UC-DetNet, Table II](https://cmsweb.com.sg/rps2prod/ccdc2026/epro/pdf/SatAIS-44.pdf) · [Continuity regularization, Section 4.5](https://arxiv.org/html/2608.17362v1) · [Author implementation](https://github.com/minjong92/Continuity-Driven-Representation-Learning-for-Industrial-Defect-Detection)
