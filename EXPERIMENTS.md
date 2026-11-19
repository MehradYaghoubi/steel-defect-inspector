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
