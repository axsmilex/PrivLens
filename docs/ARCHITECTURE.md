# PrivLens Architecture

This document describes the final two-stage detection pipeline, why it is decoupled, and how it is deployed on-device.

## System Overview

```
 ┌──────────────────────────── Android device ────────────────────────────┐
 │                                                                        │
 │  Camera capture                                                        │
 │       │                                                                │
 │       ▼                                                                │
 │  InsidePhotoObserverService          (watches for new photos,          │
 │       │                               runs off the UI thread)          │
 │       ▼                                                                │
 │  ImageIO: bitmap → 320×320 float NHWC tensor                           │
 │       │                                                                │
 │       ▼                                                                │
 │  Stage 1 — YOLOv8 (ONNX Runtime / OrtSegModel)                         │
 │       │        localizes candidate sensitive objects → bounding boxes  │
 │       ▼                                                                │
 │  Crop detected regions                                                 │
 │       │                                                                │
 │       ▼                                                                │
 │  Stage 2 — ResNet18 classifier (OrtImageClassifier)                    │
 │       │        16-way classification of each cropped region            │
 │       ▼                                                                │
 │  Decision + user notification (potential privacy hazard)               │
 │       │                                                                │
 │       ▼                                                                │
 │  BlurUtils: redact flagged regions before the photo is shared          │
 │                                                                        │
 └────────────────────────────────────────────────────────────────────────┘
```

All inference runs locally. No image ever leaves the device.

## Why Two Stages?

A single multi-class detector (YOLOv8, Phases 4–5) was asked to do two jobs at once: *find* sensitive objects and *distinguish* between 16 fine-grained categories. Under 5-fold cross-validation it plateaued around mAP@0.5 ≈ 0.48, with recall as the persistent bottleneck — scaling from YOLOv8n (3M params) to YOLOv8m (25M params) did not help.

The root cause is that many target categories are visually near-identical at detection scale: a bank statement, a mortgage report, a transcript, and a receipt are all "a rectangular piece of paper with text." The detector's classification head, operating on coarse feature maps over the whole image, cannot reliably separate them.

Decoupling the two jobs fixes this:

- **Stage 1 (YOLO) is reduced to a task it is good at** — class-agnostic localization of "an object of interest." It no longer pays a penalty for fine-grained confusion.
- **Stage 2 (ResNet18) sees clean, tightly cropped, high-resolution input** — exactly the regime image classifiers are strongest in — instead of the full noisy scene.

Under identical 5-fold cross-validation, the two-stage pipeline reached 83.7% ± 2.8% accuracy and 0.803 ± 0.044 macro-F1, versus ~0.48 mAP@0.5 for the best single-stage detector. See `../results/fivefold_results.csv`.

**Known trade-off:** end-to-end recall is gated by Stage 1. If YOLO never proposes a region, ResNet18 never sees it. The reported CV numbers measure classification on detected/cropped regions; end-to-end on-device evaluation is the next milestone.

## Model Inventory

| Component | Model | Format(s) | Location |
|---|---|---|---|
| Stage 1 localization | YOLOv8n-seg (binary, trained on Connecting Pixels) | `.pt` | `../models/yolo/` |
| Stage 2 classification | ResNet18, 16 classes (PyTorch) | trained per-fold | see `../results/fivefold_results.csv` |
| Legacy baseline / lightweight fallback | MobileNetV3-Small binary classifier | `.keras`, TFLite FP32 / INT8 (~1.1–3.7 MB) | `../models/mobilenet/` |

The INT8 TFLite exports exist specifically for the mobile deployment path; the ONNX Runtime wrappers in `../src/android/` consume ONNX exports of the same models.

## Planned Extension: OCR + Detection Fusion (Phase 8)

The object-detection pipeline and an OCR-based text/PII detector fail in complementary ways:

| Failure case | Object detection | OCR |
|---|---|---|
| Non-text sensitive object (e.g., condom box) | ✅ detects | ❌ misses (no text) |
| Sensitive text content (e.g., email screenshot, address on a letter) | ❌ often misses | ✅ detects |

Proposed fusion rule: flag the image if **either** branch fires. The Android app base already ships OCR capability (ML Kit, via the AndroidOCR project), so the fusion is an on-device rule change rather than a new model. Evaluation of the fused system is in progress.

## Design Principles

1. **Privacy by architecture** — inference on-device only; the system that protects user privacy must not itself leak photos to a server.
2. **Honest evaluation** — cross-validation over single splits; cross-dataset tests over in-distribution numbers; documented failures (see `ITERATION_LOG.md`).
3. **Deployability first** — every model choice is constrained by mobile latency and size budgets (quantized exports, ~1–13 MB weights), not leaderboard accuracy.
