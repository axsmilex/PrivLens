# PrivLens: On-Device Privacy-Sensitive Image Detection for Visually Impaired Users

PrivLens is a machine learning pipeline that detects privacy-sensitive content — IDs, bank statements, medical records, prescriptions, and similar documents — in photos taken by blind and low-vision users, and flags or redacts it **on-device, before the photo is shared**.

A visually impaired user cannot visually confirm what is in the frame before sending a photo. PrivLens makes that judgment call for them: the moment a photo is taken, the app analyzes it locally (no cloud upload) and immediately notifies the user of any potential privacy hazard in the photo. The full loop — capture, on-device inference, notification, region blurring — has been **tested and is functional on a physical Android device**.

The experiments in this repository serve a concrete product goal: **identify the best-performing model to ship inside the PrivLens app for live-user testing** with blind and low-vision participants.

**Headline result:** a two-stage YOLOv8 → ResNet18 pipeline reaches **83.7% ± 2.8% accuracy and 0.803 ± 0.044 macro-F1** under 5-fold cross-validation on real images taken by blind/low-vision users — substantially outperforming both single-model detectors and a local multimodal LLM baseline (Gemma via Ollama, 0.41 macro-F1).

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Datasets](#datasets)
- [Approach and Iteration History](#approach-and-iteration-history)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Android Integration](#android-integration)
- [Limitations and Future Work](#limitations-and-future-work)
- [Ethics and Data Statement](#ethics-and-data-statement)

---

## Problem Statement

Given a photo taken by a visually impaired user, determine whether it contains privacy-sensitive content, and where possible localize *where* that content appears so it can be flagged or blurred before the photo is shared or processed further. The system must run on-device (privacy by design) and in near real time.

Two properties make this hard:

1. **Domain shift.** Publicly available privacy datasets are photographed by sighted users; photos from blind/low-vision users are blurrier, oddly framed, and partially occluded. Models that look strong in-distribution collapse on this target domain.
2. **Fine-grained visual similarity.** Many sensitive categories (bank statement, mortgage report, transcript, receipt) are all "a piece of paper" — a single detector struggles to both find and distinguish them.

## Datasets

| Dataset | Role | Annotations | Notes |
|---|---|---|---|
| **Connecting Pixels to Privacy and Utility** (VISPR-based, ~8,000 images) | Primary training set (Phases 2–3) | 28 privacy categories, dense COCO RLE pixel masks | Photographed by sighted users |
| **BIV-Priv-Seg** (images taken by blind/low-vision users) | Held-out cross-domain evaluation; later, training via manual labels | 16 semantic categories, image-level ground truth | **Zero label overlap** with the training set |
| **Custom-labeled query set** | Target-domain supervised training (Phases 4–6) | 1,056 images manually annotated with bounding boxes (labelImg), 16 BIV-Priv-Seg categories | Created after cross-dataset evaluation exposed the domain gap |

No raw images are redistributed in this repository — see [Ethics and Data Statement](#ethics-and-data-statement).

## Approach and Iteration History

The project deliberately documents *why* each approach failed, not just the one that worked.

### Phase 1 — MobileNetV3-Small binary classifier (baseline)

Binary sensitive/non-sensitive classifier (224×224 input, Keras), trained on a small BIV-Priv-Seg split (472 train / 102 val / 102 test), exported to TFLite (FP32, INT8).

**Result:** test accuracy 0.529, AUROC 0.546 — barely above random. High recall (0.83) came only at the cost of many false positives.
**Diagnosis:** too little data and no localization signal for a pure whole-image classifier.

### Phase 2 — YOLOv8n-seg binary segmentation

Converted the Connecting Pixels COCO RLE masks to YOLO polygon format (`pycocotools`), collapsed all 28 categories into a single binary "sensitive" class, and trained YOLOv8n-seg (AdamW, 100 epochs, 640px, batch 8; splits 3,849 / 1,154 / 2,969).

**Why binary:** the training set (28 classes) and target evaluation set (16 classes) share no label space, so binary collapse was the only setup permitting valid cross-dataset evaluation.

**Result (in-domain):** precision 0.863 / recall 0.912 at image level (verified in `results/metrics_test.json`). Strong — but in-distribution only.

### Phase 3 — Cross-dataset evaluation (the reality check)

The Phase 2 model was evaluated on the BIV-Priv-Seg query set by converting pixel predictions to an image-level decision (image flagged private if predicted-private pixels exceed a fraction threshold, swept from 0.1%).

**Result:** recall 0.78, precision 0.40, best F1 0.53 — a steep drop from in-domain performance.
**Diagnosis:** genuine domain shift plus label mismatch. This motivated manually labeling target-domain data rather than relying on transfer.

### Phase 4 — Manual labeling + supervised multi-class detection

1,056 target-domain images annotated with bounding boxes across the 16 BIV-Priv-Seg categories; YOLOv8 trained directly on this data (591 train / 148 test), with a confidence-threshold sweep (best ≈ 0.44–0.45).

**Result:** mAP@0.5 0.80, best F1 0.63 on the single split. An earlier single-split sweep produced macro-F1 0.99 — later shown by cross-validation to be wildly optimistic.
**Key lesson:** on small datasets, single train/test splits can dramatically overestimate performance; 5-fold CV is the honest estimate.

### Phase 5 — Model size sweep under 5-fold CV

YOLOv8n (~3M params) → YOLOv8s (~11M, 100 epochs) → YOLOv8m (~25M, 150 epochs, 896px) compared under scikit-learn KFold (5 folds).

**Result:** average mAP@0.5 ≈ 0.48 (YOLOv8s); larger models did not help — recall stayed the bottleneck across all sizes.
**Diagnosis:** visually similar document classes (bank statement ↔ mortgage report, receipt ↔ newspaper) confuse a single detector asked to both localize and finely classify.

### Phase 6 — Two-stage pipeline: YOLO localizes, ResNet18 classifies ★

Decouple the problem: YOLO finds *where* an object of interest is; the cropped region is passed to a dedicated ResNet18 classifier for the 16-way decision.

**Result (5-fold CV):** accuracy **83.7% ± 2.8%**, macro-F1 **0.803 ± 0.044** — a substantial improvement over any single model (full per-fold table below; `results/fivefold_results.csv`).
**Key finding:** decoupling localization from fine-grained classification significantly outperforms a monolithic detector on visually similar classes.

### Phase 7 — Local multimodal LLM baseline (Gemma via Ollama)

Zero-shot prompt-based binary classification with a fully local multimodal LLM on the same 1,056 images (~1 s/image).

**Result:** accuracy 0.697 but macro-F1 only 0.414 (TP 735, FN 4, FP 316, TN 1). Prompt refinement with an explicit 16-category list pushed PRIVATE recall to 99.9% — but NON_PRIVATE specificity stayed at 0%: the model flags essentially everything as private.
**Conclusion:** general-purpose local VLMs are not yet a reliable substitute for a task-specific supervised pipeline on this problem — an evidence-backed finding, not an assumption.

### Phase 8 — Hybrid OCR + object-detection fusion (exploratory)

Proposed fusion rule: flag an image if *either* an OCR/text-PII detector *or* the object detector fires. The two fail in complementary ways: OCR misses non-text sensitive objects (a condom box); object detection misses sensitive *text* (an email screenshot).

## Results

### Summary — all experiments

| # | Model | Train data | Evaluation | Key metric | Result |
|---|---|---|---|---|---|
| 1 | MobileNetV3-Small (binary) | BIV-Priv-Seg (472) | BIV-Priv-Seg test (102) | AUROC / Acc | 0.546 / 0.529 |
| 2 | YOLOv8n-seg (binary) | Connecting Pixels (3,849) | In-domain test | Precision / Recall | 0.86 / 0.91 |
| 3 | YOLOv8n-seg (binary) | Connecting Pixels | **Cross-dataset** (BIV-Priv-Seg) | Recall / Best F1 | 0.78 / 0.53 |
| 4 | YOLOv8 (16-class) | Manually labeled BIV-Priv-Seg | Single split | mAP@0.5 / Best F1 | 0.80 / 0.63 |
| 5 | YOLOv8s (16-class) | Manually labeled BIV-Priv-Seg | 5-fold CV | mAP@0.5 | ~0.48 |
| **6** | **YOLO + ResNet18 two-stage** | **Cropped regions, 16 classes** | **5-fold CV** | **Accuracy / Macro-F1** | **0.837 / 0.803** |
| 7 | Gemma (Ollama, zero-shot) | — | BIV-Priv-Seg (1,056) | Accuracy / Macro-F1 | 0.697 / 0.414 |

### Two-stage pipeline — 5-fold cross-validation detail

| Fold | Accuracy | Macro F1 | Weighted F1 | Best epoch |
|---|---|---|---|---|
| 1 | 0.8255 | 0.7977 | 0.8260 | 9 |
| 2 | 0.8400 | 0.8156 | 0.8357 | 11 |
| 3 | 0.8716 | 0.8583 | 0.8723 | 17 |
| 4 | 0.7919 | 0.7246 | 0.7915 | 10 |
| 5 | 0.8581 | 0.8186 | 0.8542 | 13 |
| **Mean ± std** | **0.8374 ± 0.0276** | **0.8030 ± 0.0439** | **0.8359 ± 0.0273** | — |

Raw numbers: [`results/fivefold_results.csv`](results/fivefold_results.csv), [`results/fivefold_average.csv`](results/fivefold_average.csv). Training curves, PR/F1 curves, and confusion matrices are in [`results/figures/`](results/figures/).

## Repository Structure

```
privlens/
├── src/
│   ├── data_prep/       # COCO RLE → YOLO polygon conversion, binary label collapse,
│   │                    # split generation, manual-labeling helper, dataset YAML configs
│   ├── training/        # YOLOv8 seg training (GPU/CPU), MobileNetV3 classifier v1/v2,
│   │                    # segmentation trainer with Dice loss
│   ├── evaluation/      # image-level presence metrics, cross-dataset eval,
│   │                    # prediction export, TFLite / Keras inference
│   ├── llm_baseline/    # Gemma/Ollama prompt-based baseline (see folder README)
│   └── android/         # on-device integration: photo-observer service, ONNX Runtime
│                        # classifier + segmentation wrappers, region blurring
├── results/             # metrics CSVs/JSON, 5-fold CV results, splits, figures/
├── models/              # trained weights: YOLOv8n-seg best.pt, MobileNetV3 .keras + TFLite
├── docs/                # ARCHITECTURE.md, ITERATION_LOG.md
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/axsmilex/PrivLens.git
cd PrivLens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Datasets are not distributed with this repo. Request access from the original sources (VISPR / Connecting Pixels, BIV-Priv-Seg), then:

```bash
# 1. Convert COCO RLE annotations to YOLO polygon labels
python src/data_prep/decode_coco_rle_all.py
python src/data_prep/masks_to_yolo_seg.py

# 2. Collapse 28 categories to a single binary class (cross-dataset setup)
python src/data_prep/remap_all_to_zero.py

# 3. Train
python src/training/train_yolov8_set.py            # YOLOv8n-seg, GPU

# 4. Evaluate (image-level presence metrics on the query set)
python src/evaluation/eval_yolov8_presence_metrics.py
python src/evaluation/export_predictions_and_metrics.py
```

Dataset paths are configured in `src/data_prep/*.yaml`.

## Android Integration

PrivLens ships as an Android app: when the user takes a photo, the app runs inference on-device and immediately notifies them that the photo may contain a potential privacy/safety hazard, before the photo is shared anywhere. This flow has been tested and is functional on a physical Android device. The APK is built on an OCR-capable Android app base (the `AndroidOCR` project — [upstream repo](https://github.com/jgHousemaster/AndroidOCR)), extended with the PrivLens detection service and model runtime; the OCR capability also feeds the planned OCR + object-detection fusion (Phase 8).

`src/android/` contains the PrivLens-specific on-device pieces (Java, ONNX Runtime + TFLite):

- `InsidePhotoObserverService.java` — watches for newly captured photos and triggers inference off the UI thread
- `ModelManager.java` / `OrtImageClassifier.java` / `OrtSegModel.java` — model loading and ONNX Runtime inference (classification + segmentation)
- `ImageIO.java` — bitmap → normalized NHWC float tensor conversion
- `BlurUtils.java` — blurs flagged regions before the photo is exposed to sharing flows

Quantized INT8 TFLite exports of the MobileNetV3 classifier (~1.1–1.2 MB) are in `models/mobilenet/` for mobile deployment.

## Limitations and Future Work

- **End-to-end accuracy is gated by detection recall:** if YOLO never localizes an object, the ResNet18 classifier never sees it. Reported two-stage numbers are classification-stage CV results on detected/cropped regions.
- **16-category label space:** sensitive content outside these categories is not flagged. Proposed class merges (financial_document, paper_document) may trade granularity for recall.
- **OCR fusion is unfinished:** the OCR + detection fusion rule (Phase 8) is designed and partially implemented but not fully evaluated.
- **Next steps:** select the final on-device model from these experiments and deploy it in the PrivLens app for live-user testing with blind/low-vision participants; Gemma 3N (Android-compatible) few-shot baseline; class-merged retraining; end-to-end pipeline evaluation on-device.

## Ethics and Data Statement

- **No raw dataset images are included.** VISPR/Connecting Pixels and BIV-Priv-Seg carry their own licenses; the manually labeled query set contains real photos taken by blind/low-vision users and is not redistributed.
- Only code, configuration, aggregate metrics, and non-identifying figures (metric curves, confusion matrices) are published.
- The system is designed to *protect* user privacy: all inference runs on-device; no user photo leaves the phone.

## Acknowledgments

- Datasets: *Connecting Pixels to Privacy and Utility* (VISPR-based) and *BIV-Priv-Seg*.
- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), PyTorch, TensorFlow/Keras, and [labelImg](https://github.com/HumanSignal/labelImg).
- Semester-long research assistantship project (Fall 2025 – Spring 2026), Rochester Institute of Technology.
