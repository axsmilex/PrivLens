# Iteration Log

> Placeholder — condense your weekly notes here.

Phase summary (from README):

1. MobileNetV3-Small binary baseline — AUROC ≈ 0.48–0.55.
2. YOLOv8n-seg, 28→1 binary label collapse — P 0.86 / R 0.91 in-distribution.
3. Cross-dataset eval (Connecting Pixels → BIV-Priv-Seg) — acc 0.53, exposing the generalization gap.
4. Manual labeling of 1,056 target-domain images; supervised multi-class YOLOv8.
5. Model size sweep (n/s/m) under 5-fold CV — recall bottleneck identified.
6. Two-stage YOLO → ResNet18 pipeline — 83.7% acc / 80.3% macro-F1 (5-fold CV).
7. Local multimodal LLM comparison (Gemma via Ollama) — 69.7% acc / 0.41 macro-F1.
8. Hybrid OCR + detection fusion (in progress).

Note: scripts/CSVs for phases 4–7 (5-fold CV, ResNet18, threshold sweep, Gemma image baseline) were not present in the GA folder when this repo was assembled — pull them from wherever those experiments ran.
