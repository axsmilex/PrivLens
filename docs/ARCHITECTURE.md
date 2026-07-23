# Architecture

> Placeholder — the corresponding design write-up was not found in `C:\Alex\RIT\GA`.

Intended contents (per project README):

- Two-stage pipeline design: YOLOv8 localization → cropped-region classification, and why the decoupled design outperformed a single multi-class detector.
- On-device deployment path: TFLite / ONNX Runtime models invoked from the Android service (`src/android/`), with `InsidePhotoObserverService` watching for new photos and `BlurUtils` redacting flagged regions.
- Hybrid OCR + object-detection fusion rule (Phase 8, exploratory).
