"""Pure CV-pipeline logic (YOLO dataset, detection metrics, model-output parity).

No heavy runtimes here (ultralytics/onnx/rknn) — those load lazily in the
plugin tools. This package is plain Python + numpy so it is unit-testable."""
