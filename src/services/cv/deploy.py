"""Edge deploy helpers: read an ONNX model's I/O and generate the config files
the user writes by hand today — DeepStream nvinfer, Triton config.pbtxt — plus
an INT8 calibration-set selector. Pure templating; ONNX I/O is read via
onnxruntime (already a core dep), so no extra install for the common path.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_onnx_io(onnx_path: str) -> dict:
    """Return input/output tensor names, shapes, dtypes + file size via ORT."""
    if not os.path.exists(onnx_path):
        return {"error": f"model not found: {onnx_path}"}
    try:
        import onnxruntime as ort
    except Exception:
        return {"error": "onnxruntime not installed — `pip install onnxruntime`"}
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        return {"error": f"could not load ONNX: {e}"}

    def _io(t):
        return {"name": t.name, "shape": [d if isinstance(d, int) else str(d) for d in t.shape],
                "dtype": t.type}
    return {
        "inputs": [_io(t) for t in sess.get_inputs()],
        "outputs": [_io(t) for t in sess.get_outputs()],
        "size_mb": round(os.path.getsize(onnx_path) / 1e6, 2),
    }


def _input_hw(io: dict):
    """Best-effort (h, w, c) from the first input's NCHW/NHWC shape."""
    if not io.get("inputs"):
        return None, None, 3
    shape = io["inputs"][0]["shape"]
    dims = [d for d in shape if isinstance(d, int)]
    if len(shape) == 4:
        # NCHW (c<=4 at index1) vs NHWC
        if isinstance(shape[1], int) and shape[1] in (1, 3, 4):
            return shape[2], shape[3], shape[1]
        if isinstance(shape[3], int) and shape[3] in (1, 3, 4):
            return shape[1], shape[2], shape[3]
    return (dims[-2] if len(dims) >= 2 else None), (dims[-1] if dims else None), 3


def deepstream_config(onnx_path: str, class_names=None, *, network_mode: int = 2,
                      network_type: int = 0, num_classes: int | None = None) -> dict:
    """Generate a DeepStream nvinfer config + labels.txt for an ONNX detector.

    network_mode: 0=FP32 1=INT8 2=FP16. network_type: 0=detector 1=classifier.
    """
    io = read_onnx_io(onnx_path)
    if io.get("error"):
        return io
    h, w, c = _input_hw(io)
    names = class_names or []
    nc = num_classes if num_classes is not None else (len(names) or 1)
    model_file = os.path.basename(onnx_path)
    out_names = ";".join(o["name"] for o in io["outputs"])
    cfg = f"""[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format={0 if c == 3 else 2}
onnx-file={model_file}
model-engine-file={model_file}_b1_gpu0_{"fp16" if network_mode == 2 else "int8" if network_mode == 1 else "fp32"}.engine
labelfile-path=labels.txt
batch-size=1
network-mode={network_mode}
num-detected-classes={nc}
interval=0
gie-unique-id=1
network-type={network_type}
cluster-mode=2
maintain-aspect-ratio=1
infer-dims={c};{h};{w}
output-blob-names={out_names}
parse-bbox-func-name=NvDsInferParseYolo
#custom-lib-path=libnvdsinfer_custom_impl_Yolo.so

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
"""
    labels = "\n".join(names) if names else "\n".join(f"class{i}" for i in range(nc))
    return {"config_pbtxt": None, "deepstream_config": cfg, "labels_txt": labels,
            "input_hw": [h, w], "io": io}


def triton_config(onnx_path: str, *, model_name: str | None = None, platform: str = "onnxruntime_onnx",
                  max_batch_size: int = 1) -> dict:
    """Generate a Triton config.pbtxt from the model's actual I/O."""
    io = read_onnx_io(onnx_path)
    if io.get("error"):
        return io
    name = model_name or Path(onnx_path).stem

    def _dims(shape):
        # Drop the batch dim for Triton (max_batch_size handles it); keep -1 for dynamic.
        rest = shape[1:] if max_batch_size > 0 else shape
        return "[ " + ", ".join(str(d if isinstance(d, int) else -1) for d in rest) + " ]"

    def _dtype(t):
        return "TYPE_FP32" if "float" in t.lower() else "TYPE_INT64" if "int64" in t.lower() else "TYPE_FP32"

    ins = ",\n".join(
        f'  {{\n    name: "{i["name"]}"\n    data_type: {_dtype(i["dtype"])}\n    dims: {_dims(i["shape"])}\n  }}'
        for i in io["inputs"])
    outs = ",\n".join(
        f'  {{\n    name: "{o["name"]}"\n    data_type: {_dtype(o["dtype"])}\n    dims: {_dims(o["shape"])}\n  }}'
        for o in io["outputs"])
    cfg = (f'name: "{name}"\nplatform: "{platform}"\nmax_batch_size: {max_batch_size}\n'
           f'input [\n{ins}\n]\noutput [\n{outs}\n]\n'
           f'instance_group [ {{ kind: KIND_GPU count: 1 }} ]\n')
    return {"config_pbtxt": cfg, "model_name": name, "io": io}


def select_calibration_set(images_dir: str, *, n: int = 200, out_file: str | None = None) -> dict:
    """Pick a diverse calibration subset for INT8 quantization.

    Diversity by a cheap aHash spread (reuses imagescan) so the calib set covers
    varied scenes, not N near-identical frames — the documented cause of bad INT8
    accuracy. Writes a newline list if out_file given.
    """
    from .dataset import IMAGE_EXTS
    root = Path(images_dir)
    if not root.exists():
        return {"error": f"images dir not found: {images_dir}"}
    paths = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        return {"error": "no images found"}
    try:
        from .imagescan import _ahash, _hamming
        hashed = []
        for p in paths:
            try:
                hashed.append((p, _ahash(p)))
            except Exception:
                continue
        # Greedy farthest-point on aHash Hamming for spread.
        picked = [hashed[0]]
        while len(picked) < min(n, len(hashed)):
            best, bestd = None, -1
            for cand in hashed:
                if cand in picked:
                    continue
                d = min(_hamming(cand[1], q[1]) for q in picked)
                if d > bestd:
                    bestd, best = d, cand
            if best is None:
                break
            picked.append(best)
        chosen = [str(p) for p, _ in picked]
    except Exception:
        step = max(1, len(paths) // n)
        chosen = [str(p) for p in paths[::step][:n]]
    if out_file:
        Path(out_file).write_text("\n".join(chosen), encoding="utf-8")
    return {"count": len(chosen), "out_file": out_file, "sample": chosen[:10], "total_images": len(paths)}
