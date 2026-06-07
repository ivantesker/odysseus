"""CV pipeline tools for the YOLO → edge workflow, exposed as agent tools.

Three tools, registered via the plugin registry so they behave like built-ins:
  - dataset_tools  : lint / stats / split a YOLO dataset (pure, no heavy deps)
  - eval_detector  : mAP + latency for a model over a val set (lazy ultralytics)
  - convert_model  : PT → ONNX → FP16 → RKNN + PT-vs-ONNX parity (lazy deps)

The pure logic lives in src/services/cv/*; heavy runtimes (ultralytics, onnx,
onnxruntime, rknn-toolkit2) import lazily so the app doesn't need them installed
— a missing one yields a clear "pip install …" message instead of a crash.
All three are admin-only: they read/write the filesystem and spawn heavy work.
"""

from __future__ import annotations

from src.plugin_registry import register_tool
from src.services.cv import dataset as ds
from src.services.cv.paths import CvPathError, confine


def _confine_args(args: dict, fields: list) -> dict:
    """Confine path args in-place to the tool allowlist. fields: [(name, required)].

    Returns the args; raises CvPathError on a path outside the allowed roots so
    a prompt-injected agent can't read/write arbitrary locations.
    """
    for name, required in fields:
        val = args.get(name)
        if name == "run_dirs" and isinstance(val, list):
            args[name] = [confine(d, field=name) for d in val]
        elif required or (val is not None and str(val).strip()):
            args[name] = confine(val, required=required, field=name)
    return args


# ── dataset_tools ─────────────────────────────────────────────────────────────

@register_tool(
    "dataset_tools",
    admin=True,
    description=(
        "Inspect/clean a YOLO-format dataset: lint labels (malformed/out-of-bounds/"
        "bad class id), class-balance stats, a deterministic train/val split, or a "
        "full visual EDA report (returns a URL)."
    ),
    keywords=["dataset", "yolo", "labels", "annotation", "lint", "split", "class balance", "eda", "report"],
    schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["lint", "stats", "split", "eda"]},
            "labels_dir": {"type": "string"},
            "images_dir": {"type": "string"},
            "out_dir": {"type": "string", "description": "for action=split"},
            "num_classes": {"type": "integer"},
            "class_names": {"type": "array", "items": {"type": "string"}},
            "val_frac": {"type": "number", "default": 0.2},
            "seed": {"type": "integer", "default": 0},
            "copy": {"type": "boolean", "default": True},
        },
        "required": ["action"],
    },
    fenced_help=(
        "Inspect/clean a YOLO dataset.\n"
        "```dataset_tools\n"
        '{"action": "lint", "labels_dir": "D:/data/labels", "num_classes": 3}\n'
        "```\n"
        'actions: "lint" (labels_dir[, num_classes]), "stats" (images_dir + labels_dir'
        '[, class_names]), "split" (images_dir + labels_dir + out_dir[, val_frac, seed]).'
    ),
)
def dataset_tools(args, ctx=None):
    try:
        _confine_args(args, [("labels_dir", False), ("images_dir", False), ("out_dir", False)])
    except CvPathError as e:
        return {"error": str(e), "exit_code": 1}
    action = (args.get("action") or "").strip().lower()
    if action == "lint":
        if not args.get("labels_dir"):
            return {"error": "lint needs labels_dir", "exit_code": 1}
        return {"action": "lint", **ds.lint_dataset(args["labels_dir"], args.get("num_classes"))}
    if action == "stats":
        if not (args.get("images_dir") and args.get("labels_dir")):
            return {"error": "stats needs images_dir and labels_dir", "exit_code": 1}
        return {"action": "stats", **ds.dataset_stats(
            args["images_dir"], args["labels_dir"], args.get("class_names"))}
    if action == "split":
        for k in ("images_dir", "labels_dir", "out_dir"):
            if not args.get(k):
                return {"error": f"split needs {k}", "exit_code": 1}
        return {"action": "split", **ds.apply_split(
            args["images_dir"], args["labels_dir"], args["out_dir"],
            val_frac=float(args.get("val_frac", 0.2)),
            seed=int(args.get("seed", 0)),
            copy=bool(args.get("copy", True)),
        )}
    if action == "eda":
        if not args.get("labels_dir"):
            return {"error": "eda needs labels_dir", "exit_code": 1}
        from src.services.cv import eda as cv_eda
        from src.services.cv import eda_report, report_store
        result = cv_eda.compute_eda(
            args["labels_dir"], images_dir=args.get("images_dir") or None,
            class_names=args.get("class_names"),
        )
        if result.get("error"):
            return {"error": result["error"], "exit_code": 1}
        title = (args.get("title") or "Dataset EDA").strip()
        html = eda_report.render_eda_html(result, title)
        owner = (ctx or {}).get("owner")
        rid = report_store.save_report(html, owner=owner, meta={"title": title})
        return {
            "action": "eda",
            "report_id": rid,
            "report_url": f"/api/cv/report/{rid}",
            "summary": result.get("summary", {}),
            "exit_code": 0,
        }
    return {"error": f"unknown action {action!r}; use lint|stats|split|eda", "exit_code": 1}


# ── review_labels ─────────────────────────────────────────────────────────────

@register_tool(
    "review_labels",
    admin=True,
    description=(
        "Review annotations against model predictions: rank suspect labels "
        "(missing/spurious/mismatch), build a confusion matrix, and optionally "
        "diff two models (PT vs quantized). Returns a report URL."
    ),
    keywords=["review", "qa", "mistake", "confusion", "annotation", "label error", "compare", "quantize", "regression"],
    schema={
        "type": "object",
        "properties": {
            "labels_dir": {"type": "string", "description": "ground-truth YOLO labels"},
            "preds_dir": {"type": "string", "description": "model predictions (<cls> cx cy w h conf)"},
            "images_dir": {"type": "string", "description": "optional — enables the failure gallery (images with error boxes)"},
            "preds_b_dir": {"type": "string", "description": "optional 2nd model for A/B diff"},
            "class_names": {"type": "array", "items": {"type": "string"}},
            "iou": {"type": "number", "default": 0.5},
        },
        "required": ["labels_dir", "preds_dir"],
    },
    fenced_help=(
        "Review labels vs predictions → suspect labels + confusion matrix (+ A/B diff).\n"
        "```review_labels\n"
        '{"labels_dir": "D:/ds/labels", "preds_dir": "D:/ds/preds", "class_names": ["Front","Back","Side"]}\n'
        "```"
    ),
)
def review_labels(args, ctx=None):
    try:
        _confine_args(args, [("labels_dir", False), ("preds_dir", False),
                             ("images_dir", False), ("preds_b_dir", False)])
    except CvPathError as e:
        return {"error": str(e), "exit_code": 1}
    labels_dir = (args.get("labels_dir") or "").strip()
    preds_dir = (args.get("preds_dir") or "").strip()
    if not labels_dir or not preds_dir:
        return {"error": "review_labels needs labels_dir and preds_dir", "exit_code": 1}
    from src.services.cv import eda_report, report_store
    from src.services.cv import review as rv
    iou = float(args.get("iou", 0.5))
    names = args.get("class_names")
    review = {
        "iou": iou,
        "pr": rv.pr_analysis(labels_dir, preds_dir, iou_thr=iou, class_names=names),
        "suspect_labels": rv.suspect_labels(labels_dir, preds_dir, iou_thr=iou),
        "confusion_matrix": rv.confusion_matrix(labels_dir, preds_dir, iou_thr=iou, class_names=names),
    }
    if (args.get("images_dir") or "").strip():
        from src.services.cv import imagescan
        fc = rv.failure_cases(labels_dir, preds_dir, iou_thr=iou)
        review["failure_gallery"] = {
            "rendered": imagescan.render_boxed(args["images_dir"], fc["cases"], class_names=names, max_side=640),
            "totals": fc["totals"],
        }
    if (args.get("preds_b_dir") or "").strip():
        review["compare"] = rv.compare_models(labels_dir, preds_dir, args["preds_b_dir"], iou_thr=iou, class_names=names)
    title = (args.get("title") or "Model review").strip()
    html = eda_report.render_review_html(review, title)
    owner = (ctx or {}).get("owner")
    rid = report_store.save_report(html, owner=owner, meta={"title": title})
    # Memory write-back: per-class recommended thresholds + mAP, so the agent can
    # cite them next time ("last review suggested conf 0.48 for class X").
    from src.services.cv import cv_memory
    rec = review.get("pr", {}).get("recommended_conf", {})
    if rec:
        cv_memory.remember_cv(
            f"Review of {labels_dir}: mAP={review['pr'].get('map')}, "
            f"recommended per-class conf (max-F1)={rec}, "
            f"suspects={review['suspect_labels']['counts']}",
            owner=owner, category="project")
    return {
        "action": "review",
        "report_id": rid,
        "report_url": f"/api/cv/report/{rid}",
        "suspect_total": review["suspect_labels"]["total"],
        "suspect_counts": review["suspect_labels"]["counts"],
        "recommended_conf": rec,
        "exit_code": 0,
    }


# ── deploy_config ─────────────────────────────────────────────────────────────

@register_tool(
    "deploy_config",
    admin=True,
    description=(
        "Generate edge deploy configs from an ONNX model: DeepStream nvinfer "
        "config + labels.txt and a Triton config.pbtxt, with I/O shapes read "
        "from the model. Returns a report URL."
    ),
    keywords=["deploy", "deepstream", "triton", "nvinfer", "edge", "rk3588", "config", "onnx"],
    schema={
        "type": "object",
        "properties": {
            "onnx": {"type": "string", "description": "path to the .onnx model"},
            "class_names": {"type": "array", "items": {"type": "string"}},
            "network_mode": {"type": "integer", "description": "0=FP32 1=INT8 2=FP16", "default": 2},
            "max_batch_size": {"type": "integer", "default": 1},
        },
        "required": ["onnx"],
    },
    fenced_help=(
        "Generate DeepStream + Triton configs from an ONNX model.\n"
        "```deploy_config\n"
        '{"onnx": "D:/models/best.onnx", "class_names": ["Front","Back","Side"], "network_mode": 2}\n'
        "```"
    ),
)
def deploy_config(args, ctx=None):
    try:
        _confine_args(args, [("onnx", False)])
    except CvPathError as e:
        return {"error": str(e), "exit_code": 1}
    onnx = (args.get("onnx") or "").strip()
    if not onnx:
        return {"error": "deploy_config needs onnx", "exit_code": 1}
    from src.services.cv import deploy as dp
    from src.services.cv import eda_report, report_store
    names = args.get("class_names")
    ds_cfg = dp.deepstream_config(onnx, class_names=names, network_mode=int(args.get("network_mode", 2)))
    if ds_cfg.get("error"):
        return {"error": ds_cfg["error"], "exit_code": 1}
    tr = dp.triton_config(onnx, model_name=args.get("model_name"),
                          max_batch_size=int(args.get("max_batch_size", 1)))
    merged = {**ds_cfg, "config_pbtxt": tr.get("config_pbtxt")}
    html = eda_report.render_deploy_html(merged, args.get("title") or "Edge deploy")
    owner = (ctx or {}).get("owner")
    rid = report_store.save_report(html, owner=owner, meta={"title": "Edge deploy"})
    # Memory write-back + device-constraint guard (best-effort).
    from src.services.cv import cv_memory
    hw = ds_cfg.get("input_hw")
    size = (ds_cfg.get("io") or {}).get("size_mb")
    cv_memory.remember_cv(
        f"Edge deploy: {onnx} → {hw} input, {size}MB, mode={args.get('network_mode', 2)} "
        f"(0=FP32 1=INT8 2=FP16), outputs={[o['name'] for o in ds_cfg['io']['outputs']]}",
        owner=owner, category="project")
    guard = cv_memory.device_constraints(str(args.get("target") or args.get("device") or "rk3588"), owner=owner)
    return {"action": "deploy", "report_id": rid, "report_url": f"/api/cv/report/{rid}",
            "input_hw": hw, "size_mb": size,
            "outputs": [o["name"] for o in ds_cfg["io"]["outputs"]],
            "device_notes": guard, "exit_code": 0}


# ── convert_dataset ───────────────────────────────────────────────────────────

@register_tool(
    "convert_dataset",
    admin=True,
    description=(
        "Convert COCO / Pascal-VOC / Label-Studio annotations to YOLO labels. "
        "Returns counts + discovered class names."
    ),
    keywords=["convert", "coco", "voc", "pascal", "label studio", "annotation", "yolo", "import"],
    schema={
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "COCO json / VOC xml dir / Label-Studio json"},
            "format": {"type": "string", "enum": ["coco", "voc", "labelstudio"]},
            "out_dir": {"type": "string", "description": "output YOLO labels dir"},
            "class_names": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["src", "format", "out_dir"],
    },
    fenced_help=(
        "Convert annotations to YOLO format.\n"
        "```convert_dataset\n"
        '{"src": "D:/export.json", "format": "coco", "out_dir": "D:/ds/labels"}\n'
        "```"
    ),
)
def convert_dataset(args, ctx=None):
    try:
        _confine_args(args, [("src", False), ("out_dir", False)])
    except CvPathError as e:
        return {"error": str(e), "exit_code": 1}
    src = (args.get("src") or "").strip()
    out = (args.get("out_dir") or "").strip()
    if not src or not out:
        return {"error": "convert_dataset needs src and out_dir", "exit_code": 1}
    from src.services.cv import convert_data as cd
    res = cd.convert_annotations(src, args.get("format", "coco"), out, args.get("class_names"))
    if res.get("error"):
        return {"error": res["error"], "exit_code": 1}
    return {"action": "convert_dataset", **res, "exit_code": 0}


# ── compare_runs ──────────────────────────────────────────────────────────────

@register_tool(
    "compare_runs",
    admin=True,
    description=(
        "Compare training runs (Ultralytics dirs): overlay metric curves, final "
        "metrics table, and a config diff. Returns a report URL."
    ),
    keywords=["training", "run", "compare", "wandb", "results.csv", "epoch", "experiment", "model selection"],
    schema={
        "type": "object",
        "properties": {
            "run_dirs": {"type": "array", "items": {"type": "string"},
                         "description": "training run dirs (each with results.csv)"},
            "names": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["run_dirs"],
    },
    fenced_help=(
        "Compare training runs head-to-head.\n"
        "```compare_runs\n"
        '{"run_dirs": ["D:/runs/a", "D:/runs/b"]}\n'
        "```"
    ),
)
def compare_runs(args, ctx=None):
    try:
        _confine_args(args, [("run_dirs", False)])
    except CvPathError as e:
        return {"error": str(e), "exit_code": 1}
    run_dirs = args.get("run_dirs") or []
    if not run_dirs:
        return {"error": "compare_runs needs run_dirs", "exit_code": 1}
    from src.services.cv import eda_report, report_store
    from src.services.cv import runs as cv_runs
    cmp = cv_runs.compare_runs(run_dirs, names=args.get("names"))
    if cmp.get("error"):
        return {"error": cmp["error"], "exit_code": 1}
    html = eda_report.render_runs_html(cmp, args.get("title") or "Training runs")
    rid = report_store.save_report(html, owner=(ctx or {}).get("owner"), meta={"title": "Training runs"})
    return {"action": "compare_runs", "report_id": rid, "report_url": f"/api/cv/report/{rid}",
            "runs": len(cmp["runs"]), "primary": cmp.get("primary"), "exit_code": 0}


# ── drift_check ───────────────────────────────────────────────────────────────

@register_tool(
    "drift_check",
    admin=True,
    description=(
        "Detect prediction drift on new (unlabeled) data vs a baseline: PSI on "
        "confidence, boxes/image, class frequency + detection-rate delta. URL."
    ),
    keywords=["drift", "monitor", "psi", "shift", "production", "distribution", "unlabeled"],
    schema={
        "type": "object",
        "properties": {
            "baseline_preds": {"type": "string", "description": "predictions on the eval/baseline set"},
            "new_preds": {"type": "string", "description": "predictions on new/production data"},
        },
        "required": ["baseline_preds", "new_preds"],
    },
    fenced_help=(
        "Check prediction drift vs a baseline.\n"
        "```drift_check\n"
        '{"baseline_preds": "D:/preds_baseline", "new_preds": "D:/preds_new"}\n'
        "```"
    ),
)
def drift_check(args, ctx=None):
    try:
        _confine_args(args, [("baseline_preds", False), ("new_preds", False)])
    except CvPathError as e:
        return {"error": str(e), "exit_code": 1}
    base = (args.get("baseline_preds") or "").strip()
    new = (args.get("new_preds") or "").strip()
    if not base or not new:
        return {"error": "drift_check needs baseline_preds and new_preds", "exit_code": 1}
    from src.services.cv import drift as cv_drift
    from src.services.cv import eda_report, report_store
    d = cv_drift.drift(base, new)
    if d.get("error"):
        return {"error": d["error"], "exit_code": 1}
    html = eda_report.render_drift_html(d, args.get("title") or "Drift report")
    rid = report_store.save_report(html, owner=(ctx or {}).get("owner"), meta={"title": "Drift report"})
    return {"action": "drift_check", "report_id": rid, "report_url": f"/api/cv/report/{rid}",
            "overall_verdict": d["overall_verdict"], "max_psi": d["max_psi"],
            "alert": d["alert"], "exit_code": 0}


# ── eval_detector ─────────────────────────────────────────────────────────────

@register_tool(
    "eval_detector",
    admin=True,
    description=(
        "Evaluate a detector (YOLO .pt/.onnx) on a val set: mAP@0.5 per class + "
        "inference latency (mean/p50/p90/fps). Lazy-loads ultralytics."
    ),
    keywords=["eval", "evaluate", "map", "benchmark", "detector", "yolo", "latency", "mAP"],
    schema={
        "type": "object",
        "properties": {
            "model": {"type": "string", "description": "path to .pt/.onnx"},
            "data": {"type": "string", "description": "dataset yaml or images dir"},
            "imgsz": {"type": "integer", "default": 640},
            "iou": {"type": "number", "default": 0.5},
        },
        "required": ["model", "data"],
    },
    fenced_help=(
        "Evaluate a detector's accuracy + speed.\n"
        "```eval_detector\n"
        '{"model": "best.pt", "data": "data.yaml", "imgsz": 256}\n'
        "```"
    ),
)
def eval_detector(args, ctx=None):
    try:
        _confine_args(args, [("model", False), ("data", False)])
    except CvPathError as e:
        return {"error": str(e), "exit_code": 1}
    model = (args.get("model") or "").strip()
    data = (args.get("data") or "").strip()
    if not model or not data:
        return {"error": "eval_detector needs model and data", "exit_code": 1}
    try:
        from ultralytics import YOLO
    except Exception:
        return {"error": "ultralytics not installed — `pip install ultralytics`", "exit_code": 1}
    try:
        m = YOLO(model)
        res = m.val(data=data, imgsz=int(args.get("imgsz", 640)),
                    iou=float(args.get("iou", 0.5)), verbose=False)
        box = getattr(res, "box", None)
        speed = getattr(res, "speed", {}) or {}
        out = {
            "model": model,
            "map50": round(float(getattr(box, "map50", 0.0)), 4) if box else None,
            "map50_95": round(float(getattr(box, "map", 0.0)), 4) if box else None,
            "speed_ms": {k: round(float(v), 3) for k, v in speed.items()},
        }
        if box is not None and hasattr(box, "ap_class_index"):
            try:
                names = m.names
                out["per_class_map50"] = {
                    names.get(int(c), str(c)): round(float(box.ap50[i]), 4)
                    for i, c in enumerate(box.ap_class_index)
                }
            except Exception:
                pass
        return {"action": "eval", **out, "exit_code": 0}
    except Exception as e:
        return {"error": f"eval failed: {e}", "exit_code": 1}


# ── convert_model ─────────────────────────────────────────────────────────────

@register_tool(
    "convert_model",
    admin=True,
    description=(
        "Export/convert a YOLO model: PT → ONNX → FP16 → RKNN (RK3588) with a "
        "PT-vs-ONNX numeric parity check. Lazy-loads ultralytics/onnx/rknn."
    ),
    keywords=["convert", "export", "onnx", "fp16", "rknn", "rk3588", "quantize", "tensorrt", "edge"],
    schema={
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "path to .pt"},
            "formats": {"type": "array", "items": {"type": "string", "enum": ["onnx", "fp16", "rknn"]}},
            "imgsz": {"type": "integer", "default": 640},
            "opset": {"type": "integer", "default": 12},
            "target_platform": {"type": "string", "default": "rk3588"},
            "parity": {"type": "boolean", "default": True},
        },
        "required": ["source"],
    },
    fenced_help=(
        "Convert a YOLO model for edge deploy.\n"
        "```convert_model\n"
        '{"source": "best.pt", "formats": ["onnx", "rknn"], "imgsz": 256, "target_platform": "rk3588"}\n'
        "```"
    ),
)
def convert_model(args, ctx=None):
    import os

    try:
        _confine_args(args, [("source", False), ("calib_image", False), ("sample_image", False)])
    except CvPathError as e:
        return {"error": str(e), "exit_code": 1}
    source = (args.get("source") or "").strip()
    if not source or not os.path.exists(source):
        return {"error": f"source model not found: {source}", "exit_code": 1}
    formats = [f.lower() for f in (args.get("formats") or ["onnx"])]
    imgsz = int(args.get("imgsz", 640))
    opset = int(args.get("opset", 12))
    steps: dict = {}
    outputs: dict = {}

    onnx_path = None
    if "onnx" in formats or "fp16" in formats or "rknn" in formats or args.get("parity", True):
        try:
            from ultralytics import YOLO
            onnx_path = YOLO(source).export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)
            outputs["onnx"] = str(onnx_path)
            steps["onnx"] = "ok"
        except Exception as e:
            steps["onnx"] = f"failed: {e}"
            return {"action": "convert", "steps": steps, "error": "ONNX export failed (need ultralytics)", "exit_code": 1}

    if "fp16" in formats and onnx_path:
        try:
            import onnx
            from onnxconverter_common import float16
            fp16_path = str(onnx_path).replace(".onnx", "_fp16.onnx")
            onnx.save(float16.convert_float_to_float16(onnx.load(str(onnx_path))), fp16_path)
            outputs["fp16"] = fp16_path
            steps["fp16"] = "ok"
        except Exception as e:
            steps["fp16"] = f"failed: {e} (need onnx + onnxconverter-common)"

    if "rknn" in formats and onnx_path:
        try:
            from rknn.api import RKNN
            rknn = RKNN(verbose=False)
            rknn.config(target_platform=args.get("target_platform", "rk3588"))
            if rknn.load_onnx(model=str(onnx_path)) != 0:
                raise RuntimeError("load_onnx failed")
            if rknn.build(do_quantization=False) != 0:
                raise RuntimeError("build failed")
            rknn_path = str(onnx_path).replace(".onnx", ".rknn")
            rknn.export_rknn(rknn_path)
            outputs["rknn"] = rknn_path
            steps["rknn"] = "ok"
        except Exception as e:
            steps["rknn"] = f"failed: {e} (need rknn-toolkit2 / ezrknn-toolkit2)"

    parity = None
    if args.get("parity", True) and onnx_path:
        parity = _pt_onnx_parity(source, str(onnx_path), imgsz,
                                 sample_image=(args.get("calib_image") or args.get("sample_image")))

    return {
        "action": "convert",
        "source": source,
        "outputs": outputs,
        "steps": steps,
        "parity": parity,
        "exit_code": 0,
    }


def _parity_input(imgsz: int, sample_image: str | None):
    """A realistic parity input: a decoded real image if given, else uniform
    random in [0,1]. NEVER all-zeros — that under-exercises batchnorm/activations
    and gives false-confidence parity (the input is degenerate)."""
    import numpy as np
    if sample_image:
        try:
            from PIL import Image
            im = Image.open(sample_image).convert("RGB").resize((imgsz, imgsz))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr.transpose(2, 0, 1)[None, ...]
        except Exception:
            pass
    rng = np.random.RandomState(0)
    return rng.rand(1, 3, imgsz, imgsz).astype(np.float32)


def _pt_onnx_parity(pt_path: str, onnx_path: str, imgsz: int, sample_image: str | None = None) -> dict:
    """Run a realistic input through the PT and ONNX graphs, diff outputs."""
    try:
        import numpy as np  # noqa: F401

        from src.services.cv.parity import output_diff
        x = _parity_input(imgsz, sample_image)
        try:
            import torch
            from ultralytics import YOLO
            m = YOLO(pt_path).model.float().eval()
            with torch.no_grad():
                pt_out = m(torch.from_numpy(x))
            pt_arr = (pt_out[0] if isinstance(pt_out, (list, tuple)) else pt_out).cpu().numpy()
        except Exception as e:
            return {"ok": False, "error": f"PT inference failed: {e}"}
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            onnx_arr = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        except Exception as e:
            return {"ok": False, "error": f"ONNX inference failed: {e} (need onnxruntime)"}
        if onnx_arr.shape != pt_arr.shape:
            try:
                onnx_arr = onnx_arr.reshape(pt_arr.shape)
            except Exception:
                pass
        return output_diff(pt_arr, onnx_arr, rtol=1e-2, atol=1e-2)
    except Exception as e:
        return {"ok": False, "error": str(e)}
