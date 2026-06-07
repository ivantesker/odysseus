"""CV pipeline routes — the HTTP surface behind the CV tool panel.

Thin admin-only wrappers over the same logic the agent tools use
(plugins/cv_pipeline + src/services/cv), so the UI panel and the chat agent
share one implementation. Dataset actions are pure-Python; eval/convert lazy-
import the heavy CV stack and return a clear message when it's absent.
"""

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse

from core.middleware import require_admin
from src.auth_helpers import get_current_user
from src.services.cv import dataset as ds
from src.services.cv import eda as cv_eda
from src.services.cv import eda_report, report_store
from src.services.cv.paths import CvPathError, confine

router = APIRouter()


def _p(raw, field: str, *, required: bool = True):
    """Confine a user path to the allowlist or raise HTTP 400."""
    try:
        return confine(raw, required=required, field=field)
    except CvPathError as e:
        raise HTTPException(400, str(e)) from e


def setup_cv_routes() -> APIRouter:
    @router.post("/api/cv/dataset/lint")
    def cv_dataset_lint(request: Request, body: dict = Body(...)):
        require_admin(request)
        labels_dir = _p(body.get("labels_dir"), "labels_dir")
        return ds.lint_dataset(labels_dir, body.get("num_classes"))

    @router.post("/api/cv/dataset/stats")
    def cv_dataset_stats(request: Request, body: dict = Body(...)):
        require_admin(request)
        images_dir = _p(body.get("images_dir"), "images_dir")
        labels_dir = _p(body.get("labels_dir"), "labels_dir")
        return ds.dataset_stats(images_dir, labels_dir, body.get("class_names"))

    @router.post("/api/cv/dataset/split")
    def cv_dataset_split(request: Request, body: dict = Body(...)):
        require_admin(request)
        images_dir = _p(body.get("images_dir"), "images_dir")
        labels_dir = _p(body.get("labels_dir"), "labels_dir")
        out_dir = _p(body.get("out_dir"), "out_dir")
        return ds.apply_split(
            images_dir, labels_dir, out_dir,
            val_frac=float(body.get("val_frac", 0.2)),
            seed=int(body.get("seed", 0)),
            copy=bool(body.get("copy", True)),
        )

    @router.post("/api/cv/inspect")
    def cv_inspect(request: Request, body: dict = Body(...)):
        """Auto-discover a dataset from its root (data.yaml → dirs + classes)."""
        require_admin(request)
        root = _p(body.get("root"), "root")
        result = ds.inspect_dataset(root)
        if result.get("error"):
            raise HTTPException(400, result["error"])
        return result

    @router.get("/api/cv/reports")
    def cv_reports(request: Request):
        """List previously generated CV reports (owner-scoped)."""
        require_admin(request)
        return {"reports": report_store.list_reports(owner=get_current_user(request))}

    @router.post("/api/cv/eda")
    def cv_eda_report(request: Request, body: dict = Body(...)):
        """Compute dataset EDA, render an HTML report, store it, return its URL."""
        require_admin(request)
        labels_dir = _p(body.get("labels_dir"), "labels_dir")
        images_dir = _p(body.get("images_dir"), "images_dir", required=False)
        result = cv_eda.compute_eda(
            labels_dir,
            images_dir=images_dir,
            class_names=body.get("class_names"),
            num_classes=body.get("num_classes"),
            grid=int(body.get("grid", 12)),
        )
        if result.get("error"):
            raise HTTPException(400, result["error"])
        # Annotation samples (real images with boxes drawn) — on by default when
        # an images dir is given, since "show me the labels" is the first thing
        # anyone wants. Cheap (decodes only ~9 images).
        if images_dir and body.get("samples", True):
            from src.services.cv import imagescan
            result["samples"] = imagescan.sample_annotations(
                images_dir, labels_dir, n=int(body.get("n_samples", 12)))
        # Optional (slower) image-pixel scan: brightness/blur/entropy/dups.
        if body.get("image_scan") and images_dir:
            from src.services.cv import imagescan
            result["image_scan"] = imagescan.scan_images(
                images_dir, sample=int(body.get("scan_sample", 400)))
        title = (body.get("title") or "Dataset EDA").strip()
        html = eda_report.render_eda_html(result, title)
        owner = get_current_user(request)
        report_id = report_store.save_report(html, owner=owner, meta={"title": title})
        return {
            "report_id": report_id,
            "report_url": f"/api/cv/report/{report_id}",
            "summary": result.get("summary", {}),
        }

    @router.post("/api/cv/review")
    def cv_review(request: Request, body: dict = Body(...)):
        """Build a model-review report from GT + prediction label dirs.

        Body: {labels_dir, preds_dir, preds_b_dir?, class_names?, iou?}. Produces
        suspect-label triage, a confusion matrix, and (with preds_b_dir) a
        two-model accuracy diff. Returns the stored report URL.
        """
        require_admin(request)
        from src.services.cv import review as cv_review_svc
        labels_dir = _p(body.get("labels_dir"), "labels_dir")
        preds_dir = _p(body.get("preds_dir"), "preds_dir")
        iou = float(body.get("iou", 0.5))
        names = body.get("class_names")
        review = {
            "iou": iou,
            "pr": cv_review_svc.pr_analysis(labels_dir, preds_dir, iou_thr=iou, class_names=names),
            "suspect_labels": cv_review_svc.suspect_labels(labels_dir, preds_dir, iou_thr=iou),
            "confusion_matrix": cv_review_svc.confusion_matrix(labels_dir, preds_dir, iou_thr=iou, class_names=names),
        }
        images_dir = _p(body.get("images_dir"), "images_dir", required=False)
        if images_dir:
            from src.services.cv import imagescan
            fc = cv_review_svc.failure_cases(labels_dir, preds_dir, iou_thr=iou)
            review["failure_gallery"] = {
                "rendered": imagescan.render_boxed(images_dir, fc["cases"], class_names=names, max_side=640),
                "totals": fc["totals"],
            }
        preds_b = _p(body.get("preds_b_dir"), "preds_b_dir", required=False)
        if preds_b:
            review["compare"] = cv_review_svc.compare_models(labels_dir, preds_dir, preds_b, iou_thr=iou, class_names=names)
        title = (body.get("title") or "Model review").strip()
        html = eda_report.render_review_html(review, title)
        owner = get_current_user(request)
        report_id = report_store.save_report(html, owner=owner, meta={"title": title})
        return {
            "report_id": report_id,
            "report_url": f"/api/cv/report/{report_id}",
            "suspect_total": review["suspect_labels"]["total"],
            "compare": review.get("compare", {}).get("map_delta") if "compare" in review else None,
        }

    @router.post("/api/cv/eval-aggregate")
    def cv_eval_aggregate(request: Request, body: dict = Body(...)):
        """Aggregate every eval CSV under a folder into one comparison report."""
        require_admin(request)
        from src.services.cv import evalcsv
        root = _p(body.get("root"), "root")
        agg = evalcsv.aggregate_csvs(root)
        if agg.get("error") and not agg.get("runs"):
            raise HTTPException(400, agg["error"])
        title = (body.get("title") or "Eval aggregate").strip()
        html = eda_report.render_eval_aggregate_html(agg, title)
        owner = get_current_user(request)
        report_id = report_store.save_report(html, owner=owner, meta={"title": title})
        return {"report_id": report_id, "report_url": f"/api/cv/report/{report_id}",
                "runs": len(agg.get("runs", [])), "primary": agg.get("primary")}

    @router.post("/api/cv/deploy")
    def cv_deploy(request: Request, body: dict = Body(...)):
        """Generate edge deploy configs (DeepStream + Triton) from an ONNX model."""
        require_admin(request)
        from src.services.cv import deploy as dp
        onnx = _p(body.get("onnx"), "onnx")
        names = body.get("class_names")
        ds = dp.deepstream_config(onnx, class_names=names,
                                  network_mode=int(body.get("network_mode", 2)))
        if ds.get("error"):
            raise HTTPException(400, ds["error"])
        tr = dp.triton_config(onnx, model_name=body.get("model_name"),
                              max_batch_size=int(body.get("max_batch_size", 1)))
        merged = {**ds, "config_pbtxt": tr.get("config_pbtxt"), "io": ds.get("io")}
        title = (body.get("title") or "Edge deploy").strip()
        html = eda_report.render_deploy_html(merged, title)
        rid = report_store.save_report(html, owner=get_current_user(request), meta={"title": title})
        return {"report_id": rid, "report_url": f"/api/cv/report/{rid}",
                "input_hw": ds.get("input_hw"), "outputs": [o["name"] for o in ds["io"]["outputs"]]}

    @router.post("/api/cv/calibration")
    def cv_calibration(request: Request, body: dict = Body(...)):
        """Select a diverse INT8 calibration subset from an images dir."""
        require_admin(request)
        from src.services.cv import deploy as dp
        images_dir = _p(body.get("images_dir"), "images_dir")
        res = dp.select_calibration_set(images_dir, n=int(body.get("n", 200)),
                                        out_file=_p(body.get("out_file"), "out_file", required=False))
        if res.get("error"):
            raise HTTPException(400, res["error"])
        return res

    @router.post("/api/cv/convert-data")
    def cv_convert_data(request: Request, body: dict = Body(...)):
        """Convert COCO/VOC/Label-Studio annotations to YOLO."""
        require_admin(request)
        from src.services.cv import convert_data as cd
        src = _p(body.get("src"), "src")
        out = _p(body.get("out_dir"), "out_dir")
        res = cd.convert_annotations(src, body.get("format", "coco"), out, body.get("class_names"))
        if res.get("error"):
            raise HTTPException(400, res["error"])
        return res

    @router.post("/api/cv/class-map")
    def cv_class_map(request: Request, body: dict = Body(...)):
        """Build a unified class map across sources [{name, class_names}]."""
        require_admin(request)
        from src.services.cv import convert_data as cd
        sources = body.get("sources") or []
        if not sources:
            raise HTTPException(400, "sources is required")
        return cd.auto_class_map(sources)

    @router.post("/api/cv/split-stratified")
    def cv_split_stratified(request: Request, body: dict = Body(...)):
        """Leak-free stratified train/val split (keeps each source together)."""
        require_admin(request)
        from src.services.cv import convert_data as cd
        labels_dir = _p(body.get("labels_dir"), "labels_dir")
        res = cd.stratified_split(labels_dir, val_frac=float(body.get("val_frac", 0.2)),
                                  seed=int(body.get("seed", 0)),
                                  source_regex=(body.get("source_regex") or "").strip() or None,
                                  class_names=body.get("class_names"))
        if res.get("error"):
            raise HTTPException(400, res["error"])
        return res

    @router.post("/api/cv/runs/register")
    def cv_run_register(request: Request, body: dict = Body(...)):
        """Register a training run (Ultralytics dir) into the local registry."""
        require_admin(request)
        from src.services.cv import runs as cv_runs
        run_dir = _p(body.get("run_dir"), "run_dir")
        rec = cv_runs.register_run(run_dir, name=body.get("name"), owner=get_current_user(request))
        if rec.get("error"):
            raise HTTPException(400, rec["error"])
        return {"id": rec["id"], "name": rec["name"], "best": rec.get("best"), "epochs": rec.get("epochs")}

    @router.get("/api/cv/runs")
    def cv_runs_list(request: Request):
        require_admin(request)
        from src.services.cv import runs as cv_runs
        return {"runs": cv_runs.list_runs(owner=get_current_user(request))}

    @router.post("/api/cv/runs/compare")
    def cv_runs_compare(request: Request, body: dict = Body(...)):
        """Compare training runs (overlay curves + final table + config diff)."""
        require_admin(request)
        from src.services.cv import runs as cv_runs
        raw_dirs = body.get("run_dirs") or []
        if not raw_dirs:
            raise HTTPException(400, "run_dirs is required")
        run_dirs = [_p(d, "run_dir") for d in raw_dirs]
        cmp = cv_runs.compare_runs(run_dirs, names=body.get("names"))
        if cmp.get("error"):
            raise HTTPException(400, cmp["error"])
        title = (body.get("title") or "Training runs").strip()
        html = eda_report.render_runs_html(cmp, title)
        rid = report_store.save_report(html, owner=get_current_user(request), meta={"title": title})
        return {"report_id": rid, "report_url": f"/api/cv/report/{rid}",
                "runs": len(cmp["runs"]), "primary": cmp.get("primary")}

    @router.post("/api/cv/drift")
    def cv_drift(request: Request, body: dict = Body(...)):
        """PSI drift of model predictions on new data vs a baseline."""
        require_admin(request)
        from src.services.cv import drift as cv_drift
        base = _p(body.get("baseline_preds"), "baseline_preds")
        new = _p(body.get("new_preds"), "new_preds")
        d = cv_drift.drift(base, new)
        if d.get("error"):
            raise HTTPException(400, d["error"])
        title = (body.get("title") or "Drift report").strip()
        html = eda_report.render_drift_html(d, title)
        rid = report_store.save_report(html, owner=get_current_user(request), meta={"title": title})
        return {"report_id": rid, "report_url": f"/api/cv/report/{rid}",
                "overall_verdict": d["overall_verdict"], "max_psi": d["max_psi"], "alert": d["alert"]}

    @router.get("/api/cv/report/{report_id}")
    def cv_get_report(request: Request, report_id: str):
        """Serve a stored CV report as HTML (owner-scoped)."""
        owner = get_current_user(request)
        html = report_store.load_report(report_id, owner=owner)
        if html is None:
            raise HTTPException(404, "Report not found")
        return HTMLResponse(content=html)

    @router.post("/api/cv/eval")
    def cv_eval(request: Request, body: dict = Body(...)):
        require_admin(request)
        from plugins.cv_pipeline import eval_detector
        return eval_detector(body)

    @router.post("/api/cv/convert")
    def cv_convert(request: Request, body: dict = Body(...)):
        require_admin(request)
        from plugins.cv_pipeline import convert_model
        return convert_model(body)

    return router
