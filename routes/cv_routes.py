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

router = APIRouter()


def setup_cv_routes() -> APIRouter:
    @router.post("/api/cv/dataset/lint")
    def cv_dataset_lint(request: Request, body: dict = Body(...)):
        require_admin(request)
        labels_dir = (body.get("labels_dir") or "").strip()
        if not labels_dir:
            raise HTTPException(400, "labels_dir is required")
        return ds.lint_dataset(labels_dir, body.get("num_classes"))

    @router.post("/api/cv/dataset/stats")
    def cv_dataset_stats(request: Request, body: dict = Body(...)):
        require_admin(request)
        images_dir = (body.get("images_dir") or "").strip()
        labels_dir = (body.get("labels_dir") or "").strip()
        if not images_dir or not labels_dir:
            raise HTTPException(400, "images_dir and labels_dir are required")
        return ds.dataset_stats(images_dir, labels_dir, body.get("class_names"))

    @router.post("/api/cv/dataset/split")
    def cv_dataset_split(request: Request, body: dict = Body(...)):
        require_admin(request)
        for k in ("images_dir", "labels_dir", "out_dir"):
            if not (body.get(k) or "").strip():
                raise HTTPException(400, f"{k} is required")
        return ds.apply_split(
            body["images_dir"], body["labels_dir"], body["out_dir"],
            val_frac=float(body.get("val_frac", 0.2)),
            seed=int(body.get("seed", 0)),
            copy=bool(body.get("copy", True)),
        )

    @router.post("/api/cv/inspect")
    def cv_inspect(request: Request, body: dict = Body(...)):
        """Auto-discover a dataset from its root (data.yaml → dirs + classes)."""
        require_admin(request)
        root = (body.get("root") or "").strip()
        if not root:
            raise HTTPException(400, "root is required")
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
        labels_dir = (body.get("labels_dir") or "").strip()
        if not labels_dir:
            raise HTTPException(400, "labels_dir is required")
        images_dir = (body.get("images_dir") or "").strip() or None
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
        labels_dir = (body.get("labels_dir") or "").strip()
        preds_dir = (body.get("preds_dir") or "").strip()
        if not labels_dir or not preds_dir:
            raise HTTPException(400, "labels_dir and preds_dir are required")
        iou = float(body.get("iou", 0.5))
        names = body.get("class_names")
        review = {
            "pr": cv_review_svc.pr_analysis(labels_dir, preds_dir, iou_thr=iou, class_names=names),
            "suspect_labels": cv_review_svc.suspect_labels(labels_dir, preds_dir, iou_thr=iou),
            "confusion_matrix": cv_review_svc.confusion_matrix(labels_dir, preds_dir, iou_thr=iou, class_names=names),
        }
        images_dir = (body.get("images_dir") or "").strip()
        if images_dir:
            from src.services.cv import imagescan
            fc = cv_review_svc.failure_cases(labels_dir, preds_dir, iou_thr=iou)
            review["failure_gallery"] = {
                "rendered": imagescan.render_boxed(images_dir, fc["cases"], class_names=names),
                "totals": fc["totals"],
            }
        preds_b = (body.get("preds_b_dir") or "").strip()
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
        root = (body.get("root") or "").strip()
        if not root:
            raise HTTPException(400, "root is required")
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
        onnx = (body.get("onnx") or "").strip()
        if not onnx:
            raise HTTPException(400, "onnx path is required")
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
        images_dir = (body.get("images_dir") or "").strip()
        if not images_dir:
            raise HTTPException(400, "images_dir is required")
        res = dp.select_calibration_set(images_dir, n=int(body.get("n", 200)),
                                        out_file=(body.get("out_file") or "").strip() or None)
        if res.get("error"):
            raise HTTPException(400, res["error"])
        return res

    @router.post("/api/cv/convert-data")
    def cv_convert_data(request: Request, body: dict = Body(...)):
        """Convert COCO/VOC/Label-Studio annotations to YOLO."""
        require_admin(request)
        from src.services.cv import convert_data as cd
        src = (body.get("src") or "").strip()
        out = (body.get("out_dir") or "").strip()
        if not src or not out:
            raise HTTPException(400, "src and out_dir are required")
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
        labels_dir = (body.get("labels_dir") or "").strip()
        if not labels_dir:
            raise HTTPException(400, "labels_dir is required")
        res = cd.stratified_split(labels_dir, val_frac=float(body.get("val_frac", 0.2)),
                                  seed=int(body.get("seed", 0)),
                                  source_regex=(body.get("source_regex") or "").strip() or None,
                                  class_names=body.get("class_names"))
        if res.get("error"):
            raise HTTPException(400, res["error"])
        return res

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
