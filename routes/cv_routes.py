"""CV pipeline routes — the HTTP surface behind the CV tool panel.

Thin admin-only wrappers over the same logic the agent tools use
(plugins/cv_pipeline + src/services/cv), so the UI panel and the chat agent
share one implementation. Dataset actions are pure-Python; eval/convert lazy-
import the heavy CV stack and return a clear message when it's absent.
"""

from fastapi import APIRouter, Body, HTTPException, Request

from core.middleware import require_admin
from src.services.cv import dataset as ds

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
