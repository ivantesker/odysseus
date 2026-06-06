"""Numerical parity between two model outputs (e.g. PT vs ONNX vs RKNN).

After exporting/quantizing a model you want to confirm the converted graph
still produces the same numbers. These helpers compare two output tensors and
report how far apart they are, plus a pass/fail against tolerances. Pure numpy.
"""

from __future__ import annotations

import numpy as np


def output_diff(a, b, *, rtol: float = 1e-3, atol: float = 1e-3) -> dict:
    """Compare two arrays of equal shape; return diff metrics + pass flag.

    Reports max/mean absolute error, max relative error, cosine similarity of
    the flattened outputs, and whether they're allclose within (rtol, atol).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"ok": False, "error": f"shape mismatch {a.shape} vs {b.shape}"}
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(denom > 0, diff / denom, 0.0)
    fa, fb = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    cosine = float(np.dot(fa, fb) / (na * nb)) if na > 0 and nb > 0 else (1.0 if na == nb else 0.0)
    return {
        "ok": bool(np.allclose(a, b, rtol=rtol, atol=atol)),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(rel.max()),
        "cosine": round(cosine, 6),
        "shape": list(a.shape),
        "rtol": rtol,
        "atol": atol,
    }


def topk_agreement(a, b, k: int = 1) -> float:
    """Fraction of rows whose top-k argmax sets agree (classification heads).

    a, b: (N, C) logits/probs. Returns the mean over rows of
    |topk(a) ∩ topk(b)| / k — 1.0 means identical top-k everywhere.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.ndim != 2:
        return 0.0
    ta = np.argsort(-a, axis=1)[:, :k]
    tb = np.argsort(-b, axis=1)[:, :k]
    agree = [len(set(ra) & set(rb)) / k for ra, rb in zip(ta, tb)]
    return float(np.mean(agree)) if agree else 0.0
