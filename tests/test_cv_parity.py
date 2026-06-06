"""CV model-output parity — output_diff / topk_agreement."""
import numpy as np

from src.services.cv import parity as pq


def test_output_diff_equal():
    a = np.random.RandomState(0).rand(3, 4)
    r = pq.output_diff(a, a.copy())
    assert r["ok"] is True
    assert r["max_abs"] == 0.0
    assert r["cosine"] == 1.0


def test_output_diff_small_perturbation_within_tol():
    a = np.ones((2, 2))
    b = a + 1e-5
    r = pq.output_diff(a, b, rtol=1e-3, atol=1e-3)
    assert r["ok"] is True
    assert r["max_abs"] < 1e-3


def test_output_diff_large_diff_fails():
    r = pq.output_diff(np.zeros((2, 2)), np.ones((2, 2)))
    assert r["ok"] is False
    assert r["max_abs"] == 1.0


def test_output_diff_shape_mismatch():
    r = pq.output_diff(np.zeros((2, 2)), np.zeros((3, 2)))
    assert r["ok"] is False
    assert "shape mismatch" in r["error"]


def test_topk_agreement():
    a = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    b = np.array([[0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])  # row0 top1 agree, row1 differ
    assert pq.topk_agreement(a, b, k=1) == 0.5
    assert pq.topk_agreement(a, a, k=1) == 1.0
