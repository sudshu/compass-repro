"""Evaluation metrics, verbatim logic from the analysis repository.

``RAccum`` / pooled Pearson r: identical to
ml_geos_cf/scripts/eval_blocked_aug_oct_dec.py (streaming sufficient
statistics in float64, pooled over all (cell, time) samples).
"""
from __future__ import annotations

import numpy as np


class RAccum:
    """Streaming sufficient statistics for pooled Pearson r."""

    def __init__(self) -> None:
        self.n = 0.0
        self.sa = self.sb = self.sa2 = self.sb2 = self.sab = 0.0

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        a = a.ravel().astype(np.float64)
        b = b.ravel().astype(np.float64)
        self.n += a.size
        self.sa += a.sum()
        self.sb += b.sum()
        self.sa2 += (a * a).sum()
        self.sb2 += (b * b).sum()
        self.sab += (a * b).sum()

    def r(self) -> float:
        num = self.n * self.sab - self.sa * self.sb
        den = np.sqrt((self.n * self.sa2 - self.sa ** 2) *
                      (self.n * self.sb2 - self.sb ** 2))
        return float(num / (den + 1e-12))


def pooled_r(a: np.ndarray, b: np.ndarray) -> float:
    """One-shot pooled Pearson r (for small arrays / tests)."""
    acc = RAccum()
    acc.add(a, b)
    return acc.r()
