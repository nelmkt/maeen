"""Turn a window of raw device readings into physics-aware features.

The key idea is to look at *relationships between neighbouring devices*
instead of raw values: a leak shows up as flow disappearing between two
meters, a blockage as excess pressure drop for the flow carried, and a
contamination front as EC/pH changing from one device to the next. Each
quantity is compared to a baseline learned during normal operation
(commissioning), so calibration offsets of individual devices cancel out.
"""
from __future__ import annotations

import numpy as np

from .config import N_DEVICES, N_SEGMENTS, SENSORS
from .simulator import Q0

RECENT = 8
NOISE_SPAN = 15


def _quantities(S: np.ndarray) -> dict[str, np.ndarray]:
    """S: (n, devices, sensors) summary readings."""
    p, q, ph, ec, ac, vb = (S[..., i] for i in range(6))
    qm = 0.5 * (q[:, :-1] + q[:, 1:])
    return {
        "flow_loss": (q[:, :-1] - q[:, 1:]) / q[:, :1],
        "friction": (p[:, :-1] - p[:, 1:]) / np.maximum(qm, 1.0) ** 2 * 1e4,
        "dph": ph[:, 1:] - ph[:, :-1],
        "dec": ec[:, 1:] - ec[:, :-1],
        "p_resid": p[:, 1:-1] - 0.5 * (p[:, :-2] + p[:, 2:]),
        "acoustic": ac - 8 * q / Q0,
        "vibration": vb,
    }


class FeatureExtractor:
    def fit(self, X_normal: np.ndarray) -> "FeatureExtractor":
        X_normal = np.asarray(X_normal, dtype=float)
        cur = X_normal[:, -RECENT:].mean(1)
        self.base = {k: np.median(v, axis=0) for k, v in _quantities(cur).items()}
        self.noise = np.median(X_normal[:, -NOISE_SPAN:].std(1), axis=0) + 1e-6
        self.names = self._names()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            X = X[None]
        n, W = X.shape[:2]
        cur = X[:, -RECENT:].mean(1)
        ref = X[:, :RECENT].mean(1)
        qc, qr = _quantities(cur), _quantities(ref)

        feats = []
        for k in qc:
            feats.append(qc[k] - self.base[k])
            feats.append(qc[k] - qr[k])
        ac = qc["acoustic"] - self.base["acoustic"]
        feats.append(ac - np.median(ac, axis=1, keepdims=True))

        lognoise = np.log((X[:, -NOISE_SPAN:].std(1) + 1e-4) / self.noise)
        level = np.median(lognoise, axis=1, keepdims=True)
        feats.append((lognoise - level).reshape(n, -1))
        feats.append(level[:, 0])
        jump = (cur - ref)[..., :4] / (self.noise[..., :4] * np.exp(level[..., :4]))
        feats.append(jump.reshape(n, -1))

        tt = np.arange(W) - (W - 1) / 2
        slope = np.einsum("t,ntds->nds", tt, X) / np.sum(tt**2)
        feats += [slope[..., 0], slope[..., 2], slope[..., 3]]
        feats.append(np.stack([cur[:, 0, 0], cur[:, 0, 1], cur[:, -1, 1] / cur[:, 0, 1]], axis=1))
        return np.concatenate(feats, axis=1)

    def index(self, name: str) -> int:
        return self.names.index(name)

    @staticmethod
    def _names() -> list[str]:
        seg = [f"S{j + 1}" for j in range(N_SEGMENTS)]
        dev = [f"D{i + 1}" for i in range(N_DEVICES)]
        per = {"flow_loss": seg, "friction": seg, "dph": seg, "dec": seg,
               "p_resid": dev[1:-1], "acoustic": dev, "vibration": dev}
        names = []
        for k, labels in per.items():
            names += [f"{k}[{l}]" for l in labels]
            names += [f"change_{k}[{l}]" for l in labels]
        names += [f"acoustic_rel[{l}]" for l in dev]
        names += [f"rel_noise[{d}.{s}]" for d in dev for s in SENSORS]
        names += [f"noise_level[{s}]" for s in SENSORS]
        names += [f"jump[{d}.{s}]" for d in dev for s in SENSORS[:4]]
        names += [f"slope_{s}[{d}]" for s in ("pressure", "ph", "ec") for d in dev]
        names += ["inlet_pressure", "q_in", "delivery_ratio"]
        return names
