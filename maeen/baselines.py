"""Baselines used to justify the design choices in the evaluation report.

RuleBasedDetector  what a classic SCADA set-up does: fixed alarm thresholds on mass balance,
                   pressure drop, EC/pH change and signal noise, calibrated on normal data
                   (99.5th percentile), first rule that fires wins.
RawFeatureModel    the same gradient-boosting model, but on raw per-device statistics
                   (mean, std, change) without the neighbour-comparison physics features.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import PIPE_FAULTS
from .features import FeatureExtractor


class RuleBasedDetector:
    def fit(self, fe: FeatureExtractor, F_normal: np.ndarray) -> "RuleBasedDetector":
        self.fe = fe
        names = fe.names
        col = lambda prefix: np.array([i for i, n in enumerate(names) if n.startswith(prefix + "[")])
        self.cols = {k: col(k) for k in ("flow_loss", "friction", "dec", "dph", "rel_noise", "jump")}
        self.thr = {k: np.quantile(np.abs(F_normal[:, c]), 0.995, axis=0) + 1e-9 for k, c in self.cols.items()}
        return self

    def predict(self, X: np.ndarray) -> list[dict]:
        F = self.fe.transform(X)
        out = []
        for f in F:
            r = {k: f[c] / self.thr[k] for k, c in self.cols.items()}
            scores = {
                "leak": r["flow_loss"].max(),
                "blockage": r["friction"].max(),
                "contamination": np.abs(r["dec"]).max() / 3,  # big EC jump
                "corrosion": min(r["dec"].max(), -r["dph"].min()),  # EC up and pH down together
                "sensor_fault": max(np.abs(r["rel_noise"]).max(), np.abs(r["jump"]).max()),
            }
            fault, score = max(scores.items(), key=lambda kv: kv[1])
            if score <= 1:
                out.append({"fault": "normal", "segment": None})
                continue
            seg = None
            if fault in PIPE_FAULTS:
                key = {"leak": "flow_loss", "blockage": "friction"}.get(fault, "dec")
                seg = int(np.argmax(np.abs(r[key])))
            out.append({"fault": fault, "segment": seg})
        return out


def raw_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = len(X)
    cur, ref = X[:, -8:].mean(1), X[:, :8].mean(1)
    return np.concatenate([cur.reshape(n, -1), X[:, -15:].std(1).reshape(n, -1), (cur - ref).reshape(n, -1)], axis=1)


class RawFeatureModel:
    def fit(self, X: np.ndarray, y, seed: int = 0) -> "RawFeatureModel":
        R = raw_features(X)
        fault = y["fault"].to_numpy()
        self.type_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=seed).fit(R, fault)
        pipe = np.isin(fault, PIPE_FAULTS)
        self.seg_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=seed).fit(
            R[pipe], y["segment"].to_numpy()[pipe])
        return self

    def predict(self, X: np.ndarray) -> list[dict]:
        R = raw_features(X)
        types = self.type_clf.predict(R)
        segs = self.seg_clf.predict(R)
        return [{"fault": str(t), "segment": int(s) if t in PIPE_FAULTS else None} for t, s in zip(types, segs)]

