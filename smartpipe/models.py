"""The AI engine: anomaly detection, fault diagnosis, localisation and severity."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest

from .config import DEVICE_X, FAULTY_SENSORS, PIPE_FAULTS, SEGMENT_KM, severity_level
from .features import FeatureExtractor
from .recommend import headline, recommend

DEFAULT_MODEL_PATH = Path(os.getenv("SMARTPIPE_MODEL_PATH", Path(__file__).resolve().parents[1] / "models" / "pipe_ai.joblib"))


def _clf(seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=seed)


def _reg(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=seed)


class PipeAI:
    """Five cooperating models on top of one feature extractor.

    1. IsolationForest   learns what "normal" looks like (unsupervised) → anomaly score
    2. fault classifier  normal / leak / blockage / contamination / corrosion / sensor_fault
    3. segment locator   which segment (between which two devices) holds a pipe fault, + km estimate
    4. device locator    which device and which instrument is faulty (sensor faults)
    5. severity model    0–1 severity score → low / medium / high / critical
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: np.ndarray, y: pd.DataFrame, X_baseline: np.ndarray) -> "PipeAI":
        self.fe = FeatureExtractor().fit(X_baseline)
        F = self.fe.transform(X)
        fault = y["fault"].to_numpy()

        normal_idx = np.flatnonzero(fault == "normal")
        rng = np.random.default_rng(self.seed)
        rng.shuffle(normal_idx)
        cut = int(0.75 * len(normal_idx))
        self.iso = IsolationForest(n_estimators=300, random_state=self.seed).fit(F[normal_idx[:cut]])
        held_out = -self.iso.score_samples(F[normal_idx[cut:]])
        self.iso_threshold = float(np.quantile(held_out, 0.99))

        self.type_clf = _clf(self.seed).fit(F, fault)
        pipe = np.isin(fault, PIPE_FAULTS)
        self.seg_clf = _clf(self.seed).fit(F[pipe], y["segment"].to_numpy()[pipe])
        self.pos_reg = _reg(self.seed).fit(F[pipe], y["position_km"].to_numpy()[pipe])
        sf = fault == "sensor_fault"
        self.dev_clf = _clf(self.seed).fit(F[sf], y["device"].to_numpy()[sf])
        self.sensor_clf = _clf(self.seed).fit(F[sf], y["sensor"].to_numpy()[sf])
        ab = fault != "normal"
        self.sev_reg = _reg(self.seed).fit(F[ab], y["severity"].to_numpy()[ab])
        return self

    def predict(self, X: np.ndarray) -> list[dict]:
        F = self.fe.transform(X)
        anomaly = -self.iso.score_samples(F)
        p_type = self.type_clf.predict_proba(F)
        p_seg = self.seg_clf.predict_proba(F)
        pos = self.pos_reg.predict(F)
        p_dev = self.dev_clf.predict_proba(F)
        p_sen = self.sensor_clf.predict_proba(F)
        sev = np.clip(self.sev_reg.predict(F), 0, 1)
        types = self.type_clf.classes_
        q_in = F[:, self.fe.index("q_in")]
        loss_cols = [self.fe.index(f"flow_loss[S{j + 1}]") for j in range(len(self.seg_clf.classes_))]

        out = []
        for i in range(len(F)):
            k = int(np.argmax(p_type[i]))
            fault = str(types[k])
            is_anomaly = bool(anomaly[i] > self.iso_threshold)
            status = "alert" if fault != "normal" else ("watch" if is_anomaly else "normal")
            a = {
                "status": status,
                "fault": fault if status != "watch" else "watch",
                "fault_probs": {str(t): float(p) for t, p in zip(types, p_type[i])},
                "anomaly_score": float(anomaly[i]),
                "anomaly_threshold": self.iso_threshold,
                "anomaly": is_anomaly,
                "location": None,
                "severity": {"score": 0.0, "level": "low"},
                "confidence": float(p_type[i, k]),
                "estimated_loss_m3h": None,
            }
            if fault in PIPE_FAULTS:
                j = int(np.argmax(p_seg[i]))
                seg = int(self.seg_clf.classes_[j])
                lo = DEVICE_X[seg]
                a["location"] = {
                    "kind": "segment", "segment": seg + 1, "between": [seg + 1, seg + 2],
                    "position_km": float(np.clip(pos[i], lo + 0.02, lo + SEGMENT_KM - 0.02)),
                    "confidence": float(p_seg[i, j]),
                    "segment_probs": [float(p) for p in p_seg[i]],
                }
                a["confidence"] = float(p_type[i, k] * p_seg[i, j])
                if fault == "leak":
                    a["estimated_loss_m3h"] = float(max(0.0, F[i, loss_cols[seg]]) * q_in[i])
            elif fault == "sensor_fault":
                d = int(np.argmax(p_dev[i]))
                s = int(np.argmax(p_sen[i]))
                a["location"] = {
                    "kind": "device", "device": int(self.dev_clf.classes_[d]) + 1,
                    "sensor": FAULTY_SENSORS[int(self.sensor_clf.classes_[s])],
                    "confidence": float(p_dev[i, d]),
                    "device_probs": [float(p) for p in p_dev[i]],
                }
                a["confidence"] = float(p_type[i, k] * p_dev[i, d])
            if status != "normal":
                score = float(sev[i]) if status == "alert" else 0.2
                a["severity"] = {"score": score, "level": severity_level(score)}
            a["headline"] = headline(a)
            a["recommendations"] = recommend(a)
            out.append(a)
        return out

    def save(self, path: Path = DEFAULT_MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path = DEFAULT_MODEL_PATH) -> "PipeAI":
        return joblib.load(path)


def load_or_train(path: Path = DEFAULT_MODEL_PATH, n_train: int = 9000) -> PipeAI:
    path = Path(path)
    if path.exists():
        return PipeAI.load(path)
    from .train import train

    ai = train(n_train=n_train)
    ai.save(path)
    return ai
