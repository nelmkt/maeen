"""Streaming monitor: keeps the latest readings, runs the model each minute, manages incidents."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .config import WINDOW


class IncidentTracker:
    """Debounces per-minute assessments into incidents.

    An incident opens when the same fault is diagnosed for `persistence`
    consecutive minutes, and resolves after `persistence` minutes without it.
    """

    def __init__(self, persistence: int = 3):
        self.persistence = persistence
        self.incidents: list[dict] = []
        self.active: dict | None = None
        self._key = None
        self._streak = 0

    def update(self, a: dict, ts: float) -> dict:
        key = a["fault"] if a["status"] == "alert" else None
        if key == self._key:
            self._streak += 1
        else:
            self._key, self._streak = key, 1
        a["streak"] = self._streak
        a["confirmed"] = bool(key and self._streak >= self.persistence)

        if a["confirmed"]:
            if self.active is None or self.active["fault"] != key:
                if self.active is not None:
                    self._close(ts)
                self.active = {"id": len(self.incidents) + 1, "fault": key, "start_ts": ts, "end_ts": None,
                               "status": "active", "max_severity": 0.0}
                self.incidents.append(self.active)
            self.active.update(
                last_ts=ts, headline=a["headline"], location=a["location"], confidence=a["confidence"],
                severity=a["severity"], estimated_loss_m3h=a["estimated_loss_m3h"],
                max_severity=max(self.active["max_severity"], a["severity"]["score"]),
            )
        elif key is None and self.active is not None and self._streak >= self.persistence:
            self._close(ts)
        return a

    def _close(self, ts: float):
        self.active["status"] = "resolved"
        self.active["end_ts"] = ts
        self.active = None


class Monitor:
    def __init__(self, ai, window: int = WINDOW, persistence: int = 3, history: int = 180):
        self.ai = ai
        self.window = window
        self.frames: deque = deque(maxlen=max(window, history))
        self.times: deque = deque(maxlen=max(window, history))
        self.tracker = IncidentTracker(persistence)
        self.current: dict | None = None
        self.recent: deque = deque(maxlen=60)

    def push(self, frame: np.ndarray, ts: float) -> dict | None:
        self.frames.append(np.asarray(frame, dtype=float))
        self.times.append(float(ts))
        if len(self.frames) < self.window:
            return None
        X = np.stack(list(self.frames)[-self.window:])[None]
        t0 = time.perf_counter()
        a = self.ai.predict(X)[0]
        latency = (time.perf_counter() - t0) * 1000
        a["ts"] = float(ts)
        self.recent.append((a["anomaly"], a["input_deviation"], latency, a["status"]))
        self.current = self.tracker.update(a, ts)
        return self.current

    def health(self) -> dict:
        """Model-health / drift indicators over the last hour.

        Outside active incidents the input should look like the training "normal" data:
        median |z| of the features ~0.7 and ~1% of windows above the anomaly threshold.
        Much higher values mean the field data has drifted away from the training data
        (new site, sensor ageing, seasonal change) and the baseline should be re-learned.
        """
        quiet = [r for r in self.recent if r[3] != "alert"]
        if not self.recent:
            return {"status": "warming_up"}
        lat = [r[2] for r in self.recent]
        out = {"windows": len(self.recent), "latency_ms_p50": float(np.median(lat)), "latency_ms_max": float(np.max(lat))}
        if len(quiet) >= 10:
            out["anomaly_rate"] = float(np.mean([r[0] for r in quiet]))
            out["input_deviation"] = float(np.median([r[1] for r in quiet]))
            drift = out["anomaly_rate"] > 0.15 or out["input_deviation"] > 1.5
            out["status"] = "drift" if drift else "ok"
        else:
            out["status"] = "busy"
        return out

    @property
    def warmup_remaining(self) -> int:
        return max(0, self.window - len(self.frames))
