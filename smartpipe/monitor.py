"""Streaming monitor: keeps the latest readings, runs the AI each minute, manages incidents."""
from __future__ import annotations

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

    def push(self, frame: np.ndarray, ts: float) -> dict | None:
        self.frames.append(np.asarray(frame, dtype=float))
        self.times.append(float(ts))
        if len(self.frames) < self.window:
            return None
        X = np.stack(list(self.frames)[-self.window:])[None]
        a = self.ai.predict(X)[0]
        a["ts"] = float(ts)
        self.current = self.tracker.update(a, ts)
        return self.current

    @property
    def warmup_remaining(self) -> int:
        return max(0, self.window - len(self.frames))
