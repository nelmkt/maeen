"""Cloud service: ingests device readings, runs the ML model every minute, serves the dashboard.

Run:  uvicorn app.main:app --port 8000

MAEEN_MODE=simulate  (default) an in-process emulator plays the 6 devices, faults can be
                         injected from the dashboard
MAEEN_MODE=ingest    readings arrive from real devices or scripts/device_client.py via POST /api/ingest
MAEEN_TICK           seconds per simulated minute (default 1.0)
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from maeen.config import DEVICE_X, FAULTY_SENSORS, N_DEVICES, SENSORS, SENSOR_UNITS
from maeen.live import LiveSimulator
from maeen import registry
from maeen.models import load_or_train
from maeen.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
MODE = os.getenv("MAEEN_MODE", "simulate")
TICK = float(os.getenv("MAEEN_TICK", "1.0"))
HISTORY = 120


class Reading(BaseModel):
    device_id: int = Field(ge=1, le=N_DEVICES)
    ts: float = Field(description="minute timestamp shared by the readings of one cycle")
    pressure_bar: float
    flow_m3h: float
    ph: float
    ec_us_cm: float
    acoustic_db: float
    vibration_mm_s: float
    battery_pct: float | None = None
    power_w: float | None = None
    rssi_dbm: float | None = None


class IngestBatch(BaseModel):
    readings: list[Reading]


class Injection(BaseModel):
    fault: str
    segment: int | None = Field(None, ge=1, le=N_DEVICES - 1)
    device: int | None = Field(None, ge=1, le=N_DEVICES)
    sensor: str | None = None
    severity: float = Field(0.6, ge=0.05, le=1.0)
    mode: str = "drift"


class Hub:
    """All mutable state, guarded by one lock (requests and the simulator run concurrently)."""

    def __init__(self, ai):
        self.lock = threading.Lock()
        self.monitor = Monitor(ai)
        self.sim = LiveSimulator(seed=7) if MODE == "simulate" else None
        self.pending: dict[float, dict[int, Reading]] = {}
        self.telemetry: list[dict] = [{} for _ in range(N_DEVICES)]
        self.last_ts: float | None = None

    def push_frame(self, ts: float, frame: np.ndarray, telemetry: list[dict]):
        with self.lock:
            if self.last_ts is not None and ts <= self.last_ts:
                return
            self.last_ts = ts
            self.telemetry = telemetry
            self.monitor.push(frame, ts)

    def ingest(self, readings: list[Reading]) -> int:
        completed = []
        with self.lock:
            for r in readings:
                self.pending.setdefault(r.ts, {})[r.device_id] = r
            for ts in sorted(self.pending):
                if len(self.pending[ts]) == N_DEVICES:
                    completed.append((ts, self.pending.pop(ts)))
            # drop incomplete cycles that are clearly stale
            for ts in [t for t in self.pending if completed and t < completed[-1][0]]:
                del self.pending[ts]
        for ts, cycle in completed:
            rows = [cycle[d + 1] for d in range(N_DEVICES)]
            frame = np.array([[r.pressure_bar, r.flow_m3h, r.ph, r.ec_us_cm, r.acoustic_db, r.vibration_mm_s] for r in rows])
            tele = [{"battery_pct": r.battery_pct, "power_w": r.power_w, "rssi_dbm": r.rssi_dbm} for r in rows]
            self.push_frame(ts, frame, tele)
        return len(completed)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


async def _simulate_forever(hub: Hub):
    while True:
        with hub.lock:
            ts, frame, tele = hub.sim.tick()
        await asyncio.to_thread(hub.push_frame, ts, frame, tele)
        await asyncio.sleep(TICK)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ai = await asyncio.to_thread(load_or_train)
    app.state.hub = Hub(ai)
    task = asyncio.create_task(_simulate_forever(app.state.hub)) if MODE == "simulate" else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Maeen", version="0.2.0", lifespan=lifespan,
              description="ML service for water-pipeline monitoring: detect, diagnose, locate, rate and recommend.")


def hub() -> Hub:
    return app.state.hub


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    return {"ok": True, "mode": MODE}


@app.post("/api/ingest")
def ingest(batch: IngestBatch):
    h = hub()
    if h.sim is not None:
        raise HTTPException(409, "server is running its own simulator (MAEEN_MODE=simulate)")
    cycles = h.ingest(batch.readings)
    return _clean({"cycles_processed": cycles, "assessment": h.monitor.current})


@app.get("/api/state")
def state():
    h = hub()
    with h.lock:
        m = h.monitor
        frames = np.array(m.frames)[-HISTORY:] if m.frames else np.zeros((0, N_DEVICES, len(SENSORS)))
        times = list(m.times)[-HISTORY:]
        devices = []
        for i in range(N_DEVICES):
            latest = {s: float(frames[-1, i, k]) for k, s in enumerate(SENSORS)} if len(frames) else None
            devices.append({"id": i + 1, "x_km": float(DEVICE_X[i]), "latest": latest, **h.telemetry[i]})
        body = {
            "mode": MODE,
            "time": times[-1] if times else None,
            "warmup_remaining": m.warmup_remaining,
            "sensors": [{"key": s, "unit": SENSOR_UNITS[s]} for s in SENSORS],
            "devices": devices,
            "history": {"times": times, **{s: frames[:, :, k].T.round(4).tolist() for k, s in enumerate(SENSORS)}},
            "assessment": m.current,
            "incidents": list(reversed(m.tracker.incidents[-20:])),
            "model_health": m.health(),
            "model_version": m.ai.meta.get("version"),
            "ground_truth": h.sim.truth() if h.sim else None,
        }
    return _clean(body)


@app.post("/api/sim/inject")
def inject(inj: Injection):
    h = hub()
    if h.sim is None:
        raise HTTPException(409, "fault injection is only available in simulate mode")
    if inj.fault == "sensor_fault" and inj.sensor not in (None, *FAULTY_SENSORS):
        raise HTTPException(422, f"sensor must be one of {FAULTY_SENSORS}")
    try:
        with h.lock:
            truth = h.sim.inject(inj.fault, segment=inj.segment, device=inj.device, sensor=inj.sensor,
                                 severity=inj.severity, mode=inj.mode)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _clean(truth)


@app.post("/api/sim/reset")
def reset():
    h = hub()
    if h.sim is None:
        raise HTTPException(409, "fault injection is only available in simulate mode")
    with h.lock:
        h.sim.reset()
    return {"fault": "normal"}


@app.get("/api/model")
def model_info():
    """Model card of the served model, live health/drift indicators and the registry history."""
    h = hub()
    with h.lock:
        health = h.monitor.health()
    meta = {k: v for k, v in h.monitor.ai.meta.items() if k != "config"}
    return _clean({"served": meta, "config": h.monitor.ai.meta.get("config"), "health": health,
                   "registry": registry.list_versions()[:10]})


@app.get("/api/metrics")
def metrics():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "run `python -m maeen.evaluate` first")
    m = json.loads(path.read_text(encoding="utf-8"))
    w, s = m["window_test"], m["streaming_test"]
    return {
        "fault_type_accuracy": w["fault_type_accuracy"],
        "segment_accuracy": w["segment_accuracy"],
        "detection_rate": w["detection_rate"],
        "false_alarms_per_day": s["false_alarms_per_day"],
        "median_delay_min": s["median_delay_min"],
        "position_mae_m": w["position_mae_km"] * 1000,
    }
