"""Live device emulator used by the cloud demo (one reading per device per tick)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import DEVICE_X, FAULTY_SENSORS, N_DEVICES, SEGMENT_KM
from .simulator import Q0, Network, base_scenario, simulate

# severity slider (0–1) → simulator magnitude and ramp per fault
_FAULT_SETUP = {
    "leak": lambda s: (0.25 * s, 3.0),
    "blockage": lambda s: (s, 5.0),
    "contamination": lambda s: (s, 3.0),
    "corrosion": lambda s: (0.3 + 0.7 * s, 20.0),
    "sensor_fault": lambda s: (0.4 + 0.6 * s, 10.0),
}


class LiveSimulator:
    def __init__(self, net: Network | None = None, seed: int | None = None, start_t: float = 480.0):
        self.net = net or Network.default()
        self.rng = np.random.default_rng(seed)
        self.t = start_t
        self.base = replace(base_scenario(self.rng), ph_base=7.4, ec_base=450.0)
        self.scenario = self.base
        self.battery = np.full(N_DEVICES, 92.0) + self.rng.uniform(-4, 4, N_DEVICES)

    def inject(self, fault: str, segment: int | None = None, device: int | None = None,
               sensor: str | None = None, severity: float = 0.6, mode: str = "drift",
               position_km: float | None = None) -> dict:
        """segment/device are 1-based, as shown on the dashboard."""
        if fault not in _FAULT_SETUP:
            raise ValueError(f"unknown fault {fault!r}")
        severity = float(np.clip(severity, 0.05, 1.0))
        magnitude, ramp = _FAULT_SETUP[fault](severity)
        kw = dict(fault=fault, magnitude=magnitude, ramp=ramp, onset=self.t + 1)
        if fault == "sensor_fault":
            kw.update(device=(device or 1) - 1, sensor=FAULTY_SENSORS.index(sensor or "pressure"),
                      mode=mode, sign=1.0)
        else:
            seg = (segment or 3) - 1
            pos = position_km if position_km is not None else DEVICE_X[seg] + self.rng.uniform(0.2, 0.8) * SEGMENT_KM
            kw.update(segment=seg, position_km=float(pos), sign=-1.0)
        self.scenario = replace(self.base, **kw)
        return self.truth()

    def reset(self):
        self.scenario = self.base

    def truth(self) -> dict:
        s = self.scenario
        if s.fault == "normal":
            return {"fault": "normal"}
        if s.fault == "sensor_fault":
            return {"fault": s.fault, "device": s.device + 1, "sensor": FAULTY_SENSORS[s.sensor],
                    "mode": s.mode, "since": s.onset}
        return {"fault": s.fault, "segment": s.segment + 1, "between": [s.segment + 1, s.segment + 2],
                "position_km": s.position_km, "magnitude": s.magnitude, "since": s.onset}

    def tick(self):
        """Advance one minute; returns (ts, frame (devices, sensors), telemetry list)."""
        frame, _ = simulate(self.net, self.scenario, [self.t], self.rng)
        frame = frame[0]
        power = 15 * np.clip(frame[:, 1] / Q0, 0, None) ** 3  # micro-hydro turbine output, W
        self.battery = np.clip(self.battery + 0.02 * (power - 6) + self.rng.normal(0, 0.05, N_DEVICES), 20, 100)
        telemetry = [{"battery_pct": float(b), "power_w": float(p), "rssi_dbm": float(-60 - 4 * i + self.rng.normal(0, 2))}
                     for i, (b, p) in enumerate(zip(self.battery, power))]
        ts = self.t
        self.t += 1
        return ts, frame, telemetry
