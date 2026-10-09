"""Synthetic dataset generation from the simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAULT_TYPES, N_DEVICES, N_SENSORS, WINDOW
from .simulator import Network, random_scenario, simulate

DEFAULT_MIX = {"normal": 0.30, "leak": 0.25, "blockage": 0.12,
               "contamination": 0.11, "corrosion": 0.11, "sensor_fault": 0.11}


def make_dataset(n: int, seed: int, net: Network | None = None, mix: dict | None = None,
                 noise_scale: float | tuple[float, float] = 1.0, window: int = WINDOW):
    """Return windows X (n, window, devices, sensors) and a label DataFrame.

    noise_scale may be a (low, high) range to vary sensor quality between windows.
    """
    net = net or Network.default()
    mix = mix or DEFAULT_MIX
    rng = np.random.default_rng(seed)
    faults = rng.choice(list(mix), size=n, p=np.array(list(mix.values())) / sum(mix.values()))
    X = np.empty((n, window, N_DEVICES, N_SENSORS), dtype=np.float32)
    rows = []
    for i, fault in enumerate(faults):
        t0 = rng.uniform(0, 3 * 1440)
        ns = rng.uniform(*noise_scale) if isinstance(noise_scale, tuple) else noise_scale
        scn = random_scenario(rng, str(fault), t0, window, ns)
        X[i], info = simulate(net, scn, t0 + np.arange(window), rng)
        rows.append({
            "fault": str(fault), "segment": scn.segment, "position_km": scn.position_km,
            "device": scn.device, "sensor": scn.sensor, "mode": scn.mode,
            "severity": float(info["severity"][-1]), "leak_ratio": float(info["leak_ratio"][-1]),
        })
    return X, pd.DataFrame(rows)


def normal_windows(n: int, seed: int, net: Network | None = None, noise_scale=1.0) -> np.ndarray:
    """Commissioning data: normal operation only, used to learn the baseline."""
    X, _ = make_dataset(n, seed, net, mix={"normal": 1.0}, noise_scale=noise_scale)
    return X


assert set(DEFAULT_MIX) == set(FAULT_TYPES)
