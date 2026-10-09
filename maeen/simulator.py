"""Physics-inspired simulator of a water main instrumented with smart devices.

The main is 5 km long with a device every kilometre (D1 at the inlet, D6 at the
end). Each device reports pressure, flow, pH, EC, acoustic level and vibration
once per minute. Hydraulics are solved on a 10 m grid: friction loss ~ Q², a
source/pump curve at the inlet, pressure-dependent customer demand, metered
off-takes in the middle of every segment, and optional faults:

* leak           orifice flow ~ sqrt(P) at a point, plus leak noise that decays with distance
* blockage       extra local head loss at a point (partly closed valve, sediment)
* contamination  EC / pH front injected at a point and carried downstream
* corrosion      slow EC rise + pH drop downstream, pitting noise, tiny micro-leak
* sensor_fault   a single instrument gets stuck, drifts, jumps or becomes noisy

Normal operation includes daily demand cycles, a big customer switching on/off
and pump pressure steps so the model has to learn not to raise false alarms on them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import DEVICE_X, FAULT_TYPES, N_DEVICES, N_SEGMENTS, N_SENSORS, PIPE_FAULTS, PIPE_KM, SEGMENT_KM

Q0 = 100.0  # nominal demand, m³/h
P_SOURCE = 5.6  # bar, source head at zero flow
C_SOURCE = 6e-5  # bar / (m³/h)², source curve
F_FRICTION = 3e-5  # bar / km / (m³/h)²
P_REF = 3.5  # bar, pressure at which customers draw their nominal demand
K_BLOCK = 2e-4  # bar / (m³/h)², local loss of a full blockage
V_TRANSPORT = 0.25  # km/min, speed at which a water-quality front travels

GRID = 501
XS = np.linspace(0.0, PIPE_KM, GRID)
DX = XS[1] - XS[0]
CELL_SEG = np.minimum((XS[:-1] / SEGMENT_KM).astype(int), N_SEGMENTS - 1)
DEV_IDX = np.rint(DEVICE_X / DX).astype(int)
OFFTAKE_IDX = np.rint((DEVICE_X[:-1] + SEGMENT_KM / 2) / DX).astype(int)

# Measurement noise: pressure bar, flow (relative), pH, EC µS/cm, acoustic dB, vibration mm/s
NOISE = np.array([0.015, 0.007, 0.02, 3.0, 0.7, 0.04])
# Full-scale drift/offset of a faulty instrument: pressure bar, flow (relative), pH, EC µS/cm
SENSOR_FAULT_SCALE = np.array([0.3, 0.08, 0.6, 80.0])
SENSOR_FAULT_MODES = ("stuck", "drift", "offset", "noise")


@dataclass
class Network:
    """Fixed properties of the monitored pipeline (what the model calibrates against)."""

    roughness: np.ndarray  # per segment friction multiplier
    offtake: np.ndarray  # per segment metered customer off-take, fraction of demand
    bias: np.ndarray  # (devices, sensors) calibration error; flow column is a relative gain error
    vib_base: np.ndarray  # per device background vibration

    @classmethod
    def default(cls, seed: int = 2026) -> "Network":
        rng = np.random.default_rng(seed)
        bias = np.zeros((N_DEVICES, N_SENSORS))
        bias[:, 0] = rng.normal(0, 0.03, N_DEVICES)
        bias[:, 1] = rng.normal(0, 0.01, N_DEVICES)
        bias[:, 2] = rng.normal(0, 0.05, N_DEVICES)
        bias[:, 3] = rng.normal(0, 10.0, N_DEVICES)
        bias[:, 4] = rng.normal(0, 2.0, N_DEVICES)
        return cls(
            roughness=rng.uniform(0.85, 1.15, N_SEGMENTS),
            offtake=rng.uniform(0.02, 0.04, N_SEGMENTS),
            bias=bias,
            vib_base=rng.uniform(0.3, 0.6, N_DEVICES),
        )


@dataclass
class Scenario:
    fault: str = "normal"
    segment: int = -1  # 0-based segment index for pipe faults
    position_km: float = float("nan")
    magnitude: float = 0.0
    onset: float = float("inf")  # minute the fault starts
    ramp: float = 1.0  # minutes for the fault to reach full size
    device: int = -1  # 0-based device index for sensor faults
    sensor: int = -1  # index into FAULTY_SENSORS
    mode: str = ""
    sign: float = 1.0
    demand_phase: float = 0.0
    ripple_phase: float = 0.0
    quality_phase: float = 0.0
    ph_base: float = 7.4
    ec_base: float = 450.0
    pump_step_t: float = float("inf")
    pump_step_dp: float = 0.0
    demand_step_t: float = float("inf")
    demand_step_dq: float = 0.0
    noise_scale: float = 1.0


def base_scenario(rng: np.random.Generator, noise_scale: float = 1.0) -> Scenario:
    return Scenario(
        demand_phase=rng.uniform(0, 1440),
        ripple_phase=rng.uniform(0, 2 * np.pi),
        quality_phase=rng.uniform(0, 2 * np.pi),
        ph_base=rng.uniform(7.1, 7.8),
        ec_base=rng.uniform(380, 520),
        noise_scale=noise_scale,
    )


def random_scenario(
    rng: np.random.Generator,
    fault: str,
    t_start: float,
    window: int,
    noise_scale: float = 1.0,
    onset: float | None = None,
) -> Scenario:
    """Draw a random scenario whose fault (if any) is visible inside [t_start, t_start + window)."""
    assert fault in FAULT_TYPES, fault
    scn = base_scenario(rng, noise_scale)
    if rng.random() < 0.35:  # pump/valve operation at the source
        scn.pump_step_t = t_start + rng.uniform(0, window)
        scn.pump_step_dp = rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.2)
    if rng.random() < 0.35:  # large customer switching on/off
        scn.demand_step_t = t_start + rng.uniform(0, window)
        scn.demand_step_dq = rng.uniform(-0.12, 0.12)
    scn.fault = fault
    if fault == "normal":
        return scn

    if onset is None:
        latest = window - (6 if fault in ("contamination", "corrosion") else 4)
        onset = t_start + rng.uniform(-25, latest)
    scn.onset = onset
    if fault in PIPE_FAULTS:
        scn.segment = int(rng.integers(N_SEGMENTS))
        scn.position_km = float(DEVICE_X[scn.segment] + rng.uniform(0.08, 0.92) * SEGMENT_KM)
    if fault == "leak":
        scn.magnitude = float(np.exp(rng.uniform(np.log(0.01), np.log(0.25))))
        scn.ramp = rng.uniform(1, 12)
    elif fault == "blockage":
        scn.magnitude = rng.uniform(0.15, 1.0)
        scn.ramp = rng.uniform(1, 15)
    elif fault == "contamination":
        scn.magnitude = rng.uniform(0.2, 1.0)
        scn.ramp = rng.uniform(1, 6)
        scn.sign = rng.choice([-1.0, 1.0])
    elif fault == "corrosion":
        scn.magnitude = rng.uniform(0.3, 1.0)
        scn.ramp = rng.uniform(15, 35)
    elif fault == "sensor_fault":
        scn.device = int(rng.integers(N_DEVICES))
        scn.sensor = int(rng.integers(len(SENSOR_FAULT_SCALE)))
        scn.mode = str(rng.choice(SENSOR_FAULT_MODES))
        scn.magnitude = rng.uniform(0.4, 1.0)
        scn.sign = rng.choice([-1.0, 1.0])
        scn.ramp = 20.0
    return scn


def demand(scn: Scenario, t: np.ndarray) -> np.ndarray:
    q = Q0 * (1 + 0.3 * np.sin(2 * np.pi * (t + scn.demand_phase) / 1440))
    q = q * (1 + 0.03 * np.sin(2 * np.pi * t / 37 + scn.ripple_phase))
    return q * (1 + scn.demand_step_dq * (t >= scn.demand_step_t))


def _physical(net: Network, scn: Scenario, t: np.ndarray):
    """Noise-free device readings (including calibration bias) and ground-truth info."""
    T = len(t)
    f = scn.fault
    r = np.clip((t - scn.onset) / scn.ramp, 0, 1) if f != "normal" else np.zeros(T)
    qd = demand(scn, t)
    p_off = scn.pump_step_dp * (t >= scn.pump_step_t)

    leak_s = np.zeros(T)
    block_b = np.zeros(T)
    if f == "leak":
        leak_s = scn.magnitude * r
    elif f == "corrosion":
        leak_s = 0.012 * scn.magnitude * r
    elif f == "blockage":
        block_b = scn.magnitude * r
    has_sink = f in ("leak", "corrosion")
    k = int(np.clip(np.rint(scn.position_km / DX), 1, GRID - 2)) if f in PIPE_FAULTS else 0

    rough = F_FRICTION * net.roughness[CELL_SEG] * DX
    p_end = np.full(T, P_REF)
    p_leak = np.full(T, 4.0)
    for _ in range(12):  # damped fixed point: demand and leak flow both depend on pressure
        factor = np.sqrt(np.clip(p_end, 0.05, None) / P_REF)
        q_end = qd * factor
        sinks = np.zeros((T, GRID))
        sinks[:, OFFTAKE_IDX] = net.offtake * q_end[:, None]
        ql = leak_s * Q0 * np.sqrt(np.clip(p_leak, 0, None) / 4.0)
        if has_sink:
            sinks[:, k] += ql
        down = np.cumsum(sinks[:, ::-1], axis=1)[:, ::-1]  # flow withdrawn at or downstream of each grid point
        q_cell = q_end[:, None] + down[:, 1:]
        q_in = q_end + down[:, 0]
        p_in = P_SOURCE - C_SOURCE * q_in**2 + p_off
        dp = rough * q_cell**2
        if f == "blockage":
            dp[:, k] += K_BLOCK * block_b * q_cell[:, k] ** 2
        p = np.concatenate([p_in[:, None], p_in[:, None] - np.cumsum(dp, axis=1)], axis=1)
        p_end = 0.5 * p_end + 0.5 * p[:, -1]
        if has_sink:
            p_leak = 0.5 * p_leak + 0.5 * p[:, k]

    q_dev = np.concatenate([q_cell[:, DEV_IDX[:-1]], q_end[:, None]], axis=1)
    p_dev = p[:, DEV_IDX]
    x = DEVICE_X[None, :]
    ones = np.ones((1, N_DEVICES))
    ph = (scn.ph_base + 0.04 * np.sin(2 * np.pi * t / 180 + scn.quality_phase))[:, None] * ones
    ec = (scn.ec_base + 15 * np.sin(2 * np.pi * t / 240 + 1.3 * scn.quality_phase))[:, None] * ones
    acoustic = 30 + 8 * q_dev / Q0
    vib = net.vib_base + 0.25 * (q_dev / Q0) ** 2
    severity = np.zeros(T)
    leak_ratio = ql / q_in if f == "leak" else np.zeros(T)

    if f in PIPE_FAULTS:
        d = np.abs(x - scn.position_km)
    if f == "leak":
        acoustic = acoustic + 20 * np.log10(1 + 40 * (ql / Q0)[:, None] * np.exp(-d / 0.7))
        vib = vib + 3 * (ql / Q0)[:, None] * np.exp(-d / 0.5)
        severity = np.clip(leak_ratio / 0.2, 0, 1)
    elif f == "blockage":
        acoustic = acoustic + 10 * np.log10(1 + 4 * block_b[:, None] * np.exp(-d / 0.5))
        vib = vib + 1.8 * block_b[:, None] * np.exp(-d / 0.45)
        severity = block_b
    elif f in ("contamination", "corrosion"):
        downstream = x > scn.position_km
        delay = np.where(downstream, (x - scn.position_km) / V_TRANSPORT, 0)
        rr = np.clip((t[:, None] - scn.onset - delay) / scn.ramp, 0, 1) * downstream
        atten = np.exp(-np.clip(x - scn.position_km, 0, None) / 6)
        first = rr[:, scn.segment + 1]
        if f == "contamination":
            ec = ec + scn.magnitude * 500 * rr * atten
            ph = ph + scn.sign * scn.magnitude * 1.1 * rr * atten
            severity = (0.35 + 0.65 * scn.magnitude) * first
        else:
            ec = ec + scn.magnitude * 90 * rr * atten
            ph = ph - scn.magnitude * 0.35 * rr * atten
            local = np.exp(-d / 0.6)
            acoustic = acoustic + 10 * np.log10(1 + 1.5 * scn.magnitude * r[:, None] * local)
            vib = vib + 0.3 * scn.magnitude * r[:, None] * local
            severity = (0.15 + 0.45 * scn.magnitude) * first
    elif f == "sensor_fault":
        severity = np.where(t >= scn.onset, 0.3 + 0.2 * scn.magnitude, 0.0)

    true = np.stack([p_dev, q_dev, ph, ec, acoustic, vib], axis=-1)
    true[..., 0] += net.bias[:, 0]
    true[..., 1] *= 1 + net.bias[:, 1]
    true[..., 2:5] += net.bias[:, 2:5]
    info = {"severity": severity, "leak_flow": ql if f == "leak" else np.zeros(T), "leak_ratio": leak_ratio, "q_in": q_in}
    return true, info


def _apply_sensor_fault(net: Network, scn: Scenario, t: np.ndarray, meas: np.ndarray, rng: np.random.Generator):
    active = t >= scn.onset
    if not active.any():
        return
    d, s = scn.device, scn.sensor
    full = scn.sign * scn.magnitude * SENSOR_FAULT_SCALE[s]
    if scn.mode == "stuck":
        frozen, _ = _physical(net, replace(scn, fault="normal"), np.array([scn.onset]))
        meas[active, d, s] = frozen[0, d, s]
        return
    if scn.mode == "drift":
        delta = full * np.clip((t - scn.onset) / scn.ramp, 0, 1)
    elif scn.mode == "offset":
        delta = full * active
    else:  # noise
        delta = rng.normal(0, 8 * NOISE[s] * scn.magnitude * scn.noise_scale, len(t)) * active
    if s == 1:  # flow faults are relative
        meas[:, d, s] *= 1 + delta
    else:
        meas[:, d, s] += delta


def simulate(net: Network, scn: Scenario, t, rng: np.random.Generator):
    """Return measured readings with shape (len(t), devices, sensors) plus ground-truth info."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    true, info = _physical(net, scn, t)
    noise = NOISE * scn.noise_scale
    eps = rng.standard_normal(true.shape)
    meas = true.copy()
    meas[..., 0] += eps[..., 0] * noise[0]
    meas[..., 1] *= 1 + eps[..., 1] * noise[1]
    meas[..., 2:] += eps[..., 2:] * noise[2:]
    if scn.fault == "sensor_fault":
        _apply_sensor_fault(net, scn, t, meas, rng)
    return meas, info
