"""Shared constants describing the monitored pipeline and the fault taxonomy."""
import numpy as np

N_DEVICES = 6
SEGMENT_KM = 1.0
DEVICE_X = np.arange(N_DEVICES, dtype=float) * SEGMENT_KM  # device positions along the main (km)
PIPE_KM = float(DEVICE_X[-1])
N_SEGMENTS = N_DEVICES - 1

SENSORS = ("pressure", "flow", "ph", "ec", "acoustic", "vibration")
N_SENSORS = len(SENSORS)
SENSOR_UNITS = {
    "pressure": "bar",
    "flow": "m³/h",
    "ph": "pH",
    "ec": "µS/cm",
    "acoustic": "dB",
    "vibration": "mm/s",
}
# Sensors that can develop an instrument fault in the simulator.
FAULTY_SENSORS = SENSORS[:4]

WINDOW = 30  # minutes of readings the model looks at for every decision

FAULT_TYPES = ("normal", "leak", "blockage", "contamination", "corrosion", "sensor_fault")
PIPE_FAULTS = ("leak", "blockage", "contamination", "corrosion")

SEVERITY_LEVELS = (("low", 0.25), ("medium", 0.5), ("high", 0.75), ("critical", np.inf))


def severity_level(score: float) -> str:
    for name, upper in SEVERITY_LEVELS:
        if score < upper:
            return name
    return "critical"
