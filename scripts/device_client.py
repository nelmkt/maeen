"""Emulate the 6 field devices sending readings to the cloud API over HTTP.

This is what a real device (ESP32 / Raspberry Pi with the sensors) would do every minute.
Start the server in ingest mode first:

    MAEEN_MODE=ingest uvicorn app.main:app --port 8000
    python scripts/device_client.py --url http://localhost:8000 --fault leak --segment 3 --at 45
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maeen.live import LiveSimulator  # noqa: E402


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--interval", type=float, default=0.5, help="real seconds between simulated minutes")
    ap.add_argument("--fault", choices=["leak", "blockage", "contamination", "corrosion", "sensor_fault"])
    ap.add_argument("--segment", type=int, default=3)
    ap.add_argument("--device", type=int, default=4)
    ap.add_argument("--sensor", default="pressure")
    ap.add_argument("--severity", type=float, default=0.6)
    ap.add_argument("--at", type=int, default=45, help="minute at which the fault starts")
    args = ap.parse_args()

    sim = LiveSimulator(seed=11)
    for minute in range(args.minutes):
        if args.fault and minute == args.at:
            print("injecting", sim.inject(args.fault, segment=args.segment, device=args.device,
                                          sensor=args.sensor, severity=args.severity))
        ts, frame, tele = sim.tick()
        readings = [{
            "device_id": i + 1, "ts": ts,
            "pressure_bar": f[0], "flow_m3h": f[1], "ph": f[2], "ec_us_cm": f[3], "acoustic_db": f[4], "vibration_mm_s": f[5],
            **tele[i],
        } for i, f in enumerate(frame.tolist())]
        res = post(args.url.rstrip("/") + "/api/ingest", {"readings": readings})
        a = res.get("assessment")
        status = "warming up" if a is None else f"{a['status']:6s} {a['headline']['en']} (conf {a['confidence']:.0%})"
        print(f"minute {minute:4d}  {status}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
