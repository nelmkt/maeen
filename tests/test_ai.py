import numpy as np

from maeen.config import FAULT_TYPES
from maeen.live import LiveSimulator
from maeen.monitor import IncidentTracker, Monitor
from maeen.recommend import recommend


def test_feature_names_match_columns(small_model):
    ai, _ = small_model
    X = np.random.default_rng(0).normal(size=(3, 30, 6, 6)) + 5
    assert ai.fe.transform(X).shape == (3, len(ai.fe.names))


def test_assessment_schema_and_big_leak_is_found(small_model):
    ai, _ = small_model
    sim = LiveSimulator(seed=3)
    frames = [sim.tick()[1] for _ in range(20)]
    sim.inject("leak", segment=4, severity=0.8)
    frames += [sim.tick()[1] for _ in range(15)]
    a = ai.predict(np.stack(frames[-30:]))[0]
    for key in ("status", "fault", "fault_probs", "location", "severity", "confidence", "headline", "recommendations"):
        assert key in a
    assert a["fault"] == "leak"
    assert a["location"]["between"] == [4, 5]
    assert 3.0 <= a["location"]["position_km"] <= 4.0
    assert a["headline"]["ar"] == "احتمال تسريب بين الجهاز 4 والجهاز 5"


def test_monitor_opens_incident_after_persistence(small_model):
    ai, _ = small_model
    sim = LiveSimulator(seed=5)
    mon = Monitor(ai, persistence=3)
    for minute in range(60):
        if minute == 35:
            sim.inject("blockage", segment=2, severity=0.9)
        ts, frame, _ = sim.tick()
        mon.push(frame, ts)
    assert mon.tracker.incidents and mon.tracker.incidents[0]["fault"] == "blockage"
    assert mon.tracker.incidents[0]["start_ts"] >= 516 + 2


def test_tracker_resolves():
    tr = IncidentTracker(persistence=2)
    alert = {"status": "alert", "fault": "leak", "headline": {}, "location": None, "confidence": 0.9,
             "severity": {"score": 0.5, "level": "medium"}, "estimated_loss_m3h": 1.0}
    normal = dict(alert, status="normal", fault="normal")
    for ts, a in enumerate([alert, alert, alert, normal, normal]):
        tr.update(dict(a), ts)
    assert tr.incidents[0]["status"] == "resolved" and tr.active is None


def test_every_fault_gets_bilingual_recommendations():
    seg = {"kind": "segment", "segment": 2, "between": [2, 3], "position_km": 1.4, "confidence": 0.9, "segment_probs": []}
    dev = {"kind": "device", "device": 4, "sensor": "ph", "confidence": 0.9, "device_probs": []}
    for fault in (*FAULT_TYPES, "watch"):
        for level in ("low", "medium", "high", "critical"):
            status = {"normal": "normal", "watch": "watch"}.get(fault, "alert")
            a = {"status": status, "fault": fault, "location": dev if fault == "sensor_fault" else seg,
                 "severity": {"score": 0.5, "level": level}, "confidence": 0.4, "estimated_loss_m3h": 3.0}
            recs = recommend(a)
            assert recs and all(r["en"] and r["ar"] and r["priority"] for r in recs)
