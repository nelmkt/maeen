"""Tests for the ML-engineering parts: registry, explainability, baselines, quality gate, drift."""
import numpy as np

from conftest import SMALL_CONFIG, TMP
from maeen import registry
from maeen.baselines import RawFeatureModel, RuleBasedDetector
from maeen.data import make_dataset, normal_windows
from maeen.evaluate import check_gate
from maeen.live import LiveSimulator
from maeen.monitor import Monitor


def test_registry_roundtrip(small_model, monkeypatch):
    ai, _ = small_model
    monkeypatch.setattr(registry, "REGISTRY", TMP / "registry")
    monkeypatch.setattr(registry, "LATEST", TMP / "LATEST")
    version = registry.register(ai, {"config": SMALL_CONFIG, "n_features": len(ai.fe.names)})
    assert registry.latest_version() == version
    loaded = registry.load()
    assert loaded.meta["version"] == version and loaded.meta["git_commit"]
    registry.save_metrics(version, {"window_test": {"fault_type_accuracy": 0.9, "segment_accuracy": 0.8}})
    assert registry.list_versions()[0]["fault_type_accuracy"] == 0.9


def test_leak_evidence_points_at_the_leaking_segment(small_model):
    ai, _ = small_model
    sim = LiveSimulator(seed=3)
    frames = [sim.tick()[1] for _ in range(20)]
    sim.inject("leak", segment=2, severity=0.9)
    frames += [sim.tick()[1] for _ in range(15)]
    a = ai.predict(np.stack(frames[-30:]))[0]
    assert a["evidence"], "an alert must come with evidence"
    assert all(e["en"] and e["ar"] and abs(e["z"]) >= 3 for e in a["evidence"])
    assert any("D2" in e["en"] or "D3" in e["en"] for e in a["evidence"])


def test_baselines_run(small_model):
    ai, _ = small_model
    X, y = make_dataset(200, seed=9)
    rules = RuleBasedDetector().fit(ai.fe, ai.fe.transform(normal_windows(100, seed=8)))
    raw = RawFeatureModel().fit(X, y)
    for out in (rules.predict(X[:20]), raw.predict(X[:20])):
        assert len(out) == 20 and all("fault" in o for o in out)


def test_quality_gate():
    m = {"window_test": {"acc": 0.95, "far": 0.05}}
    res = check_gate(m, {"window_test.acc": {"min": 0.9}, "window_test.far": {"max": 0.02}})
    assert [r["passed"] for r in res] == [True, False]


def test_monitor_health_reports_latency_and_drift(small_model):
    ai, _ = small_model
    sim = LiveSimulator(seed=4)
    mon = Monitor(ai)
    for _ in range(50):
        ts, frame, _ = sim.tick()
        mon.push(frame, ts)
    h = mon.health()
    assert h["status"] in ("ok", "drift", "busy") and h["latency_ms_p50"] > 0
