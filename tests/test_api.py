import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(small_model, monkeypatch):
    _, path = small_model
    monkeypatch.setenv("SMARTPIPE_MODE", "ingest")
    monkeypatch.setenv("SMARTPIPE_MODEL_PATH", str(path))
    import smartpipe.models
    import app.main

    importlib.reload(smartpipe.models)
    main = importlib.reload(app.main)
    with TestClient(main.app) as c:
        yield c


def readings(ts, frame):
    return [{"device_id": i + 1, "ts": ts, "pressure_bar": f[0], "flow_m3h": f[1], "ph": f[2],
             "ec_us_cm": f[3], "acoustic_db": f[4], "vibration_mm_s": f[5]} for i, f in enumerate(frame.tolist())]


def test_ingest_to_assessment(client):
    from smartpipe.live import LiveSimulator

    sim = LiveSimulator(seed=1)
    assert client.get("/api/health").json()["mode"] == "ingest"
    for _ in range(31):
        ts, frame, _ = sim.tick()
        r = client.post("/api/ingest", json={"readings": readings(ts, frame)})
        assert r.status_code == 200
    state = client.get("/api/state").json()
    assert state["assessment"]["status"] in ("normal", "watch", "alert")
    assert len(state["history"]["pressure"]) == 6
    assert client.get("/").status_code == 200


def test_partial_cycle_waits_for_all_devices(client):
    from smartpipe.live import LiveSimulator

    ts, frame, _ = LiveSimulator(seed=2).tick()
    r = client.post("/api/ingest", json={"readings": readings(ts, frame)[:3]})
    assert r.json()["cycles_processed"] == 0
    r = client.post("/api/ingest", json={"readings": readings(ts, frame)[3:]})
    assert r.json()["cycles_processed"] == 1


def test_rejects_bad_input_and_sim_endpoints_in_ingest_mode(client):
    assert client.post("/api/ingest", json={"readings": [{"device_id": 7, "ts": 1}]}).status_code == 422
    assert client.post("/api/sim/inject", json={"fault": "leak", "segment": 1}).status_code == 409
