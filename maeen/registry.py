"""Minimal file-based model registry.

models/registry/<version>/model.joblib   the trained PipeAI
models/registry/<version>/card.json      metadata: config, data, features, training time, git commit
models/registry/<version>/metrics.json   written by `python -m maeen.evaluate`
models/LATEST                            name of the version the API serves
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(os.getenv("MAEEN_REGISTRY", ROOT / "models" / "registry"))
LATEST = REGISTRY.parent / "LATEST"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def register(ai, card: dict) -> str:
    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    folder = REGISTRY / version
    folder.mkdir(parents=True, exist_ok=True)
    card = {"version": version, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(), **card}
    ai.meta = card
    joblib.dump(ai, folder / "model.joblib", compress=3)
    (folder / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    LATEST.write_text(version, encoding="utf-8")
    return version


def latest_version() -> str | None:
    if LATEST.exists():
        v = LATEST.read_text(encoding="utf-8").strip()
        if (REGISTRY / v / "model.joblib").exists():
            return v
    return None


def load(version: str | None = None):
    version = version or latest_version()
    if version is None:
        raise FileNotFoundError("no registered model — run `python -m maeen.train`")
    return joblib.load(REGISTRY / version / "model.joblib")


def save_metrics(version: str, metrics: dict):
    (REGISTRY / version / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")


def list_versions() -> list[dict]:
    out = []
    for folder in sorted(REGISTRY.glob("v*"), reverse=True):
        card = json.loads((folder / "card.json").read_text(encoding="utf-8")) if (folder / "card.json").exists() else {}
        m = folder / "metrics.json"
        if m.exists():
            w = json.loads(m.read_text(encoding="utf-8"))["window_test"]
            card["fault_type_accuracy"] = w["fault_type_accuracy"]
            card["segment_accuracy"] = w["segment_accuracy"]
        out.append(card)
    return out
