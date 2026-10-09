"""Train the AI on simulated data and register the model:

    python -m maeen.train                          # configs/default.json
    python -m maeen.train --config configs/ci.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from . import registry
from .data import make_dataset, normal_windows
from .models import PipeAI
from .simulator import Network

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train(cfg: dict, verbose: bool = True) -> PipeAI:
    d = cfg["data"]
    seed = d["seed"]
    net = Network.default()
    t = time.time()
    X_base = normal_windows(d["n_baseline"], seed=seed + 1, net=net)
    # sensor quality varies between windows so the AI copes with cheap or ageing sensors
    X, y = make_dataset(d["n_train"], seed=seed + 2, net=net, noise_scale=tuple(d["train_noise"]))
    sim_s = time.time() - t
    if verbose:
        print(f"simulated {d['n_train']} training windows + {d['n_baseline']} commissioning windows in {sim_s:.1f}s")
        print(y["fault"].value_counts().to_string())
    t = time.time()
    ai = PipeAI(seed=seed, params=cfg["model"]).fit(X, y, X_base)
    fit_s = time.time() - t
    if verbose:
        print(f"trained models in {fit_s:.1f}s ({len(ai.fe.names)} features)")
    ai.meta = {
        "config": cfg,
        "training_windows": int(len(y)),
        "class_counts": {k: int(v) for k, v in y["fault"].value_counts().items()},
        "n_features": len(ai.fe.names),
        "feature_hash": hashlib.sha1("|".join(ai.fe.names).encode()).hexdigest()[:10],
        "simulation_seconds": round(sim_s, 1),
        "training_seconds": round(fit_s, 1),
    }
    return ai


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", help="also save the model to this path (outside the registry)")
    args = ap.parse_args()
    ai = train(load_config(args.config))
    version = registry.register(ai, ai.meta)
    print(f"registered model {version} → {registry.REGISTRY / version}")
    if args.out:
        ai.save(Path(args.out))


if __name__ == "__main__":
    main()
