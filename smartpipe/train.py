"""Train the AI on simulated data:  python -m smartpipe.train"""
from __future__ import annotations

import argparse
import time

from .data import make_dataset, normal_windows
from .models import DEFAULT_MODEL_PATH, PipeAI
from .simulator import Network


TRAIN_NOISE = (0.6, 2.2)


def train(n_train: int = 9000, n_baseline: int = 800, seed: int = 0, verbose: bool = True) -> PipeAI:
    net = Network.default()
    t = time.time()
    X_base = normal_windows(n_baseline, seed=seed + 1, net=net)
    # sensor quality varies between windows so the AI copes with cheap or ageing sensors
    X, y = make_dataset(n_train, seed=seed + 2, net=net, noise_scale=TRAIN_NOISE)
    if verbose:
        print(f"simulated {n_train} training windows + {n_baseline} commissioning windows in {time.time() - t:.1f}s")
        print(y["fault"].value_counts().to_string())
    t = time.time()
    ai = PipeAI(seed=seed).fit(X, y, X_base)
    if verbose:
        print(f"trained models in {time.time() - t:.1f}s ({len(ai.fe.names)} features)")
    return ai


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = ap.parse_args()
    ai = train(args.n_train, seed=args.seed)
    ai.save(args.out)
    print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
