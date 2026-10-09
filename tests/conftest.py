import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TMP = Path(__file__).resolve().parent / ".tmp"

SMALL_CONFIG = {
    "name": "test",
    "data": {"n_train": 1500, "n_baseline": 200, "train_noise": [0.6, 2.2], "seed": 0},
    "model": {"max_iter": 150, "learning_rate": 0.1, "l2_regularization": 1.0, "iso_estimators": 100, "anomaly_quantile": 0.99},
    "eval": {"n_test": 300, "n_noisy": 100, "runs_per_fault": 2, "normal_runs": 2, "importance_samples": 0},
}


@pytest.fixture(scope="session")
def small_model():
    """A quickly trained model (small data) shared by the tests."""
    from maeen.train import train

    ai = train(SMALL_CONFIG, verbose=False)
    path = TMP / "pipe_ai.joblib"
    ai.save(path)
    return ai, path
