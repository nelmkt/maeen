import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def small_model():
    """A quickly trained model (small data) shared by the tests."""
    from smartpipe.train import train

    ai = train(n_train=1500, n_baseline=200, verbose=False)
    path = Path(__file__).resolve().parent / ".tmp" / "pipe_ai.joblib"
    ai.save(path)
    return ai, path
