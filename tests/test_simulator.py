from dataclasses import replace

import numpy as np

from maeen.simulator import Network, Q0, _physical, base_scenario

NET = Network.default()
T = np.arange(100.0, 110.0)


def scenario(**kw):
    return replace(base_scenario(np.random.default_rng(0)), **kw)


def test_normal_flow_balance_matches_offtakes():
    true, _ = _physical(NET, scenario(), T)
    q = true[..., 1] / (1 + NET.bias[:, 1])
    loss = (q[:, :-1] - q[:, 1:]) / q[:, -1:]
    assert np.allclose(loss, NET.offtake, rtol=0.05)
    assert np.all(np.diff(true[..., 0] - NET.bias[:, 0], axis=1) < 0)  # pressure falls along the main


def test_leak_removes_flow_in_its_segment_only():
    normal, _ = _physical(NET, scenario(), T)
    leak, info = _physical(NET, scenario(fault="leak", segment=2, position_km=2.5, magnitude=0.1, onset=0), T)
    extra = (leak[..., 1][:, :-1] - leak[..., 1][:, 1:]) - (normal[..., 1][:, :-1] - normal[..., 1][:, 1:])
    assert extra[:, 2].mean() > 0.05 * Q0
    assert np.abs(extra[:, [0, 1, 3, 4]]).max() < 0.3 * extra[:, 2].mean()
    assert info["severity"][-1] > 0.3
    # leak noise is loudest at the devices closest to it
    assert set(np.argsort(leak[-1, :, 4] - normal[-1, :, 4])[-2:]) == {2, 3}


def test_blockage_adds_pressure_drop_in_its_segment():
    normal, _ = _physical(NET, scenario(), T)
    blk, _ = _physical(NET, scenario(fault="blockage", segment=1, position_km=1.5, magnitude=0.8, onset=0), T)
    drop = lambda x: x[..., 0][:, :-1] - x[..., 0][:, 1:]
    assert np.argmax((drop(blk) - drop(normal)).mean(0)) == 1


def test_contamination_only_affects_downstream_devices():
    normal, _ = _physical(NET, scenario(), T)
    con, _ = _physical(NET, scenario(fault="contamination", segment=3, position_km=3.4, magnitude=0.8, onset=0, ramp=1), T)
    dec = con[-1, :, 3] - normal[-1, :, 3]
    assert np.allclose(dec[:4], 0) and np.all(dec[4:] > 200)
