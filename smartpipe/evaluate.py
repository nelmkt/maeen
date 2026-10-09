"""Measure how well the AI detects, diagnoses and locates faults:  python -m smartpipe.evaluate

All test data is generated with seeds never used in training. Four test suites:
  1. window test     3000 random 30-min windows (all fault types, random size/location/onset)
  2. noisy test      1500 windows with 2x sensor noise (robustness)
  3. leak-size sweep detection & localisation vs leak size
  4. streaming test  full 3-hour runs fed minute by minute through the incident logic:
                     detection delay, false alarms per day, diagnosis at the moment of alert
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import FAULT_TYPES, PIPE_FAULTS, WINDOW, severity_level
from .data import make_dataset
from .models import DEFAULT_MODEL_PATH, load_or_train
from .monitor import IncidentTracker
from .simulator import Network, random_scenario, simulate

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def _summarise(preds: list[dict], y: pd.DataFrame) -> dict:
    pred_fault = np.array([p["fault"] if p["status"] == "alert" else "normal" for p in preds])
    true = y["fault"].to_numpy()
    is_fault = true != "normal"
    flagged = np.array([p["status"] != "normal" for p in preds])
    res = {
        "n": int(len(y)),
        "fault_type_accuracy": accuracy_score(true, pred_fault),
        "fault_type_macro_f1": f1_score(true, pred_fault, average="macro"),
        "detection_rate": float(flagged[is_fault].mean()),
        "false_alarm_rate": float(flagged[~is_fault].mean()),
        "isolation_forest_detection_rate": float(np.mean([p["anomaly"] for p, f in zip(preds, is_fault) if f])),
        "isolation_forest_false_alarm_rate": float(np.mean([p["anomaly"] for p, f in zip(preds, is_fault) if not f])),
    }
    pipe = np.isin(true, PIPE_FAULTS) & (pred_fault == true)
    seg_pred = np.array([p["location"]["segment"] - 1 if p["location"] and p["location"]["kind"] == "segment" else -9 for p in preds])
    pos_pred = np.array([p["location"]["position_km"] if p["location"] and p["location"]["kind"] == "segment" else np.nan for p in preds])
    seg_true = y["segment"].to_numpy()
    res["segment_accuracy"] = float((seg_pred[pipe] == seg_true[pipe]).mean())
    res["segment_within_1"] = float((np.abs(seg_pred[pipe] - seg_true[pipe]) <= 1).mean())
    res["position_mae_km"] = float(np.nanmean(np.abs(pos_pred[pipe] - y["position_km"].to_numpy()[pipe])))
    res["segment_accuracy_by_fault"] = {
        f: float((seg_pred[pipe & (true == f)] == seg_true[pipe & (true == f)]).mean()) for f in PIPE_FAULTS
    }
    sf = (true == "sensor_fault") & (pred_fault == true)
    dev_pred = np.array([p["location"]["device"] - 1 if p["location"] and p["location"]["kind"] == "device" else -9 for p in preds])
    sen_pred = np.array([p["location"]["sensor"] if p["location"] and p["location"]["kind"] == "device" else "" for p in preds])
    from .config import FAULTY_SENSORS
    sen_true = np.array([FAULTY_SENSORS[s] if s >= 0 else "" for s in y["sensor"]])
    res["sensor_fault_device_accuracy"] = float((dev_pred[sf] == y["device"].to_numpy()[sf]).mean())
    res["sensor_fault_instrument_accuracy"] = float((sen_pred[sf] == sen_true[sf]).mean())
    ok = is_fault & (pred_fault == true)
    sev_pred = np.array([p["severity"]["score"] for p in preds])
    res["severity_mae"] = float(np.abs(sev_pred[ok] - y["severity"].to_numpy()[ok]).mean())
    res["severity_level_accuracy"] = float(np.mean([severity_level(a) == severity_level(b)
                                                    for a, b in zip(sev_pred[ok], y["severity"].to_numpy()[ok])]))
    res["per_class"] = classification_report(true, pred_fault, labels=list(FAULT_TYPES), output_dict=True, zero_division=0)
    res["confusion_matrix"] = confusion_matrix(true, pred_fault, labels=list(FAULT_TYPES)).tolist()
    return res


def leak_size_sweep(preds: list[dict], y: pd.DataFrame) -> list[dict]:
    leak = (y["fault"] == "leak").to_numpy()
    ratio = y["leak_ratio"].to_numpy() * 100
    bins = [(0, 2), (2, 5), (5, 10), (10, 100)]
    rows = []
    for lo, hi in bins:
        m = leak & (ratio >= lo) & (ratio < hi)
        if not m.any():
            continue
        p = [preds[i] for i in np.flatnonzero(m)]
        det = np.array([q["status"] != "normal" for q in p])
        diag = np.array([q["fault"] == "leak" for q in p])
        seg = np.array([q["location"]["segment"] - 1 if q["location"] and q["location"]["kind"] == "segment" else -9 for q in p])
        rows.append({"leak_pct_of_flow": f"{lo}–{hi}%", "n": int(m.sum()), "detected": float(det.mean()),
                     "diagnosed_as_leak": float(diag.mean()),
                     "segment_correct": float((seg[diag] == y["segment"].to_numpy()[m][diag]).mean()) if diag.any() else None})
    return rows


def streaming_test(ai, net, runs_per_fault: int = 40, normal_runs: int = 80, T: int = 180, onset_at: int = 90, seed: int = 303):
    rng = np.random.default_rng(seed)
    out = []
    for fault in FAULT_TYPES:
        n = normal_runs if fault == "normal" else runs_per_fault
        for _ in range(n):
            t0 = rng.uniform(0, 3 * 1440)
            scn = random_scenario(rng, fault, t0, T, onset=t0 + onset_at if fault != "normal" else None)
            X, _ = simulate(net, scn, t0 + np.arange(T), rng)
            windows = sliding_window_view(X, WINDOW, axis=0).transpose(0, 3, 1, 2)  # (T-W+1, W, dev, sens)
            preds = ai.predict(windows)
            tracker = IncidentTracker()
            first = None
            false_alarms = 0
            for k, a in enumerate(preds):
                minute = k + WINDOW - 1
                was_active = tracker.active
                tracker.update(a, minute)
                opened = tracker.active is not None and tracker.active is not was_active
                if opened and (fault == "normal" or minute < onset_at):
                    false_alarms += 1
                elif opened and first is None and minute >= onset_at:
                    first = (minute, a)
            row = {"fault": fault, "false_alarms": false_alarms, "minutes_observed": T - WINDOW + 1 if fault == "normal" else onset_at - WINDOW + 1}
            if fault != "normal":
                row["detected"] = first is not None
                if first:
                    minute, a = first
                    row["delay_min"] = minute - onset_at
                    row["type_correct"] = a["fault"] == fault
                    loc = a["location"] or {}
                    if fault == "sensor_fault":
                        row["location_correct"] = loc.get("device") == scn.device + 1
                    else:
                        row["location_correct"] = loc.get("segment") == scn.segment + 1
            out.append(row)
    df = pd.DataFrame(out)
    faults = df[df.fault != "normal"]
    per_fault = []
    for f, g in faults.groupby("fault", sort=False):
        d = g[g.detected == True]  # noqa: E712
        per_fault.append({
            "fault": f, "runs": int(len(g)), "detected": float(g.detected.mean()),
            "median_delay_min": float(d.delay_min.median()) if len(d) else None,
            "type_correct_at_alert": float(d.type_correct.mean()) if len(d) else None,
            "location_correct_at_alert": float(d.location_correct.mean()) if len(d) else None,
        })
    minutes = df.minutes_observed.sum()
    return {"per_fault": per_fault, "false_alarms_per_day": float(df.false_alarms.sum() / minutes * 1440),
            "normal_minutes_observed": int(minutes)}


def _plots(res: dict, sweep: list[dict], out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = np.array(res["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(FAULT_TYPES)), FAULT_TYPES, rotation=35, ha="right")
    ax.set_yticks(range(len(FAULT_TYPES)), FAULT_TYPES)
    for i in range(len(FAULT_TYPES)):
        for j in range(len(FAULT_TYPES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.6 else "#222")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Fault diagnosis — confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [r["leak_pct_of_flow"] for r in sweep]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [r["detected"] for r in sweep], 0.4, label="detected", color="#2a6fdb")
    ax.bar(x + 0.2, [r["segment_correct"] or 0 for r in sweep], 0.4, label="correct segment", color="#18a589")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("leak size (% of pipeline flow)")
    ax.set_title("Leak detection and localisation vs leak size")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "leak_size_sweep.png", dpi=140)
    plt.close(fig)


def _pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def write_markdown(m: dict, path: Path):
    s, n, st = m["window_test"], m["noisy_test"], m["streaming_test"]
    L = ["# Evaluation report", "",
         f"Generated by `python -m smartpipe.evaluate` on simulated data never seen in training "
         f"({s['n']} test windows, {n['n']} noisy windows, {sum(r['runs'] for r in st['per_fault'])} streaming fault runs, "
         f"{st['normal_minutes_observed'] / 1440:.1f} days of normal streaming).", "",
         "## Headline numbers", "",
         "| Metric | Test set | 2× sensor noise |", "|---|---|---|"]
    for key, label in [("fault_type_accuracy", "Fault type accuracy (6 classes)"),
                       ("fault_type_macro_f1", "Fault type macro-F1"),
                       ("detection_rate", "Detection rate (any fault flagged)"),
                       ("false_alarm_rate", "False alarm rate (normal windows flagged)"),
                       ("segment_accuracy", "Correct segment (pipe faults)"),
                       ("segment_within_1", "Segment within ±1"),
                       ("sensor_fault_device_accuracy", "Correct device (sensor faults)"),
                       ("severity_level_accuracy", "Correct severity level")]:
        L.append(f"| {label} | {_pct(s[key])} | {_pct(n[key])} |")
    L.append(f"| Leak/fault position error (MAE) | {s['position_mae_km'] * 1000:.0f} m | {n['position_mae_km'] * 1000:.0f} m |")
    L += ["", "*Training mixes sensor quality from 0.6× to 2.2× the nominal noise; the test set uses 1×, "
          "the robustness column uses 2× (low-cost sensors, poor installation).*"]
    L += ["", "Segment accuracy by fault: " + ", ".join(f"{k} {_pct(v)}" for k, v in s["segment_accuracy_by_fault"].items()), "",
          "## Per-class results (test set)", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for f in FAULT_TYPES:
        r = s["per_class"][f]
        L.append(f"| {f} | {_pct(r['precision'])} | {_pct(r['recall'])} | {_pct(r['f1-score'])} | {int(r['support'])} |")
    L += ["", "![confusion matrix](confusion_matrix.png)", "",
          "## Leak size sweep", "", "| Leak size | Windows | Detected | Diagnosed as leak | Correct segment |", "|---|---|---|---|---|"]
    for r in m["leak_size_sweep"]:
        L.append(f"| {r['leak_pct_of_flow']} | {r['n']} | {_pct(r['detected'])} | {_pct(r['diagnosed_as_leak'])} | {_pct(r['segment_correct'])} |")
    L += ["", "![leak size sweep](leak_size_sweep.png)", "",
          "## Streaming test (minute-by-minute, alert after 3 consistent minutes)", "",
          f"False alarms: **{st['false_alarms_per_day']:.2f} per day** of normal operation.", "",
          "| Fault | Runs | Detected | Median delay | Right type at alert | Right location at alert |", "|---|---|---|---|---|---|"]
    for r in st["per_fault"]:
        delay = "—" if r["median_delay_min"] is None else f"{r['median_delay_min']:.0f} min"
        L.append(f"| {r['fault']} | {r['runs']} | {_pct(r['detected'])} | {delay} | {_pct(r['type_correct_at_alert'])} | {_pct(r['location_correct_at_alert'])} |")
    L += ["", "> All results are on **simulated** data. They show the method works and where it struggles "
          "(very small leaks, slow corrosion); real-world accuracy has to be re-measured once field data is available.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--n-test", type=int, default=3000)
    args = ap.parse_args()
    t = time.time()
    ai = load_or_train(Path(args.model))
    net = Network.default()

    X, y = make_dataset(args.n_test, seed=101, net=net)
    preds = ai.predict(X)
    Xn, yn = make_dataset(args.n_test // 2, seed=202, net=net, noise_scale=2.0)
    metrics = {
        "window_test": _summarise(preds, y),
        "noisy_test": _summarise(ai.predict(Xn), yn),
        "leak_size_sweep": leak_size_sweep(preds, y),
        "streaming_test": streaming_test(ai, net),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    _plots(metrics["window_test"], metrics["leak_size_sweep"], REPORTS)
    write_markdown(metrics, REPORTS / "EVALUATION.md")
    print((REPORTS / "EVALUATION.md").read_text(encoding="utf-8"))
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
