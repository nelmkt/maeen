"""Per-prediction evidence: which measured signals deviate most from normal operation.

Every feature is converted to a robust z-score against the distribution seen during
normal operation (median / IQR of the training "normal" windows). The strongest
deviations *at the predicted location* are turned into readable sentences, so the
operator sees why the model raised the alert ("flow lost between D3 and D4: +6.2% of
inflow, acoustic at D4: +7 dB"), not just a probability.
"""
from __future__ import annotations

import re

import numpy as np

from .config import SENSORS
from .recommend import SENSOR_NAMES

_SEN_AR = {**{k: v[1] for k, v in SENSOR_NAMES.items()}, "acoustic": "الصوت", "vibration": "الاهتزاز"}
_SEN_EN = {**{k: v[0] for k, v in SENSOR_NAMES.items()}, "acoustic": "acoustic", "vibration": "vibration"}
_TAG = re.compile(r"^(?P<base>[a-z_]+)(?:\[(?P<tag>[^\]]+)\])?$")


class Explainer:
    def fit(self, F_normal: np.ndarray, names: list[str], friction_base: np.ndarray) -> "Explainer":
        self.names = names
        self.friction_base = np.asarray(friction_base)
        self.median = np.median(F_normal, axis=0)
        q75, q25 = np.percentile(F_normal, [75, 25], axis=0)
        self.scale = np.maximum((q75 - q25) / 1.349, 1e-6)
        return self

    def zscores(self, F: np.ndarray) -> np.ndarray:
        return (F - self.median) / self.scale

    def evidence(self, f: np.ndarray, z: np.ndarray, location: dict | None, k: int = 4) -> list[dict]:
        idx = np.arange(len(self.names))
        if location:
            keep = self._relevant(location)
            local = np.array([i for i in idx if keep(self.names[i])])
            idx = local if len(local) else idx
        order = idx[np.argsort(-np.abs(z[idx]))]
        out, seen = [], set()
        for i in order:
            if abs(z[i]) < 3 or len(out) >= k:
                break
            m = _TAG.match(self.names[i])
            key = (m["base"].removeprefix("change_").removesuffix("_rel"), m["tag"]) if m else self.names[i]
            if key in seen:  # one item per signal and place, keep the evidence diverse
                continue
            seen.add(key)
            text = self._describe(self.names[i], f[i] - self.median[i])
            out.append({"feature": self.names[i], "z": float(z[i]), **text})
        return out

    @staticmethod
    def _relevant(loc: dict):
        if loc["kind"] == "segment":
            a, b = loc["between"]
            tags = {f"S{loc['segment']}", f"D{a}", f"D{b}"}
            return lambda n: any(f"[{t}" in n for t in tags)
        d = f"D{loc['device']}"
        return lambda n: f"[{d}" in n

    def _describe(self, name: str, v: float) -> dict:
        m = _TAG.match(name)
        base, tag = (m["base"], m["tag"]) if m else (name, "")
        changed = base.startswith("change_")
        base = base.removeprefix("change_")
        when_en = " over the last 30 min" if changed else " vs normal"
        when_ar = " خلال آخر 30 دقيقة" if changed else " مقارنة بالوضع الطبيعي"
        seg = lambda t: tuple(int(t[1:]) + d for d in (0, 1))  # "S3" -> (3, 4)
        if base == "flow_loss":
            a, b = seg(tag)
            return {"en": f"Flow lost between D{a} and D{b}: {v * 100:+.1f}% of inflow{when_en}",
                    "ar": f"فقد في التدفق بين الجهاز {a} والجهاز {b}: {v * 100:+.1f}٪ من تدفق الدخول{when_ar}"}
        if base == "friction":
            a, b = seg(tag)
            pct = 100 * v / self.friction_base[a - 1]
            return {"en": f"Pressure drop in D{a}-D{b} for the flow carried: {pct:+.0f}%{when_en}",
                    "ar": f"هبوط الضغط في المقطع {a}-{b} نسبة للتدفق: {pct:+.0f}٪{when_ar}"}
        if base == "dec":
            a, b = seg(tag)
            return {"en": f"EC rise from D{a} to D{b}: {v:+.0f} µS/cm{when_en}",
                    "ar": f"تغير الموصلية EC من الجهاز {a} إلى {b}: {v:+.0f} µS/cm{when_ar}"}
        if base == "dph":
            a, b = seg(tag)
            return {"en": f"pH change from D{a} to D{b}: {v:+.2f}{when_en}",
                    "ar": f"تغير pH من الجهاز {a} إلى {b}: {v:+.2f}{when_ar}"}
        if base in ("acoustic", "acoustic_rel"):
            return {"en": f"Acoustic level at {tag}: {v:+.1f} dB{when_en}",
                    "ar": f"مستوى الصوت عند الجهاز {tag[1:]}: {v:+.1f} dB{when_ar}"}
        if base == "vibration":
            return {"en": f"Vibration at {tag}: {v:+.2f} mm/s{when_en}",
                    "ar": f"الاهتزاز عند الجهاز {tag[1:]}: {v:+.2f} mm/s{when_ar}"}
        if base == "p_resid":
            return {"en": f"Pressure at {tag} inconsistent with neighbours: {v:+.2f} bar{when_en}",
                    "ar": f"ضغط الجهاز {tag[1:]} لا يتوافق مع الأجهزة المجاورة: {v:+.2f} bar{when_ar}"}
        if base in ("rel_noise", "jump"):
            dev, sensor = tag.split(".")
            en_s, ar_s = _SEN_EN[sensor], _SEN_AR[sensor]
            if base == "jump":
                return {"en": f"{en_s} at {dev} shifted by {v:+.1f}σ within 30 min",
                        "ar": f"قراءة {ar_s} في الجهاز {dev[1:]} تغيرت بمقدار {v:+.1f}σ خلال 30 دقيقة"}
            if v < -2:
                return {"en": f"{en_s} at {dev} is frozen (no natural fluctuation)",
                        "ar": f"قراءة {ar_s} في الجهاز {dev[1:]} ثابتة بشكل غير طبيعي (متجمدة)"}
            return {"en": f"{en_s} at {dev} is {np.exp(v):.1f}× noisier than its neighbours",
                    "ar": f"قراءة {ar_s} في الجهاز {dev[1:]} أكثر تذبذباً بـ {np.exp(v):.1f} مرة من الأجهزة المجاورة"}
        if base.startswith("slope_"):
            sensor = base.removeprefix("slope_")
            return {"en": f"{_SEN_EN[sensor]} trend at {tag}: {v * 30:+.3g} per 30 min",
                    "ar": f"اتجاه {_SEN_AR[sensor]} عند الجهاز {tag[1:]}: {v * 30:+.3g} لكل 30 دقيقة"}
        return {"en": f"{name}: {v:+.3g}{when_en}", "ar": f"{name}: {v:+.3g}{when_ar}"}


assert set(_SEN_EN) == set(SENSORS)
