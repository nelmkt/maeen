"""Decision support: turn an AI assessment into a headline and suggested actions (EN + AR)."""
from __future__ import annotations

FAULT_NAMES = {
    "normal": ("Normal", "طبيعي"),
    "leak": ("leak", "تسريب"),
    "blockage": ("blockage", "انسداد"),
    "contamination": ("water-quality contamination", "تلوث في جودة المياه"),
    "corrosion": ("corrosion", "تآكل"),
    "sensor_fault": ("sensor fault", "عطل في الحساس"),
    "watch": ("unclassified anomaly", "تغير غير طبيعي غير مصنف"),
}
SEVERITY_NAMES = {"low": ("Low", "منخفضة"), "medium": ("Medium", "متوسطة"),
                  "high": ("High", "عالية"), "critical": ("Critical", "حرجة")}
SENSOR_NAMES = {"pressure": ("pressure", "الضغط"), "flow": ("flow", "التدفق"),
                "ph": ("pH", "الأس الهيدروجيني pH"), "ec": ("EC", "الموصلية الكهربائية EC")}
PRIORITIES = {"immediate": ("Immediate", "فوري"), "24h": ("Within 24 h", "خلال 24 ساعة"),
              "scheduled": ("Scheduled", "مجدول"), "monitor": ("Monitor", "متابعة")}


def _r(priority: str, en: str, ar: str) -> dict:
    return {"priority": priority, "en": en, "ar": ar}


def headline(a: dict) -> dict:
    status, fault, loc = a["status"], a["fault"], a.get("location")
    if status == "normal":
        return {"en": "All segments operating normally", "ar": "جميع مقاطع الخط تعمل بشكل طبيعي"}
    if status == "watch":
        return {"en": "Unusual readings detected - under observation", "ar": "تم رصد قراءات غير معتادة - قيد المتابعة"}
    en, ar = FAULT_NAMES[fault]
    if loc and loc["kind"] == "segment":
        a_, b_ = loc["between"]
        return {"en": f"Possible {en} between Device {a_} and Device {b_}",
                "ar": f"احتمال {ar} بين الجهاز {a_} والجهاز {b_}"}
    s_en, s_ar = SENSOR_NAMES[loc["sensor"]]
    return {"en": f"Possible {s_en} sensor fault on Device {loc['device']}",
            "ar": f"احتمال عطل في حساس {s_ar} في الجهاز {loc['device']}"}


def recommend(a: dict) -> list[dict]:
    status, fault, loc = a["status"], a["fault"], a.get("location")
    level = a["severity"]["level"]
    urgent = level in ("high", "critical")
    if status == "normal":
        return [_r("monitor", "No action needed. The AI keeps re-assessing the line every minute.",
                   "لا يلزم أي إجراء. يعيد النظام تقييم الخط كل دقيقة.")]
    if status == "watch":
        return [
            _r("24h", "Readings deviate from the learned baseline but match no known fault. Take a manual reading to verify.",
               "القراءات تختلف عن النمط الطبيعي لكنها لا تطابق عطلاً معروفاً. يُنصح بأخذ قراءة يدوية للتحقق."),
            _r("monitor", "Keep monitoring; the alert escalates automatically if the pattern persists.",
               "الاستمرار في المراقبة، وسيتم رفع مستوى التنبيه تلقائياً إذا استمر النمط."),
        ]

    recs: list[dict] = []
    if fault == "sensor_fault":
        d = loc["device"]
        s_en, s_ar = SENSOR_NAMES[loc["sensor"]]
        recs += [
            _r("24h", f"Recalibrate or replace the {s_en} sensor on Device {d}.",
               f"معايرة أو استبدال حساس {s_ar} في الجهاز {d}."),
            _r("immediate", f"Device {d} {s_en} readings are flagged as low-trust; the AI relies on neighbouring devices meanwhile.",
               f"تم تعليم قراءات {s_ar} في الجهاز {d} كقراءات غير موثوقة، ويعتمد النظام على الأجهزة المجاورة مؤقتاً."),
            _r("scheduled", f"Check Device {d} power (UPS battery, micro-turbine) and wireless link during the visit.",
               f"فحص طاقة الجهاز {d} (بطارية UPS والتوربين) والاتصال اللاسلكي أثناء الزيارة."),
        ]
    else:
        a_, b_ = loc["between"]
        km = loc["position_km"]
        seg_en, seg_ar = f"segment D{a_}-D{b_}", f"المقطع بين الجهاز {a_} والجهاز {b_}"
        if fault == "leak":
            loss = a.get("estimated_loss_m3h") or 0.0
            if urgent:
                recs += [
                    _r("immediate", f"Isolate {seg_en}: close the isolation valves next to Device {a_} and Device {b_}.",
                       f"عزل {seg_ar} بإغلاق صمامات العزل القريبة من الجهازين."),
                    _r("immediate", f"Dispatch a repair crew with an acoustic correlator to km {km:.1f} (±0.3 km).",
                       f"إرسال فريق صيانة مع جهاز الترابط الصوتي إلى الكيلومتر {km:.1f} (±0.3 كم)."),
                    _r("immediate", "Lower inlet pressure by 10-15% to limit water loss until the repair is done.",
                       "خفض ضغط الدخول بنسبة 10-15٪ لتقليل الفاقد حتى يتم الإصلاح."),
                    _r("24h", f"Notify customers downstream of Device {b_} about a possible supply interruption.",
                       f"إبلاغ المشتركين بعد الجهاز {b_} باحتمال انقطاع مؤقت في الإمداد."),
                ]
            elif level == "medium":
                recs += [
                    _r("24h", f"Schedule a leak-detection survey of {seg_en} around km {km:.1f} within 24 h.",
                       f"جدولة فحص كشف تسرب في {seg_ar} عند الكيلومتر {km:.1f} تقريباً خلال 24 ساعة."),
                    _r("monitor", f"Raise the sampling rate of Devices {a_} and {b_} to track whether the leak grows.",
                       f"رفع معدل القراءات للجهازين {a_} و{b_} لمتابعة تطور التسرب."),
                ]
            else:
                recs += [
                    _r("scheduled", f"Add {seg_en} (around km {km:.1f}) to the next routine inspection (≤ 7 days).",
                       f"إضافة {seg_ar} (عند الكيلومتر {km:.1f}) إلى جدول الفحص الدوري القادم (خلال 7 أيام)."),
                    _r("monitor", "Keep monitoring; escalate automatically if the loss exceeds 5 m³/h.",
                       "الاستمرار في المراقبة ورفع التنبيه تلقائياً إذا تجاوز الفاقد 5 م³/ساعة."),
                ]
            if loss > 0:
                recs.append(_r("monitor", f"Estimated water loss ≈ {loss:.1f} m³/h (≈ {loss * 24:.0f} m³/day).",
                               f"الفاقد المقدّر ≈ {loss:.1f} م³/ساعة (≈ {loss * 24:.0f} م³/يوم)."))
        elif fault == "blockage":
            recs += [
                _r("immediate" if urgent else "24h",
                   f"Check valve positions in {seg_en} - a partly closed valve is the most common cause.",
                   f"التحقق من وضع الصمامات في {seg_ar}، فالصمام المغلق جزئياً هو السبب الأكثر شيوعاً."),
                _r("24h" if urgent else "scheduled",
                   f"Inspect near km {km:.1f} for debris or sediment and plan flushing / pigging of the segment.",
                   f"فحص المنطقة القريبة من الكيلومتر {km:.1f} بحثاً عن رواسب أو عوائق والتخطيط لغسيل المقطع."),
            ]
            if urgent:
                recs.append(_r("immediate", f"Verify supply pressure for customers downstream of Device {b_}.",
                               f"التحقق من ضغط الإمداد للمشتركين بعد الجهاز {b_}."))
        elif fault == "contamination":
            recs += [
                _r("immediate", f"Collect water samples at Device {b_} and downstream for lab analysis (bacteria, turbidity, chlorine).",
                   f"أخذ عينات مياه عند الجهاز {b_} وما بعده لتحليلها مخبرياً (بكتيريا، عكارة، كلور)."),
                _r("immediate", f"Isolate and flush {seg_en}; check for back-flow or cross-connections near km {km:.1f}.",
                   f"عزل وغسيل {seg_ar} والتحقق من وجود تدفق عكسي أو توصيلات غير نظامية قرب الكيلومتر {km:.1f}."),
            ]
            if urgent:
                recs.append(_r("immediate", f"Issue a precautionary water-quality notice to customers after Device {a_}.",
                               f"إصدار تنبيه احترازي حول جودة المياه للمشتركين بعد الجهاز {a_}."))
            recs.append(_r("24h", "Review disinfectant dosing at the source.", "مراجعة جرعات التعقيم عند المصدر."))
        elif fault == "corrosion":
            recs += [
                _r("scheduled", f"Schedule an ultrasonic wall-thickness inspection of {seg_en} within 2 weeks.",
                   f"جدولة فحص سماكة جدار الأنبوب بالموجات فوق الصوتية في {seg_ar} خلال أسبوعين."),
                _r("24h", "Review water chemistry: low pH accelerates corrosion - adjust pH correction (target 7.2-8.0).",
                   "مراجعة كيمياء المياه: انخفاض pH يسرّع التآكل - ضبط جرعات تعديل pH (الهدف 7.2-8.0)."),
            ]
            if urgent:
                recs.append(_r("scheduled", f"Add {seg_en} to the rehabilitation / replacement plan.",
                               f"إضافة {seg_ar} إلى خطة التأهيل أو الاستبدال."))
            recs.append(_r("monitor", "Track the EC and pH trend for this segment monthly.",
                           "متابعة اتجاه EC وpH لهذا المقطع شهرياً."))

    if a["confidence"] < 0.5:
        recs.append(_r("monitor", "Confidence is moderate - confirm on site before any disruptive action.",
                       "مستوى الثقة متوسط - يُنصح بالتأكد ميدانياً قبل اتخاذ أي إجراء يؤثر على الخدمة."))
    return recs
