# Model card — Maeen PipeAI

## Model details
- **What:** an ensemble of five models over one physics-aware feature extractor (165 features): an IsolationForest anomaly detector, a 6-class fault classifier, a segment classifier with a km regressor, a device/instrument classifier, and a severity regressor. An evidence explainer reports why each alert was raised.
- **Algorithms:** scikit-learn `IsolationForest` and `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor`.
- **Version:** every trained model is registered under `models/registry/<version>/` with a `card.json` (config, git commit, class counts, feature hash, training time) and its evaluation `metrics.json`.

## Intended use
- Decision support for water-utility operators. It flags, diagnoses and locates leaks, blockages, water-quality contamination, corrosion and faulty sensors on an instrumented main, and suggests actions.
- A human stays in the loop. Recommendations that disrupt supply (isolating a segment, customer notices) should be confirmed on site, and the dashboard says so when confidence is moderate.

## Out of scope
- Branched networks, or lines with a different device spacing, without retraining.
- Any public-health decision based on the model alone. Contamination alerts must trigger lab sampling.
- Real-world claims of accuracy before field validation.

## Training data
- 9,000 labelled 30-minute windows from the physics simulator, plus 800 normal "commissioning" windows used to learn the baseline.
- Class mix: normal 30% · leak 25% · blockage 11% · contamination 11% · corrosion 11% · sensor fault 11%.
- Domain randomisation: sensor noise 0.6–2.2×, demand cycles, pump and consumer transients, fault size, location, start time and ramp speed.

## Evaluation
- Data held out by seed: 3,000 test windows, 1,500 windows at 2× noise, 200 streaming fault runs and 17 days of normal streaming.
- Headline results:

| Metric | Result |
|---|---|
| Fault-type accuracy | 94.1% (93.5% at 2× noise) |
| Correct segment | 99.9% |
| Position error | ~110 m |
| False alarms in streaming | 0.06 per day |
| Median detection delay | 5–11 min |

- The model beats the threshold-rule baseline (78.5% accuracy, 28.8% false alarms) and the raw-feature ML baseline (88.6% accuracy). Full details are in [reports/EVALUATION.md](reports/EVALUATION.md).

## Limitations
- **Simulation gap.** All training and test data is simulated. Real pipes have elevation, branching, unmetered demand, temperature effects and sensor behaviours the simulator does not model.
- **Sensor faults in their first minutes** are the weakest class (63% window-level recall). They are mostly caught within about 10 minutes as the fault grows.
- **Two faults at once** were not part of training. The model will report the dominant one.
- **Confidence scores** are model probabilities and are not yet calibrated against field outcomes.

## Monitoring and maintenance
- `Monitor.health()` tracks the anomaly rate and input deviation outside of alerts. A sustained rise means the data has drifted, and the baseline should be re-learned or the model retrained.
- Every change runs through the CI quality gate ([configs/quality_gate.json](configs/quality_gate.json)) before it can be merged.
