# Reliable Model Evaluation Platform

An initial, model-agnostic toolkit for classification reliability. It reports accuracy, calibration error, Brier score, conformal coverage, prediction-set size, and singleton/empty-set rates. It is designed as the foundation for an enterprise evaluation workflow, with separate calibration and test roles.

## Quick start

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src pytest -q
```

```python
from reliable_eval import build_report, render_markdown
report = build_report(test_probabilities, test_labels, calibration_probabilities=cal_probabilities, calibration_labels=cal_labels, alpha=0.1)
print(render_markdown(report))
```

The report distinguishes empirical test coverage from the nominal target and does not claim conditional or OOD guarantees. Roadmap: subgroup and worst-group reports, OOD/corruption stress tests, bootstrap uncertainty, delayed-label drift monitoring, CI quality gates, and medical-image conformal triage.

## License
MIT
