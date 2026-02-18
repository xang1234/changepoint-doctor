# API Reference

Complete Python API documentation for `changepoint-doctor`.

```{toctree}
:maxdepth: 2

offline-detectors
online-detectors
result-types
diagnostics
functions
```

## Quick import reference

```python
import cpd

# Offline detectors
cpd.Pelt, cpd.Binseg, cpd.Fpop

# Online detectors
cpd.Bocpd, cpd.Cusum, cpd.PageHinkley

# Result types
cpd.OfflineChangePointResult, cpd.OnlineStepResult

# Diagnostics
cpd.Diagnostics, cpd.PruningStats, cpd.SegmentStats

# Functions
cpd.detect_offline, cpd.smoke_detect
```
