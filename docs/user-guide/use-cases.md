# Use Cases

`changepoint-doctor` is designed for real-world time series analysis. Here are common scenarios and how to approach them.

## Industrial sensor monitoring (mean-shift detection)

A factory sensor measures temperature or vibration. You need to detect when the process shifts to a new operating regime.

**Approach:** Use PELT with L2 cost for fast mean-shift detection on the full batch.

```python
import cpd

# sensor_data: np.ndarray of float64
result = cpd.Pelt(model="l2", min_segment_len=30).fit(sensor_data).predict(pen="bic")
print("Regime changes at:", result.change_points)
```

**Why PELT?** It runs in O(n) with pruning, handles unknown numbers of change points via penalty-based stopping, and L2 cost is robust for mean shifts in continuous-valued sensors.

## Financial regime detection (variance change)

Market returns exhibit regime changes where volatility shifts. Detecting these transitions is critical for risk management.

**Approach:** Use PELT with Normal cost (models both mean and variance) or NormalFullCov for multivariate portfolios.

```python
result = cpd.Pelt(model="normal", min_segment_len=20).fit(returns).predict(pen="bic")

# For multivariate portfolio returns
result = cpd.detect_offline(
    portfolio_returns,
    detector="pelt",
    cost="normal_full_cov",
    constraints={"min_segment_len": 20},
    stopping={"pen": "bic"},
)
```

## Network traffic anomaly detection (streaming BOCPD)

Monitor network traffic in real time and alert on distributional changes.

**Approach:** Use BOCPD with Gaussian NIG model for continuous metrics, tuning the hazard rate and alert policy.

```python
import cpd

bocpd = cpd.Bocpd(
    model="gaussian_nig",
    hazard=1.0 / 500.0,  # expect a change roughly every 500 observations
    max_run_length=1024,
    alert_policy={"threshold": 0.5, "cooldown": 10, "min_run_length": 20},
)

for packet_rate in live_stream():
    step = bocpd.update(packet_rate)
    if step.alert:
        trigger_investigation(step.t, step.p_change)
```

## A/B testing (detecting treatment effect onset)

Detect when an A/B test treatment effect appears in a metric stream, using online detection with checkpoint/restore for fault tolerance.

```python
import cpd

cusum = cpd.Cusum(
    drift=0.5,
    threshold=8.0,
    target_mean=baseline_mean,
    alert_policy={"threshold": 0.5, "cooldown": 5},
)

for metric_value in experiment_stream():
    step = cusum.update(metric_value)
    if step.alert:
        print(f"Treatment effect detected at t={step.t}")
        break

# Save state for fault tolerance
state = cusum.save_state(format="bytes")
```

## Climate data (seasonal + trend)

Climate data often has strong seasonal patterns and long-term trends. Preprocessing removes these before detection.

**Approach:** Use `detect_offline()` with preprocessing to detrend and deseasonalize before running PELT.

```python
result = cpd.detect_offline(
    temperature_data,
    detector="pelt",
    cost="l2",
    constraints={"min_segment_len": 30},
    stopping={"pen": "bic"},
    preprocess={
        "detrend": {"method": "linear"},
        "deseasonalize": {"method": "stl_like", "period": 12},
        "robust_scale": {},
    },
)
```

See {doc}`preprocessing` for the full preprocessing pipeline documentation.
