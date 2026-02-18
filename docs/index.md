# changepoint-doctor

A Rust-first change-point detection toolkit with Python bindings and a recommendation engine.

`changepoint-doctor` provides fast offline (batch) segmentation and online (streaming) change detection, backed by a Rust core for performance and a Python API for ease of use. The **doctor** recommendation engine helps choose good detector pipelines for your data.

```bash
pip install changepoint-doctor
```

```python
import numpy as np
import cpd

x = np.concatenate([np.zeros(50), np.full(50, 5.0), np.full(50, -2.0)])
result = cpd.Pelt(model="l2").fit(x).predict(n_bkps=2)
print(result.breakpoints)  # [50, 100, 150]
```

---

::::{grid} 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting-started/index
:link-type: doc

Install from PyPI or build from source. Run your first change-point detection in 5 minutes.
:::

:::{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc

Algorithms, cost models, the doctor engine, preprocessing, and real-world use cases.
:::

:::{grid-item-card} API Reference
:link: api-reference/index
:link-type: doc

Complete Python API: detectors, result types, diagnostics, and the `detect_offline()` function.
:::

:::{grid-item-card} Architecture
:link: architecture/index
:link-type: doc

Rust crate hierarchy, core traits, feature flags, and Rust-side usage examples.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

getting-started/index
user-guide/index
api-reference/index
architecture/index
notebooks/01_offline_algorithms
notebooks/02_online_algorithms
notebooks/03_doctor_recommendations
troubleshooting
changelog
```

## Bibliography

```{bibliography}
```
