# Installation

## Install from PyPI

```bash
python -m pip install --upgrade pip
python -m pip install changepoint-doctor
```

Verify the installation:

```bash
python -c "import cpd; print(cpd.__version__)"
```

> Install/import naming: install with `python -m pip install changepoint-doctor`, then import with `import cpd` in Python. Optional compatibility alias: `import changepoint_doctor as cpd`.

## Build from source

Building from source requires a Rust toolchain (edition 2024, MSRV 1.93) and [maturin](https://www.maturin.rs/).

```bash
# Clone the repository
git clone https://github.com/xang1234/changepoint-doctor.git
cd changepoint-doctor/cpd/python

# Install build dependencies
python -m pip install --upgrade pip maturin
python -m pip install --upgrade ".[dev]"

# Build and install the extension in development mode
maturin develop --release --manifest-path ../crates/cpd-python/Cargo.toml
```

Verify:

```bash
python -c "import cpd; print(cpd.__version__)"
```

## Python extras vs Rust feature flags

Python extras install optional Python tooling only:

- `python -m pip install "changepoint-doctor[plot]"`
- `python -m pip install "changepoint-doctor[notebooks]"`
- `python -m pip install "changepoint-doctor[parity]"`
- `python -m pip install "changepoint-doctor[dev]"`

Rust feature flags are configured when building the extension/workspace (for
example `maturin develop --features preprocess,serde ...` or `cargo ... --features ...`).

## Apple Silicon notes

On Apple Silicon (M1/M2/M3), use a native `arm64` shell and Python interpreter. Do not mix `arm64` Rust toolchain with `x86_64` Python binaries.

**Verify architecture alignment before building:**

```bash
# Host architecture (expected: arm64)
uname -m

# Python interpreter architecture
python -c "import platform; print(platform.machine())"
```

If the architectures do not match, recreate your virtual environment with the correct interpreter:

```bash
cd cpd/python
rm -rf .venv
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
python -m pip install --upgrade ".[dev]"
```

**Common mismatch symptoms:**
- `ld: warning: ignoring file ... built for macOS-x86_64`
- `ld: symbol(s) not found for architecture arm64`
- `pyo3-build-config` selecting the wrong interpreter architecture

## Feature flags

The Rust workspace supports feature flags that enable optional capabilities:

| Feature | Description |
|---|---|
| `rayon` | Parallel execution for supported algorithms |
| `serde` | JSON serialization for results and pipeline configs |
| `tracing` | Structured logging via the `tracing` crate |
| `simd` | SIMD-accelerated numeric kernels |
| `kernel` | Kernel-based change-point detection (experimental) |
| `kernel-approx` | Approximate kernel methods |
| `blas` | BLAS-accelerated linear algebra |
| `gp` | Gaussian process change-point detection (experimental) |
| `preprocess` | Signal preprocessing (detrend, deseasonalize, winsorize, scale) |
| `repro-strict` | Strict reproducibility mode with deterministic numeric paths |

Default PyPI wheels are BLAS-free. The `serde` and `preprocess` features are enabled by default in the Python extension.

## Wheel platform matrix

| Platform | Python versions | NumPy |
|---|---|---|
| Linux manylinux x86_64 | 3.9, 3.10, 3.11, 3.12, 3.13 | 1.26.x, 2.x |
| macOS universal2 | 3.9, 3.10, 3.11, 3.12, 3.13 | 1.26.x, 2.x |
| Windows amd64 | 3.9, 3.10, 3.11, 3.12, 3.13 | 1.26.x, 2.x |

Python 3.13 + NumPy 1.26.x is excluded. Python 3.13 rows are currently marked experimental.

## Dependencies

The Python package requires:
- Python >= 3.9
- NumPy >= 1.20

Optional dependency groups:
- `plot` for result plotting (`result.plot(...)`)
- `notebooks` for Jupyter notebook workflows
- `parity` for parity/test workflows
- `dev` for local contributor workflows
