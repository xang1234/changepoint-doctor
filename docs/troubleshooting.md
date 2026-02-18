# Troubleshooting

Common errors and how to fix them.

## 1. `TypeError: expected float32 or float64`

**Cause:** Integer or object arrays passed into `.fit(...)` or `detect_offline(...)`.

**Fix:** Cast to float first:

```python
x = np.asarray(x, dtype=np.float64)
```

## 2. Input contains NaN/missing values and detection fails

**Cause:** Python APIs reject missing values under `MissingPolicy::Error` (the default).

**Fix:** Impute or drop NaNs before calling detectors:

```python
x = x[~np.isnan(x)]
# or
x = np.nan_to_num(x, nan=np.nanmean(x))
```

## 3. `RuntimeError: fit(...) must be called before predict(...)`

**Cause:** `.predict(...)` called on an unfitted high-level detector.

**Fix:** Always call `.fit(x)` first:

```python
result = cpd.Pelt(model="l2").fit(x).predict(n_bkps=2)
```

## 4. Extension import fails after Rust/Python upgrade

**Cause:** Wheel or extension built against a different interpreter environment.

**Fix:** Rebuild the extension in the active environment:

```bash
cd cpd/python
maturin develop --release --manifest-path ../crates/cpd-python/Cargo.toml
```

## 5. Apple Silicon linker mismatch (`arm64` vs `x86_64`)

**Cause:** Host shell/interpreter/libpython architectures do not match.

**Fix:** Follow the Apple Silicon guide in the {doc}`getting-started/installation` page to verify architecture alignment and recreate your virtual environment.

## 6. `ValueError` from `detect_offline(preprocess=...)`

**Cause:** Unknown preprocessing stage keys or invalid method/parameter combinations.

**Fix:** Check the [preprocessing reference](user-guide/preprocessing.md) for valid keys. The canonical stages are `detrend`, `deseasonalize`, `winsorize`, and `robust_scale`.

## 7. SegNeigh is too slow or runs out of memory

**Cause:** SegNeigh (exact DP) has O(k * m^2) time complexity, where m is the effective candidate count.

**Fix:**
- Increase `jump` or `min_segment_len` to reduce m
- Prefer PELT or FPOP for large n when k is unknown
- See the SegNeigh sizing guide in {doc}`user-guide/offline-algorithms`

## 8. `from_json()` rejects a payload with an unsupported schema version

**Cause:** The JSON payload has a `diagnostics.schema_version` outside the supported window (currently `1..=2`).

**Fix:** Ensure the payload was produced by a compatible version. Check `diagnostics.schema_version` in the JSON. See the [serialization contract](user-guide/serialization.md) for version compatibility details.
