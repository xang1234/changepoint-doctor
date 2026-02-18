# Serialization

## Result JSON contract

```{include} ../../cpd/docs/result_json_contract.md
```

## Online detector checkpoints

All online detectors (`Bocpd`, `Cusum`, `PageHinkley`) support checkpoint/restore for fault tolerance and state migration.

### Save state

```python
# Save to bytes (compact binary, default)
state_bytes = bocpd.save_state(format="bytes")

# Save to dict (inspectable Python dict)
state_dict = bocpd.save_state(format="dict")

# Save to file
bocpd.save_state(format="bytes", path="/tmp/bocpd_checkpoint.bin")
```

### Load state

```python
# Load from bytes
bocpd.load_state(state_bytes, format="bytes")

# Load from dict
bocpd.load_state(state_dict, format="dict")

# Load from file
bocpd.load_state(path="/tmp/bocpd_checkpoint.bin", format="bytes")
```

### Cross-session workflow

```python
import cpd

# Session 1: process first batch and checkpoint
bocpd = cpd.Bocpd(model="gaussian_nig", hazard=1/200, max_run_length=512)
steps1 = bocpd.update_many(batch1)
bocpd.save_state(format="bytes", path="/tmp/checkpoint.bin")

# Session 2: restore and continue
bocpd2 = cpd.Bocpd(model="gaussian_nig", hazard=1/200, max_run_length=512)
bocpd2.load_state(path="/tmp/checkpoint.bin", format="bytes")
steps2 = bocpd2.update_many(batch2)  # continues from where session 1 left off
```
