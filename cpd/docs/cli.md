# CLI Guide

`cpd` is the Rust CLI for offline detection, pipeline execution, doctor recommendations,
and evaluation workflows.

## Install

Build from repository root:

```bash
cd cpd
cargo build -p cpd-cli --release
./target/release/cpd --help
```

Run without an explicit build step:

```bash
cd cpd
cargo run -p cpd-cli -- --help
```

## Input Formats

`cpd` accepts `.csv` and `.npy` signal inputs for `detect`, `run`, and `doctor`.

- `.csv`: numeric data, 1D or rectangular 2D. A single header row is tolerated.
- `.npy`: NumPy arrays with `f4`/`f8` dtype and 1D/2D shape.

Example CSV fixture:

```bash
python - <<'PY'
import csv

values = [0.0] * 40 + [5.0] * 40 + [-2.0] * 40
with open("/tmp/cpd_signal.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["value"])
    writer.writerows([[value] for value in values])
PY
```

## JSON Output Modes

All subcommands emit JSON to stdout by default or to `--output <path>` when provided.

- Default mode is pretty JSON.
- `--pretty-json` forces pretty JSON.
- `--compact-json` emits single-line compact JSON.

Examples:

```bash
cd cpd
cargo run -p cpd-cli -- detect --input /tmp/cpd_signal.csv --pretty-json
cargo run -p cpd-cli -- detect --input /tmp/cpd_signal.csv --compact-json
cargo run -p cpd-cli -- detect --input /tmp/cpd_signal.csv --output /tmp/detect.json
```

## `detect`

Run offline detection directly from CLI flags.

```bash
cd cpd
cargo run -p cpd-cli -- detect \
  --algorithm pelt \
  --cost l2 \
  --penalty bic \
  --input /tmp/cpd_signal.csv
```

Known-`k` mode:

```bash
cd cpd
cargo run -p cpd-cli -- detect \
  --algorithm segneigh \
  --cost l2 \
  --k 2 \
  --input /tmp/cpd_signal.csv
```

## `run`

Execute a pipeline JSON spec against input data.

```bash
cat > /tmp/cpd_pipeline.json <<'JSON'
{
  "detector": {"kind": "pelt"},
  "cost": "l2",
  "constraints": {"min_segment_len": 1},
  "stopping": {"n_bkps": 2}
}
JSON

cd cpd
cargo run -p cpd-cli -- run \
  --pipeline /tmp/cpd_pipeline.json \
  --input /tmp/cpd_signal.csv
```

## `doctor`

Generate recommendation-ranked pipelines for an input series.

```bash
cd cpd
cargo run -p cpd-cli -- doctor \
  --input /tmp/cpd_signal.csv \
  --objective balanced \
  --min-confidence 0.2 \
  --output /tmp/doctor.json
```

## `eval`

Evaluate predictions against ground truth JSON.

Offline example:

```bash
cp cpd/tests/fixtures/migrations/result/offline_result.v1.json /tmp/predictions.json
cp cpd/tests/fixtures/migrations/result/offline_result.v1.json /tmp/ground_truth.json

cd cpd
cargo run -p cpd-cli -- eval \
  --predictions /tmp/predictions.json \
  --ground-truth /tmp/ground_truth.json \
  --tolerance 1
```

## JSON Workflow Tips

- Use `--output` for stable artifacts in automation or CI.
- Use `--compact-json` when piping to log processors or line-oriented tools.
- Use `--pretty-json` when reviewing output manually.
