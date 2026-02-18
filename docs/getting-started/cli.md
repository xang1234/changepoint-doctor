# CLI Workflow

Use the `cpd` CLI for JSON-first workflows without writing Python glue code.

Repository CLI reference: [`cpd/docs/cli.md`](../../cpd/docs/cli.md).

## Build

From repository root:

```bash
cd cpd
cargo build -p cpd-cli --release
./target/release/cpd --help
```

## Install/import naming

> Install/import naming: install with `python -m pip install changepoint-doctor`, then import with `import cpd` in Python. Optional compatibility alias: `import changepoint_doctor as cpd`.

This page covers `cpd` (CLI). The callout above is included so CLI and Python onboarding stay aligned.

## End-to-end JSON flow

Create a sample CSV:

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

Run detect:

```bash
cd cpd
cargo run -p cpd-cli -- detect --input /tmp/cpd_signal.csv --output /tmp/detect.json
```

Generate doctor recommendations:

```bash
cd cpd
cargo run -p cpd-cli -- doctor --input /tmp/cpd_signal.csv --output /tmp/doctor.json
```

Evaluate predictions:

```bash
cp cpd/tests/fixtures/migrations/result/offline_result.v1.json /tmp/predictions.json
cp cpd/tests/fixtures/migrations/result/offline_result.v1.json /tmp/ground_truth.json
cd cpd
cargo run -p cpd-cli -- eval --predictions /tmp/predictions.json --ground-truth /tmp/ground_truth.json
```

## Output mode controls

- Default output is pretty JSON.
- `--pretty-json` forces pretty JSON.
- `--compact-json` emits compact one-line JSON.
- `--output <path>` writes JSON to a file instead of stdout.
