# Notebook Examples

This folder contains quickstart notebooks for common changepoint-doctor workflows:

- `01_offline_algorithms.ipynb`: compare offline detectors on noisy KPI-style data.
- `02_online_algorithms.ipynb`: run streaming detectors on service-latency data.
- `03_doctor_recommendations.ipynb`: generate Python-first doctor recommendations with `cpd.doctor(...)`, run the top pipeline immediately, and optionally export a CLI JSON artifact.

## Repo Layout

Expected clone layout (from repository root):

```text
changepoint-doctor/
  cpd/
    python/
      examples/
        notebooks/
```

## Launch From A Fresh Clone

Run these commands from repository root:

```bash
cd cpd/python
python -m pip install --upgrade pip
python -m pip install --upgrade "changepoint-doctor[notebooks]==0.0.3"
jupyter lab
```

Then open `examples/notebooks/` in Jupyter and run notebooks top-to-bottom.

## Notes

- Sample data is generated in notebook cells with fixed random seeds for reproducibility.
- Notebook 3 is Python-first and uses `cpd.doctor(...)` for executable offline recommendations; it also shows how to export a matching CLI JSON artifact when you need a file-backed report.
