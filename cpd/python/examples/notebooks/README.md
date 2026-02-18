# Notebook Examples

This folder contains quickstart notebooks for common changepoint-doctor workflows:

- `01_offline_algorithms.ipynb`: compare offline detectors on noisy KPI-style data.
- `02_online_algorithms.ipynb`: run streaming detectors on service-latency data.
- `03_doctor_recommendations.ipynb`: generate doctor recommendations with live CLI execution and snapshot fallback.

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
python -m pip install --upgrade "changepoint-doctor[notebooks]==0.0.2"
jupyter lab
```

Then open `examples/notebooks/` in Jupyter and run notebooks top-to-bottom.

## Notes

- Sample data is generated in notebook cells with fixed random seeds for reproducibility.
- Notebook 3 tries live `cpd doctor` first, then falls back to `data/doctor_recommendations_snapshot.json` if CLI execution is unavailable.
