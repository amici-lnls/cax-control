# cax-control

Analysis and optimization notebooks for the CAX beamline simulation.

## Layout

- `scripts/` — shared, importable helpers used by the notebooks:
  - `knobspace_data.py`: discovers and loads the 81×81 TX/RY scan CSVs from `data/`, builds the `[TX, RY]` grids and holds the objective normalisation constants.
  - `objective_functions.py`: `f1…f8`, their delta-method uncertainties and minima tables.
  - `plotting.py`: uncertainty-aware highlight bands, observable/objective maps, optimizer-history panels, knob-space path overlays and `save_figure()`.
  - `cax_optimization.py`: `measure_f1()`, the seeded `CAXRCDSConfig`, the generic `CAXRCDS` optimizer and the four study presets.
  - `notebook_setup.py`: makes the repo importable from any Jupyter working directory and applies the shared plot style.
  - `caustic.py` / `image_statistics.py`: caustic fitting and beam-image analysis.
- `notebooks/optimization/` — one notebook per RCDS study:
  - `caxsim_optimization_TXRY.ipynb`
  - `caxsim_optimization_TYRX.ipynb`
  - `caxsim_optimization_TYRXRZ.ipynb`
  - `caxsim_optimization_5D.ipynb`
  - `CAX_optimization.ipynb` and the other existing notebooks are kept unchanged as historical reference.
- `notebooks/knobspace_exploration/` — scan acquisition and the original minima-analysis notebook (kept unchanged).
- `data/` — scan CSVs and saved RCDS histories (`data/rcds_runs/`).
- `figures/` — default output directory for notebook figures (created on demand).
- `tests/` — unit tests for the shared modules.

## Running a study

Open any `caxsim_optimization_*.ipynb` from within this checkout and run all
cells. The first cell bootstraps `sys.path`, so the notebook works whether
Jupyter was started at the repository root or in a subdirectory. Each study
is seeded via `make_*_config(seed=...)` and saves its histories to
`data/rcds_runs/` and its figures to `figures/`.

The optimization cells trace 100,000 rays per evaluation (500–600 evaluations),
so they are intentionally kept one notebook at a time.

## Tests

```bash
python -m pytest tests/test_knobspace_data.py tests/test_optimization.py
```

`tests/test_image_statistics.py` additionally requires the `shadowpy` runtime,
which is not present in every environment.
