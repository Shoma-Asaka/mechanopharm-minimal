# mechanopharm-minimal

A minimal reference implementation of the mechanopharmacology framework developed in the accompanying manuscript.

This repository provides transparent, lightweight code for:
- the minimal two-state model,
- the minimal three-state adaptive-protection model,
- response-landscape visualization,
- extraction of experimentally relevant fingerprints from concentration–mechanics–time response data, and
- simple example workflows for figure reproduction and exploratory analysis.

The goal of this repository is to make the theoretical structure of the manuscript reproducible and easy to inspect. It is intended as a **minimal reference implementation**, not as a complete system-specific analysis platform.

## Scope

Included:
- two-state steady-state analysis,
- three-state adaptive-protection dynamics,
- illustrative response-landscape generation,
- simple fingerprint extraction routines,
- runnable example scripts,
- a notebook for figure-oriented exploration.

Not included:
- system-specific parameter calibration,
- a full experimental fitting framework,
- assay-specific preprocessing pipelines,
- GUI tools,
- claims of universal quantitative applicability across platforms.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Minimal usage

```bash
python examples/demo_minimal.py
```

This generates example outputs in `outputs/`.

## Minimal workflow for an experimental dataset

Assume you have endpoint measurements on a grid of concentration `c` and mechanical condition `m`, with optional time courses.

1. Arrange endpoint data as a matrix `E[c_index, m_index]` or as a tidy table.
2. Use `fingerprints.ec50_vs_m(...)` to quantify mechanical dose–response shifts.
3. Use `fingerprints.find_mechanical_optima(...)` to test for interior optima.
4. Use `fingerprints.peak_metrics_by_condition(...)` on time-course data to extract peak amplitude and peak time.
5. Compare the observed signature set against the minimal expectations:
   - two-state: shifted dose–response, possible sign reversal,
   - three-state: natural intermediate optimum, transient peak, delayed protection.

## Repository layout

```text
mechanopharm_minimal/
  __init__.py
  models.py
  fingerprints.py
  plotting.py
examples/
  demo_minimal.py
notebooks/
  reproduce_figures.ipynb
outputs/
  synthetic_endpoint.csv
  synthetic_timecourse.csv
  two_state_landscape.png
  three_state_transient.png
  summary.txt
tests/
  test_smoke.py
README.md
LICENSE
CITATION.cff
pyproject.toml
requirements.txt
.gitignore
```

## Relation to the manuscript

This code is intended to accompany the manuscript:

**A Thermodynamically Constrained Minimal Theory of Mechanopharmacology**

Suggested figure-to-code mapping:
- **Two-state analysis** → `mechanopharm_minimal/models.py`
- **Three-state adaptive-protection analysis** → `mechanopharm_minimal/models.py`
- **Response fingerprints / summary metrics** → `mechanopharm_minimal/fingerprints.py`
- **Example plots and demo runs** → `examples/demo_minimal.py`
- **Interactive exploration / figure reproduction** → `notebooks/reproduce_figures.ipynb`

## Citation

Before journal publication, please cite the specific archived GitHub/Zenodo release corresponding to the manuscript version you used.
After journal publication, please cite the final published article as the primary scholarly reference.

> If you use this repository in academic work before the paper is formally published, please cite the tagged release associated with the manuscript stage together with its Zenodo record. After journal publication, please cite the final published article as the primary scholarly reference.

A machine-readable citation file is included in `CITATION.cff`.

## Versioning

A tagged release should be created for each manuscript stage, for example:
- `v0.1.0` — submission version
- `v0.2.0` — revised version
- `v1.0.0` — accepted/publication version

For manuscript submission, cite the specific tagged release associated with that version of the paper.

## Important caution

This repository does **not** claim universal quantitative fitting across assay platforms. Parameter calibration, measurement noise models, identifiability, and assay-specific mechanics-to-descriptor mapping remain system dependent.

## License

This project is released under the MIT License. See `LICENSE`.

