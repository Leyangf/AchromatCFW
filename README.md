# AchromatCFW

AchromatCFW provides a small set of utilities to analyse colour fringe width
(CFW) in optical systems.  Defocus information measured with Zemax OpticStudio
is combined with user–defined sensor responses and daylight spectra to
estimate the visible fringe width across focus.

## Repository layout

```
├── data/raw/            # Spectral data used in the examples
├── notebooks/           # Demonstration notebook
├── src/achromatcfw/     # Python package
│   ├── cfw.py           # Core CFW routines (Numba accelerated)
│   ├── spectrum.py      # Spectral I/O helpers
│   └── zemax.py         # Zemax integration and CLI
└── tests/               # Unit tests
```

## Installation

Create and activate the Conda environment and install the package in editable
mode:

```bash
conda env create -f environment.yml
conda activate masterthesis
pip install -e .
```

## Command line usage

The package includes a helper that connects to Zemax via the ZOS‑API to fetch
the longitudinal chromatic focal shift curve and compute CFW statistics:

```bash
python -m achromatcfw.zemax path/to/system.zmx \
    --defocus-range 500 --xrange 200 --F 8.0 --gamma 1.0
```

It prints the maximum and mean colour‑fringe width across the specified defocus
range.  Run with `-h` for the full list of options.

## Development

Run the unit tests with:

```bash
pytest -q
```

`notebooks/cfw_demo.ipynb` demonstrates the interactive API and continues to
work with the reorganised package structure.
