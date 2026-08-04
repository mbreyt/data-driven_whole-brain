"""Central path configuration.

Every path used by the notebooks, scripts and Snakemake workflows is derived
here, so that nothing in this repository refers to a hard-coded location on a
particular machine.

Paths resolve relative to the repository root by default, and each can be
overridden with an environment variable. The overrides exist because the
simulation workflows run on an HPC system where the bulk data lives on scratch
storage rather than next to the code:

    WHOLE_BRAIN_DATA_DIR     bulk data root            (default: <repo>/data)
    WHOLE_BRAIN_SCRATCH_DIR  simulation output root    (default: $WHOLE_BRAIN_DATA_DIR/results)

Typical use::

    from scripts.python.config import DATA_DIR, RESULTS_DIR, ATLAS_LUT

The data itself is not in git; see the Data availability section of the README.
"""

import os
from pathlib import Path

# <repo>/scripts/python/config.py -> parents[2] is the repository root.
PROJECT_DIR = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    """Return $`var` as a Path, falling back to `default`."""
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


# --- Roots -----------------------------------------------------------------

DATA_DIR = _env_path("WHOLE_BRAIN_DATA_DIR", PROJECT_DIR / "data")

# Simulation output. On the cluster this points at scratch, e.g.
#   export WHOLE_BRAIN_SCRATCH_DIR=/p/scratch/vbt/martin/flax
SCRATCH_DIR = _env_path("WHOLE_BRAIN_SCRATCH_DIR", DATA_DIR / "results")

RESULTS_DIR = DATA_DIR / "results"

# --- Inputs ----------------------------------------------------------------

#: Schaefer 2018 atlas lookup table (100 parcels, 7 networks). Tracked in git.
ATLAS_LUT = DATA_DIR / "Schaefer2018_100Parcels_7Networks_order_LUT.txt"

#: Structural connectomes, one directory per subject, each with
#: `weights.txt` and `lengths.txt`. Not tracked in git; see README.
CONNECTOME_DIR = DATA_DIR / "scz-connectomes"

#: Parcellated resting-state fMRI timeseries for the analysed subject.
FMRI_DIR = DATA_DIR / "fmri"

#: Trained Flax/orbax model checkpoints.
TRAINED_MODELS_DIR = DATA_DIR / "trained_models"

# --- Results ---------------------------------------------------------------

#: Bifurcation continuation output. Tracked in git; produced by
#: notebooks/bifurcation-analysis.ipynb
BIFURCATION_DIR = RESULTS_DIR / "bifurcation_analysis"

#: Subject analysed in the paper.
SUBJECT = os.environ.get("WHOLE_BRAIN_SUBJECT", "sub-003")


def connectome(subject: str = SUBJECT):
    """Return `(weights, lengths)` file paths for `subject`."""
    d = CONNECTOME_DIR / subject
    return d / "weights.txt", d / "lengths.txt"


def require(path: Path) -> Path:
    """Return `path`, raising a pointed error if it is missing.

    Most inputs are distributed via Zenodo rather than git, so a bare
    FileNotFoundError deep inside a notebook is a confusing first experience.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}\n"
            "This file is not tracked in git. See the Data availability section "
            "of the README for how to obtain it, or set WHOLE_BRAIN_DATA_DIR to "
            "point at an existing copy."
        )
    return path
