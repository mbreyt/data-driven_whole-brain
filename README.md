# Data-driven mean-field within whole-brain models

Code and analysis accompanying the manuscript *"Data-driven mean-field within
whole-brain models"* (Breyton et al., under review).

We train a neural-network mean-field model of coupled QIF neurons, embed it in a
whole-brain network model, and use simulation-based inference (SBI) to recover
its parameters from simulated and empirical resting-state fMRI. This repository
reproduces the figures and provides the workflows that generate the underlying
simulations.

> **Status.** Manuscript under review. Citation details will be added on
> acceptance; see [CITATION.cff](CITATION.cff).

## Repository layout

```
scripts/
  python/
    config.py            Central path configuration (see "Configuration")
    ml_models.py         JAX/Flax models: MPR, QIF-MLP mean field, whole-brain TVB
    features.py          BOLD summary-feature extraction (two pipelines)
    neural_mass.py       Balloon–Windkessel BOLD forward model
    noise_generator.py   Colored-noise generator for QIF simulations
    qif_sim_utils.py     Spiking-QIF reference simulations (Brian2)
    train-mlp.py         Train the neural-network mean field (produces the checkpoint)
  snakes/
    QIF-sims.snake                   Spiking-QIF reference simulations
    sbi-sims.snake                   Whole-brain simulations for SBI training
    extract-features-sbi-sims.snake  BOLD feature extraction from simulations
notebooks/
    sbi-on-sims-training.ipynb   Train SBI posteriors on simulated data
    sbi-on-sims-figures.ipynb    SBI validation figures
    sbi-on-empirical-data.ipynb  SBI applied to the empirical subject
    bifurcation-figures.ipynb    Bifurcation figures (Python)
    bifurcation-analysis.ipynb   Bifurcation continuation (Julia)
    artifacts/                   Ground-truth arrays + trained SBI posteriors
data/                    Inputs and results (mostly distributed via Zenodo, see below)
```

## Installation

Python 3.12 is required.

```bash
python -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install -r requirements.txt
```

The Julia bifurcation notebook is optional and has a separate toolchain — see
[Bifurcation analysis](#bifurcation-analysis).

## Configuration

All paths are resolved by [scripts/python/config.py](scripts/python/config.py)
relative to the repository root, so nothing is tied to a specific machine.
Notebooks and Snakemake workflows import the `scripts` package (each notebook
does this in its first cell); **run the workflows from the repository root**:

```bash
snakemake -s scripts/snakes/sbi-sims.snake --cores 4
```

Three environment variables override the defaults — this is how the simulation
workflows target HPC scratch storage rather than the repository:

| Variable | Default | Purpose |
|---|---|---|
| `WHOLE_BRAIN_DATA_DIR` | `<repo>/data` | Root for all input/output data |
| `WHOLE_BRAIN_SCRATCH_DIR` | `<data>/results` | Root for simulation output (large) |
| `WHOLE_BRAIN_SUBJECT` | `sub-003` | Subject whose connectome is simulated |

```bash
export WHOLE_BRAIN_SCRATCH_DIR=/p/scratch/<project>/<user>
snakemake -s scripts/snakes/sbi-sims.snake --cores 8
```

## Data availability

The bulk simulation data (~40 GB) is archived on Zenodo:

**https://doi.org/10.5281/zenodo.19821612**

| File | Contents |
|---|---|
| `mlp_training_data.tar` | Training data for the neural-network mean field |
| `sbi_data.tar` | Feature vectors used for the empirical SBI |
| `sbi_sims.tar` | Whole-brain simulations |

Download and extract into `data/` (or into `$WHOLE_BRAIN_DATA_DIR`), e.g.:

```bash
tar -xf sbi_sims.tar -C data/results/
```

Small inputs needed to reproduce the figures are tracked directly in git, so the
figure notebooks run immediately after cloning:

- `data/Schaefer2018_100Parcels_7Networks_order_LUT.txt` — atlas lookup table
- `data/results/bifurcation_analysis/*.csv` — bifurcation continuation output
- `data/empirical_features_sub-003.npy` — anonymised empirical summary-feature
  vector for the analysed subject
- `notebooks/artifacts/` — ground-truth parameter arrays (`gt_params_*.npy`) and
  trained SBI posteriors (`posterior_*.pkl`) handed from the inference notebooks
  to the figure notebooks

### Human-subject data

The whole-brain model is run on a single subject's structural connectome
(`sub-003`) and validated against that subject's resting-state fMRI. To protect
participant privacy:

- Only the **anonymised summary-feature vector** used in the empirical figure is
  distributed (`data/empirical_features_sub-003.npy`): a two-number derived
  statistic (FCD skewness, mean BOLD zero-crossings), sufficient to reproduce
  the empirical inference figure.
- The **structural connectome and raw/parcellated fMRI** are available on
  reasonable request to the authors, subject to the governing ethics approval
  and data-sharing agreement.

The figure notebooks fall back to the shipped feature vector automatically, so
they reproduce without access to the restricted data. The simulation workflows
require the connectome; point `WHOLE_BRAIN_DATA_DIR` at a copy once obtained
(expected at `data/scz-connectomes/<subject>/{weights,lengths}.txt`).

## Reproducing the results

1. **Set up** the environment (above) and, for the simulation-based steps, fetch
   the Zenodo archives into `data/`.

2. **Simulations** (compute-intensive; designed for a GPU cluster). Skip these if
   you only want the figures — the required outputs are on Zenodo.
   ```bash
   snakemake -s scripts/snakes/QIF-sims.snake --cores N                  # spiking QIF reference
   snakemake -s scripts/snakes/sbi-sims.snake --cores N                  # whole-brain SBI simulations
   snakemake -s scripts/snakes/extract-features-sbi-sims.snake --cores N # BOLD features
   ```

3. **Inference and figures** — run the notebooks in order:
   - `sbi-on-sims-training.ipynb` — trains the SBI posteriors on simulated
     data (writes the `artifacts/gt_params_*.npy` / `artifacts/posterior_*.pkl`
     handoff files)
   - `sbi-on-sims-figures.ipynb` — SBI validation figures
   - `sbi-on-empirical-data.ipynb` — SBI applied to the empirical subject
   - `bifurcation-figures.ipynb` — bifurcation figures (reads the CSVs below)

### Feature extraction: two distinct pipelines

Both feature definitions live in
[scripts/python/features.py](scripts/python/features.py) so the workflows and
notebooks share one source of truth. They are intentionally different:

- **SBI on simulations** uses `get_features_sims` — a high-dimensional vector
  (FC/FCD blocks + per-network statistics).
- **SBI on the empirical subject** uses `get_features_data` — a reduced
  7-feature vector; the same extraction generated the `sbi_data` training set
  (stored feature arrays are `(N, 7)`). The inference selects features
  `[5, 6] = [FCD skewness, zero-crossing]`.

### Bifurcation analysis

The bifurcation curves in `data/results/bifurcation_analysis/` are tracked in
git, so Julia is **not** needed to redraw the figures — only to regenerate the
continuation data. Provenance of the CSVs, by model:

| Prefix | Model | Producer |
|---|---|---|
| `mlp_qif*` | trained neural-network mean field | `notebooks/bifurcation-analysis.ipynb` (this repo) |
| `mpr*` | Montbrió–Pazó–Roxin mean field | a separate MPR continuation notebook (not included) |
| `mlp_fre*` | firing-rate-equation mean field | provided; generator not included |

The `mlp_qif*` curves are produced with
[BifurcationKit.jl](https://github.com/bifurcation-kit/BifurcationKit.jl); the
final cell of `bifurcation-analysis.ipynb` regenerates every `mlp_qif_cusp_p*`
CSV consumed by the p-dependence cusp figure in a single parameter sweep. It
reads the trained weights from `data/results/mlp_qif_params.pkl`.

## License

Code is released under the [Apache License 2.0](LICENSE); data on Zenodo under
CC-BY-4.0.