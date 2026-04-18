# gauSTM

`gauSTM` is a Gaussian-driven STM simulator built around the Bardeen approximation.
It reads Gaussian formatted checkpoint (`.fchk`) files directly, reconstructs AO overlaps with PySCF, and produces MATLAB-style STM matrices that can be rendered as grayscale images.

The current codebase supports three workflows:
- single-state STM from one Gaussian reference state
- delta-SCF / Dyson-orbital STM between initial and final Gaussian states
- TDDFT excited-state STM where an excited final state is expanded into determinant components read from a Gaussian TDDFT log and accumulated at the Dyson level

## Highlights

- Gaussian `.fchk` parser for unrestricted and restricted references
- PySCF AO-overlap backend with Gaussian/FCHK AO ordering recovery
- batched CPU and GPU Bardeen scan backends
- Gaussian-tip STM scanning on 2D grids
- `s` and Cartesian probe components such as `p_x` via `STM.GaussTip`
- TDDFT state selection by either a single state (`STM.TDState`) or a state range (`STM.TDStateMin` / `STM.TDStateMax`)
- output metadata that records the excited-state index and excitation energy in the generated `.m` file headers

## Repository Layout

- `gaustm/`: package directory containing the implementation modules
- `stm_bardeen.py`: thin CLI wrapper for the main scan driver
- `plot_stm.py`: thin CLI wrapper for matrix rendering
- `run_batch_gpu.sh`: generic GPU batch runner for a directory of `.finp` jobs
- `examples/`: cleaned template `.finp` inputs
- `test/validation_summary.md`: compact summary of representative local validation runs
- `docs/implementation_notes_zh.md`: concise Chinese implementation note kept for reference

## Installation

A minimal CPU environment needs:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned requirements files below reflect the validated local environment used during development.

For GPU runs, install the optional GPU requirements in a CUDA-compatible environment:

```bash
pip install -r requirements-gpu.txt
```

## Core Usage

General form:

```bash
cd /path/to/job_dir
/path/to/gauSTM/.venv/bin/python \
  /path/to/gauSTM/stm_bardeen.py \
  input.finp \
  1 \
  gpu \
  > result.m \
  2> result.log
```

Positional arguments:
- `input.finp`: STM input file
- `n_cores`: host worker count
- `backend`: `legacy`, `cpu`, `batched`, `gpu`, or `auto`

Render one matrix from the resulting `.m` file:

```bash
/path/to/gauSTM/.venv/bin/python \
  /path/to/gauSTM/plot_stm.py \
  result.m \
  result.png \
  --matrix Ex001LMN1
```

## Input Rules

### Common keys

- `STM.Molfchk`: initial or reference Gaussian `.fchk`
- `STM.Gauss`: enable Gaussian tip mode
- `STM.Scan`, `STM.ScanA`, `STM.ScanB`, `STM.ScanC`: 2D scan settings
- `STM.RowBatch`: batched backend row grouping
- `%block STM.GaussTip`: tip exponent, requested Cartesian component, and starting point
- `%block STM.VScan`: scan step vectors

### Single-state MO projection

Use:
- `STM.Dyson F`
- `STM.TDA F`
- `STM.MOAlpha T` or `STM.MOBeta T`

### delta-SCF / Dyson route

Use:
- `STM.Dyson T`
- `STM.TDA F`
- `STM.Molfchk` and `STM.MolFinalfchk`

### TDDFT determinant-accumulation route

Use:
- `STM.Dyson T`
- `STM.TDA T`
- `STM.Molfchk`: initial state
- `STM.MolFinalfchk`: final reference state used to build excited determinants
- `STM.TDLog`: Gaussian TDDFT `.log` for the final reference state

State selection:
- set `STM.TDState N` to compute exactly one excited state
- omit `STM.TDState` and set `STM.TDStateMin` / `STM.TDStateMax` to compute a range
- older inputs that use `STM.TDState 0` are still accepted and interpreted as range mode

### Important TDDFT note

The TDDFT route depends on the Gaussian `.log` printing the excited-state composition lines, for example:

```text
Excited State   1: ...
   147A -> 148A   0.99982
```

Those determinant components are accumulated explicitly at the Dyson level.
If the log omits the configuration lines, the TDDFT mode cannot reconstruct the excited-state expansion.

## Probe Behavior

- `STM.GaussTip` now respects the requested Cartesian component directly.
- For example, `(1, 0, 0)` produces only the requested `p_x` channel instead of dumping the whole `p` shell.

## Current Scope and Limits

- 2D scan mode (`STM.Scan = 2`) is the implemented path
- relative `.fchk` and `.log` paths are resolved from the current working directory
- the current release has been validated on development cases whose AO basis shells extend through a pure `G` shell
- the current public validation set covers `s` and `p_x` Gaussian-tip components
- higher-angular-momentum AO shells or higher-order tip components are not part of the current validation set and should be treated as less-tested territory, even though parts of the code path have been generalized beyond the original `F`-shell-only implementation

## Validation

See [test/validation_summary.md](test/validation_summary.md) for a compact summary of representative local runs and timings.
The repository intentionally does not keep bulky generated `.m`, `.png`, and `.log` files under version control.

## Citation

If you use `gauSTM` in academic work, please cite the repository.
GitHub will automatically expose citation metadata from [CITATION.cff](CITATION.cff).
A short citation request is also recorded in [NOTICE](NOTICE).

At present, the associated manuscript has not yet been published.
Until that paper is available, please cite the software repository itself.
After the paper is published, users should cite both the repository and the paper.

## Notes for GitHub Upload

The repository citation metadata in `CITATION.cff` points to:
`https://github.com/chizhang-chem/gauSTM`
If needed, update the author metadata there before publishing.
