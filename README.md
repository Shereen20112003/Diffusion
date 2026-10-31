# diffusion

## Planar experimental pilot

The implementation in `experiments\run_experiments.py` follows the planned
two-moons and eight-Gaussians experiments in
`overleaf\experiments\experiment_and_results.tex`.
It trains direct Euclidean diffusion and three-coordinate diffusion followed
by a small 3-to-2 MLP. Decoder comparisons are raw embedded MSE (primary),
embedded Huber, object-space MSE, and standardized embedded MSE.
Reference comparisons are predeclared manual peripheral directions, random
training subsets, and FPS. E2M and non-Euclidean data are later work.

### Setup (PowerShell, from this directory)

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r .\experiments\requirements.txt
& .\.venv\Scripts\python.exe -m unittest discover -s .\experiments -p "test_*.py"
```

The CPU build avoids downloading GPU dependencies on a machine without CUDA.
On a GPU Dev Box, use the appropriate official PyTorch build instead and pass
`--device cuda`. No GitHub Actions, cloud resources, or external datasets are
needed. The script uses two CPU threads by default.

### Run

```powershell
& .\.venv\Scripts\python.exe .\experiments\run_experiments.py --output .\experiments\results\pilot
```

The output directory must not already exist, to prevent overwriting a run.
Omit `--output` for a new UTC-timestamped directory.
Defaults cover both datasets, all three landmark rules, and all four decoder
losses, with one training seed, 4,096 training observations, 1,024 validation
observations, and 512 independent test and generated observations.
Each DDPM trains for 2,500 updates; each decoder candidate trains for 1,000.
Sampling uses 200 DDPM steps and a linear beta schedule from 0.0001 to 0.08.
The equal input standardization policy and diffusion budgets apply to both
generators. There is no projection or clipping of generated points.

The two moons use equally likely arcs `(cos(t), sin(t))` and
`(1-cos(t), 0.5-sin(t))`, with uniform `t` in `[0, pi]` and Gaussian noise
standard deviation 0.05. The eight equally likely Gaussian centers lie on a
radius-2 circle; each has covariance `0.15^2 I`. These unbounded-noise
experiments do not satisfy the paper's compact-support assumption.

The base is the training observation nearest its coordinate-wise median,
not an exact medoid. The manual rule selects distinct training extrema in
directions 0, 120, and 240 degrees, fixed before inspecting results.
References and coordinate scales are fitted on training observations only.
The raw Euclidean contrast is computed as the equivalent signed projection;
tests compare this with the paper's distance formula.

Decoder checkpoints are selected by validation raw embedded MSE, a common
criterion across objectives. Huber thresholds 0.1, 0.5, and 1.0 are compared
on validation data only. This gives Huber a larger total tuning budget,
which is recorded separately; each candidate has the same training budget.
Changing a landmark set refits both its coordinate DDPM and decoders.
The unchanged direct baseline is reused for the three landmark comparisons.
All decoders within a configuration receive identical generated coordinates.

### Outputs and interpretation

- `config.json`, `environment.txt`: settings, source hash, versions, and seeds.
- `metrics.csv`, `summary.csv`: exact empirical W2 in original 2D units,
  held-out reconstruction errors, and timings.
- Per-dataset/seed/landmark folders: three-panel `comparison_primary.png`
  and `.pdf`; `comparison_all_losses.png` and `.pdf`; references and learning
  curves; saved data, generated samples, and PyTorch model checkpoints.
- `completion.json`: only written after the complete run finishes.

W2 uses exact equal-weight assignment between equally sized sample sets,
followed by the square root. A real-real comparison gives finite-sample
context, not a strict lower bound. No test scores select losses or checkpoints.
Do not compare raw DDPM training-loss magnitudes across input dimensions.

One seed is a **pilot**, not evidence of superiority or equivalence.
Across-seed standard deviation is left empty rather than reported as zero
when only one seed is run. A later run can use `--seeds 0 1 2 3 4`.
Use `--datasets two_moons`, `--landmarks manual`, or `--losses embedded_mse`
for a smaller run. Results are ignored by Git but remain on disk.
Training does not modify the manuscript or automatically insert pilot plots
into the paper.

### First completed pilot

`experiments\results\pilot` contains the first one-seed run, completed in about
nine minutes on two CPU threads. Its manual-landmark primary comparison was:

| Dataset | Direct DDPM W2 | Coordinate DDPM + embedded-MSE MLP W2 | Real-real W2 |
|---|---:|---:|---:|
| Two moons | 0.2201 | 0.2460 | 0.1372 |
| Eight Gaussians | 0.5646 | 0.4486 | 0.4416 |

Both generators capture broad structure but do not yet reproduce the thin
moons or eight tight clusters. Small paired reconstruction errors do not imply
accurate generation. The primary decoder's held-out object squared error is
approximately 0.0000013 to 0.0000114 across reference configurations, while
the generated distributions remain visibly diffuse.

These are exploratory measurements, not evidence that one method is better.
The large real-real variation for the Gaussian mixture also cautions against
reading too much into one 512-point score. Since these pilot plots have been
inspected, later model-development decisions require a fresh final evaluation
set and independent training seeds for a confirmatory comparison.
