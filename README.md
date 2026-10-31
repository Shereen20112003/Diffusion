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

### Local validation-only training investigation

The original pilot is preserved. The subsequent development script varies
training duration while keeping architecture, optimizer, noise schedule,
training stream, and sampling seed fixed:

```powershell
& .\.venv\Scripts\python.exe .\experiments\diagnose_training.py --output .\experiments\results\training_diagnostic
```

It compares checkpoints selected by validation noise error after 2,500 and
10,000 optimizer updates, for both datasets and both generators. It uses
manual landmarks and the primary embedded-MSE decoder only, to isolate
diffusion training from decoder/loss/reference changes. The decoder is
trained once per dataset and kept fixed. Parameter counts differ only where
the two-dimensional versus three-dimensional denoiser inputs/outputs require it.

All reported W2 values in this diagnostic are **development-validation
measurements**, using newly generated validation draws rather than loading
the pilot's test observations. Fixed validation noise supports comparable
noise-error profiles at every diffusion timestep. Generated-sample diagnostics
include distance to the clean moon arcs, or nearest Gaussian center and
nearest-center frequencies. Reference values from real validation data
are included; observations need not lie exactly on the clean geometry.
Thresholds (0.15 from moon arcs, 0.45 from Gaussian centers) are three times
the respective noise standard deviations and are descriptive, not acceptance
tests or theorem guarantees.

The equations were checked against the authors' DDPM implementation:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py
(`q_sample`, `q_posterior`, and `p_sample`). The implemented reverse mean is
the equivalent epsilon parameterization, with posterior variance and no noise
at the final step. Unlike an image implementation using values in [-1, 1],
these standardized point data are not clipped to that range.
The schedule is configured for this pilot, not presented as an exact
reproduction of the original DDPM image experiments. Squared noise error is
summed over coordinates rather than averaged, matching our manuscript
convention; it differs by a fixed dimensional factor.

A separate validation ablation enables four fixed spatial Fourier bands
alongside the original noisy coordinates with `--spatial-frequencies 4`.
Both denoisers get the same feature rule and hidden-layer width; the decoder,
training distribution, and diffusion schedule stay unchanged. This tests
whether spatial features help resolve small-scale structure. It adds input
parameters, so it is not a parameter-matched comparison against the original
plain-input architecture. Defaults remain unchanged and no prior run is
overwritten.

```powershell
& .\.venv\Scripts\python.exe .\experiments\diagnose_training.py --spatial-frequencies 4 --output .\experiments\results\spatial_diagnostic
```

For an interrupted diagnostic that saved all checkpoint samples and noise
profiles, `--render-only --output <existing-run-directory>` rebuilds its
comparison figures without retraining or fabricating a training-completion
marker. Subsequent changes are local until reviewed and approved for pushing.

### Validation investigation outcomes

`experiments\results\training_diagnostic` preserves the plain-input
2,500/10,000-update comparison. The command was interrupted after all eight
checkpoint measurements were saved; figures were recovered with
`--render-only`. Its `rendering.json` is not a normal training-completion
certificate.

`experiments\results\spatial_diagnostic` completed normally in about seven
minutes on two CPU threads. At 10,000 updates the spatial-feature models
visually recover eight separate clusters and much sharper moon arcs.
The following are development-validation diagnostics, not final test scores:

| Dataset / method | Plain-input mean distance | Spatial-feature mean distance | Real validation mean distance |
|---|---:|---:|---:|
| Two moons / direct | 0.1572 | 0.0532 | 0.0390 |
| Two moons / coordinate + MLP | 0.1539 | 0.0632 | 0.0390 |
| Eight Gaussians / direct | 0.2991 | 0.1903 | 0.1896 |
| Eight Gaussians / coordinate + MLP | 0.2910 | 0.2248 | 0.1896 |

Distance here means nearest clean arc or nearest Gaussian center, not W2.
At 10,000 updates the spatial-feature validation W2 values are 0.2608
(direct) and 0.1711 (coordinate) for moons, and 0.2988 (direct) and 0.3665
(coordinate) for eight Gaussians. In particular, improved geometry does not
necessarily reduce W2: mixture frequencies and finite-sample variation also
affect that score. The extra-geometry fractions remain above real-data
references, particularly for coordinate generation.

These results do not establish a universal advantage for spatial features,
an optimal architecture, or method superiority. The follow-up changes input
parameter counts as well as representation. Both competing methods receive
the same feature rule, but comparisons to the original plain-input model
are an architecture ablation, not an exactly parameter-matched trial.
Independent seeds, validation-based configuration selection, and a fresh
final evaluation set remain necessary. The next figures to inspect are
`comparison_updates_10000.png` (or `.pdf`) under each dataset in the
`spatial_diagnostic` directory.
