# Conditional Metric-Star Pilot: Implementation and Experiment Guide

Date: 2026-11-03.

**Experiment identifier:** `conditional_metric_star_pilot`.

This implements the supplied 2026-11-02 conditional metric-star handoff. It is
**not** the unconditional two-moons, eight-Gaussians, or earlier fixed-star study.
The approved execution scope is first-stage correctness tests and one bounded,
single-seed smoke experiment. Second-stage ablations are implemented as opt-in
code, but their experiment runs have not been launched.

The supplied handoff refers to a revised manuscript and companion diagnostics
that were not supplied at their cited paths. The implementation below follows
the explicit mathematical specification in the handoff; it does not claim to
reproduce those unavailable scripts or verify the exact manuscript appendices.

## 1. Questions and staged experiments

| Track | Purpose | Identifier / implementation status |
|---|---|---|
| Geometry and paired oracle reconstruction | Check the encoder, exact decoder, and atom preservation before training | `held_out_reconstruction.csv`; executed in smoke |
| Conditional oracle coordinates | Isolate reconstruction from generator error, using true encoded responses | `oracle_coordinate`; executed in smoke |
| Gaussian-perturbed coordinates | Check controlled coordinate robustness, not DDPM accuracy | `perturbed_coordinate`; executed in smoke |
| Learned conditional generation | Compare conditional coordinate DDPM, projected direct DDPM, empirical resampling, and X-ignored control | `learned_generator`; executed at smoke budgets only |
| Known-law threshold bias | Quantify atoms induced by thresholding fresh true-source draws | `known_law_threshold_diagnostic`; executed in smoke |
| Second-stage ablations | Learned residual weights, projected MLP, larger A, alternative base, label-aware comparator | Implemented behind `--second-stage`; not launched |
| Oracle denoiser | Integrate over the three arms and central atom; check quadrature refinement | Not implemented; optional separate track |
| Appendix-J alignment | Check the specific restricted theoretical sampler against practical DDPM | Not implemented; exact revised source required |

No factorial sweep over all datasets, losses, architectures, and landmark rules
has been added.

## 2. Response geometry and named scenarios

Let `X ~ Uniform[0,1]`. The response is a point on a fixed unit tripod:
`Y=(J,R)`, with `(0,0)` the center and `J=1,2,3`, `0<R<=1` the arms.
Saved data and generated samples include explicit `J` and `R` arrays. Internally,
the existing star geometry is reused through a sparse three-vector, with radius
in the corresponding arm coordinate and zeros elsewhere.

The intrinsic distance is `|R-S|` on the same arm and `R+S` on different arms.
The planar drawing is for display and the ambient comparator only.

$$
\pi_j(x)=
\frac{\exp\{1.5\cos(2\pi(x-(j-1)/3))\}}
{\sum_{k=1}^3 \exp\{1.5\cos(2\pi(x-(k-1)/3))\}}.
$$

First draw the center with probability `p0(x)`. Otherwise select the arm using
`pi(x)` and independently draw the radius from the mixture below.

| Exact scenario name | Center probability | Noncentral radius law |
|---|---|---|
| `S0_no_atom` | 0 | 0.25 Uniform(0,1) + 0.75 Beta(2+2x,4-2x) |
| `S1_varying_atom` | 0.05 + 0.30 sin^2(pi x) | Same as S0 |
| `S2_near_vertex_no_atom` | 0 | 0.25 Uniform(0,1) + 0.75 Beta(1/2,2) |

All arms have positive probability, the uniform component provides full support,
and none of the laws has endpoint atoms. S0/S1 have noncentral near-center mass
of order `t`, while S2 has order `sqrt(t)`.

For S2, `P(0<R<=0.01)=0.114625` exactly. Thus thresholding even perfect source
draws at 0.01 creates an 11.4625% false central atom. This is a threshold-bias
diagnostic, not evidence that a fitted generator failed.

## 3. Reference encoder and exact decoder

The **primary** base `o` is the squared-intrinsic-distance sample medoid of an
independent response reference sample. Ties use the first reference-array index.
The three encoding landmarks `A` are selected by FPS starting from `o`,
excluding all zero-distance duplicates and previously selected metric locations.

The reconstruction set `B` is the **entire reference sample**, including every
repeated center observation. It is not the old runner's 64-object FPS subset.
The reference sample, landmarks, reconstruction collection, and preprocessing
remain fixed for every value of X. Only the DDPM receives X.

$$
E_j^{\rm raw}(y)=
\frac{d(y,o)^2+d(o,a_j)^2-d(y,a_j)^2}{2d(o,a_j)},\qquad
E_j(y)=\frac{E_j^{\rm raw}(y)-\mu_j}{S_j}.
$$

Means and sample standard deviations are fit on reference responses only.
Scales are clipped to `[0.25,4]`. The saved reference JSON reports the actual
base and landmarks, scales, minimum base-to-landmark distance, center
multiplicity, and certified coordinate radius.

The controlled diagnostic encoder instead uses the center, one endpoint per
arm, zero centering, and `S=sqrt(3) I`:

$$
E((j,r))=r(2e_j-\mathbf{1})/\sqrt{3},\qquad
\sqrt{2/3}\,d(y,y')\leq\|E(y)-E(y')\|\leq d(y,y').
$$

It is **explicitly diagnostic**, not silently substituted for the primary
sample-based encoder. Since each contrast is affine on each arm, the maximum
encoded norm at the center and endpoints certifies a support radius. A maximum
over observed training coordinates is not substituted for that bound.

Gaussian decoder weights are proportional to
`exp(-||z-E(b)||^2/(2h^2))`. The scale `h_scale` is the median positive pairwise
encoded distance within B. All four ratios `h/h_scale` in
`{0.05,0.10,0.20,0.40}` are evaluated. The oracle-diagnostic prespecification is
0.10; the primary trained method's ratio is chosen using held-out paired
validation, not the test conditional law.

For weights `w_l`, let `T_j=sum_{b_l on arm j} w_l r_l`,
`T=sum_l w_l r_l`, and `g_j=2T_j-T`. The exact weighted Frechet mean is
`(j,g_j)` if any `g_j>0`, otherwise the center. At most one is positive.
No iterative barycenter solver is used.

Geometry and decoding use float64. **A positive radius, however small, remains
noncentral in the raw output.** Exact-zero and near-zero positive sign margins
are recorded separately. No tolerance-based center snapping is hidden in the
raw decoder. Gaussian localization diagnostics are also saved, including the
bounded-residual adjustment for the optional learned-weight decoder.

## 4. Correct method names

| Method identifier | Meaning |
|---|---|
| `conditional_coordinate_ddpm_gaussian_frechet` | Primary: DDPM on fixed reference coordinates, conditioned on X, followed by Gaussian Frechet reconstruction |
| `conditional_planar_ddpm_segment_projection` | Conditional 2D DDPM, explicitly projected onto the nearest known planar arm segment; scored intrinsically |
| `conditional_empirical_gaussian_x_resampling` | Resample paired training responses with Gaussian weights in X; tune X bandwidth on validation |
| `x_ignored_coordinate_ddpm_gaussian_frechet` | Independently fitted control with X replaced by a constant |
| `real_real_mc_reference` | Two independent true samples at each condition; Monte Carlo reference, not a lower bound |
| `oracle_primary_sample_medoid_gaussian_frechet` | True encoded coordinates, or their explicitly labeled perturbations, using the primary sample-based encoder |
| `oracle_controlled_center_endpoints_gaussian_frechet` | Controlled diagnostic geometry, not a fitted generator |
| `true_source_threshold_bias_diagnostic` | Fresh source samples subjected to the threshold grid; never the OT ground truth |

Both diffusion comparators use the same optimizer-update and reverse-step
budgets, posterior reverse variances, width, batch size, and checkpoint schedule.
The existing spatial-feature denoiser is reused with a scalar X input; its
response outputs predict Gaussian noise. This is a practical epsilon-DDPM,
not the bounded-clean-prediction sampler in Appendix J.

## 5. Splits, streams, and budgets

| Quantity | Default pilot proposal | Executed smoke |
|---|---:|---:|
| Independent master seeds | 100,101,102,103,104 | 100 |
| Reference pairs | 512 | 48 |
| Generator-training pairs | 4096 | 128 |
| Validation pairs | 1024 | 48 |
| Validation conditions used | 256 | 8 |
| Independent generated draws per validation condition | 128 | 16 |
| Midpoint test conditions | 21 | 3 |
| Samples per law for each exact OT comparison | 512 | 32 |
| Probability samples per condition and evaluation draw | 4096 | 64 |
| Independent evaluation draws after fitting | 3 | 1 |
| Optimizer updates per diffusion model | 10000 | 40 |
| Checkpoint interval | 1000 | 20 |
| Reverse diffusion steps | 1000 | 12 |
| Batch size | 256 | 32 |
| Hidden width | 128 | 32 |

Training-update counts and reverse steps are distinct. Default pilot budgets
are available as configuration, **not claims of completed five-seed training**.
The smoke overrides are saved in each configuration.

Random streams are role-based and independent across reference, training,
validation, fitting, validation sampling, test truth, generator sampling, and
oracle perturbations. Within fitting, initialization and optimizer sampling use
separate streams. Every seed trains its own network; repeated evaluation draws
are not counted as additional training runs.

Test conditions are `(l-1/2)/L`. No exact test conditions are deliberately
inserted into training. Increasing training size to 1024 or 8192 is exposed
through `--train-size`, but should be executed only after correctness and
conditioning have been established. The smoke does not establish conditioning.

## 6. Validation without test-law access

Checkpoint and decoder bandwidth selection use generated event probabilities
against independently held-out paired observations. The finite event set has:

- Four labels: center and the three arm components, combining each endpoint
  with its arm.
- Twenty-seven events `{J=j,0<R<=t}`, for three arms and
  `t=0.1,0.2,...,0.9`.

The criterion is the mean sum of Brier losses for these 31 events. All methods
use the same held-out pairs, event grid, and Monte Carlo count. Decoder
candidates reuse the same validation coordinates. X-resampling has four
bandwidth candidates and no diffusion training; comparable tuning effort does
not mean equal total compute.

This is a proper score for the stated finite event set, not a validation
certificate for the entire conditional law. The analytical test law is used
only after fitting and selection.

## 7. Raw and full-vertex threshold sensitivity

Use the full vertex set `{center,three endpoints}` and the fixed thresholds
`{0,0.0025,0.005,0.01,0.02,0.05}`. Move outputs at intrinsic distance at most
`a` from a vertex to that vertex. Zero means unchanged raw output.

Every graph-valued learned comparator uses the same postprocessing. All
thresholds and decoder bandwidths for a coordinate generator reuse identical
generated coordinates. Oracle perturbations use shared source draws and
Gaussian noise across bandwidths and perturbation scales `0,0.02,0.10`.

The ground-truth OT samples are **never thresholded**. No threshold is chosen
using test performance. Outputs include all six thresholds, not just the
best-looking one. The center-only alternative is not run.

The seven categories are center, three open arms, and three endpoints. Their
probabilities, analytic truths, and pointwise binomial Monte Carlo standard
errors are saved. Endpoint atoms remain visible rather than disappearing into
the arm-component totals.

## 8. Evaluation, uncertainty, and runtime

The primary conditional metric is:

$$
\left(\frac1L\sum_{\ell=1}^L
W_2(\widehat P_{x_\ell},P_{x_\ell};d)^2\right)^{1/2}.
$$

Each W2 uses exact equal-weight assignment with **squared intrinsic distance**
as cost, then takes the square root of the mean assigned cost.

Other outputs include per-X W2, integrated seven-category TV, absolute and
signed central-atom error, false endpoint mass, arm-component probabilities,
joint arm/radius CDFs conditional on X, validity, paired reconstruction error,
Gaussian localization, and sign margins. The CDF CSV entries are explicitly
`P(J=j,0<R<=t | X=x)`, not CDFs normalized within a selected arm.

Evaluation draws are averaged within a training seed before between-seed
uncertainty is reported. Integrated W2 is RMS across conditions, not a mean
of per-X W2. Selected-pipeline summaries pool the validation-selected bandwidth
within each seed even when different seeds select different bandwidths.
Full bandwidth sensitivity remains available separately.

Single-seed between-training-run SDs and confidence intervals are undefined
and saved as empty values, not zero. Probability Monte Carlo standard errors
are not called training uncertainty. The real-real reference is neither a
lower bound nor a bias correction and is never subtracted from model error.

Fitting (including selection), optimizer-only, validation, sampling, decoding,
and planar projection timings are recorded separately. Optimizer and validation
timings are components of fitting: do not add them a second time. End-to-end
wall time also includes OT, saving, diagnostics, and plotting.

## 9. Implemented second-stage ablations: not launched

Enable these only with `--second-stage`:

1. `conditional_coordinate_ddpm_bounded_residual_frechet`: add
   `2*tanh(network(z))` to Gaussian logits, keeping relative residual range at
   most four. The coordinate DDPM and coordinate draws are reused.
2. `coordinate_mlp_explicit_segment_projection`: a learned coordinate-to-planar
   MLP followed by explicit nearest-segment projection; it is **not** claimed
   to satisfy the weighted-Frechet theorem.
3. `conditional_coordinate_ddpm_gaussian_frechet__medoid_k6` and `__medoid_k12`:
   refit the coordinate DDPM with six or twelve distinct FPS landmarks.
4. `conditional_coordinate_ddpm_gaussian_frechet__frechet_base_k3`: replace the
   medoid by the empirical Frechet mean. Hold the reference sample, training
   data, B, and A fixed wherever admissible; record forced landmark changes.
   Refit preprocessing and DDPM. The equal-dimensional base comparison uses
   the same initialization/optimizer stream.
5. `conditional_categorical_radial_histogram`: a label-aware conditional neural
   comparator with an explicit center category and 20 uniform radial histogram
   cells per arm. It uses label-aware cross-entropy training and the same
   finite-event validation score. This is a transparent local baseline, not an
   external benchmark reproduction or an exact continuous radial density.

Every reference record includes whether medoid and empirical Frechet mean
coincide. A root CSV and completion statistic report the coincidence frequency
without artificially breaking a tie to manufacture a difference.

## 10. Run commands and artifact layout

From the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s .\experiments -p "test_*.py" -v

& .\.venv\Scripts\python.exe .\experiments\run_conditional_star.py `
  --smoke --max-seconds 480 `
  --output .\experiments\results\conditional_star_smoke_20261103
```

The following are **future-run commands, not executed results**. Choose a new
output folder and approve an appropriate runtime ceiling before running:

```powershell
# Five independent first-stage pilot training seeds; not run.
& .\.venv\Scripts\python.exe .\experiments\run_conditional_star.py `
  --max-seconds 28800 --output .\experiments\results\conditional_metric_star_five_seed

# Opt-in second-stage pilot ablations; not run.
& .\.venv\Scripts\python.exe .\experiments\run_conditional_star.py `
  --second-stage --max-seconds 28800 `
  --output .\experiments\results\conditional_metric_star_second_stage
```

The example ceiling is a limit, not a runtime prediction. Existing nonempty
output directories are refused. On failure or timeout, an explicit failure JSON
is saved and the error is re-raised; partial artifacts are not marked complete.
There is no optimizer-step resume implementation.

Root outputs:

- Configuration, environment, source hashes, progress, and completion records.
- `per_x_metrics.csv`: raw scenario/seed/draw/X/method/bandwidth/threshold scores.
- `integrated_metrics.csv`: bandwidth-specific summaries.
- `selected_integrated_metrics.csv`: complete validation-selected pipelines.
- `paired_comparisons.csv`: paired per-X W2/TV differences.
- `integrated_paired_comparisons.csv`: per-seed differences of integrated scores.
- `base_coincidences.csv`: honest medoid/Frechet-mean coincidence records.

Each scenario/seed folder additionally contains independent split arrays,
explicit `(J,R)` responses, model checkpoints, candidate-selection histories,
reference metadata, held-out reconstruction records, raw generated coordinates
and responses, untouched OT truth, localization/sign-margin diagnostics, a
conditioning diagnostic, runtime CSV, and six plots:
reference geometry, per-X probabilities, raw/threshold sensitivity, bandwidth
sensitivity, radius CDFs, and paired W2/TV comparisons.

Long method identifiers remain intact in CSVs and sample metadata. Sample and
DDPM-checkpoint filenames use a short readable prefix plus a stable hash to
avoid Windows path-length failures in deeply nested workspace directories.

## 11. Novelty, external paper, and claim boundaries

The newly supplied [Minimax Rates for Conditional Generative Learning under
Wasserstein Loss](https://arxiv.org/abs/2610.35490), arXiv:2610.35490v1,
was submitted on 2026-10-26. Its abstract was checked on 2026-11-03.
It studies minimax conditional-law estimation with Holder continuity and an
unrestricted covariate distribution; extensions to higher Wasserstein orders
require additional density/integrability conditions.

This is relevant statistical theory, not evidence that a conditional DDPM
extension itself is novel or that a tripod with a central atom satisfies its
higher-order assumptions. No official implementation or experimental
reproduction has been established from the abstract alone.

The proposed distinction remains the general metric-space formulation,
geometry-compatible Frechet reconstruction, recovery guarantees, and behavior
at nonsmooth branching/atomic responses. The pilot makes **no superiority,
significance, minimax-rate, or theorem-certification claim**.

Appendix-J alignment would separately require independent references, a known
support radius, bounded clean prediction during training and sampling,
posterior reverse variance, a positive stopping index, final ball projection,
recorded time probabilities, and no guidance. Those features would still not
certify the finite-class/optimization premise for an SGD network. A 1000-step
practical budget is not a numerical certificate from a conservative proof.

## Implementation

- [Geometry, laws, encoder, and decoder](../experiments/conditional_star.py)
- [Conditional models and optional comparators](../experiments/conditional_star_models.py)
- [Runner, evaluation, and artifacts](../experiments/run_conditional_star.py)
- [Focused tests](../experiments/test_conditional_star.py)
