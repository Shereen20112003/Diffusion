# Conditional Metric-Star Pilot: Smoke Validation and Completion Status

Date: 2026-11-03.

## Executive summary

**2026-11-04 decoder-only follow-up:** The
[new confusion-matrix audit](tripod_decoder_debug_meeting_20261104.md)
shows that at the original ratio 0.10, **all 13 true centers were lost and the
two decoded centers were false positives**. The aggregate percentages below
must not be interpreted as two preserved true centers. No DDPM was run in that
follow-up.

The supplied conditional metric-star handoff has been implemented as a separate
experiment, **`conditional_metric_star_pilot`**, with the correct scenario and
method names. A bounded first-stage smoke run completed for all three scenarios
at master seed 100. Second-stage ablations are implemented but their experiment
runs have **not** been launched, as requested.

**This is development validation, not a completed full-budget pilot, a
five-seed benchmark, or evidence that conditional-law recovery has succeeded.**
No significance, superiority, minimax-rate, or theorem-certification claim is
made.

The complete mathematical specification, method identifiers, validation
procedure, full proposed budgets, command lines, and claim boundaries are in the
[implementation and experiment guide](conditional_metric_star_pilot.md).

## Authoritative completed run

- [Final output directory](../experiments/results/conditional_star_smoke_20261103/)
- [Completion marker](../experiments/results/conditional_star_smoke_20261103/completion.json)
- [Exact configuration and source hashes](../experiments/results/conditional_star_smoke_20261103/config.json)
- [Raw per-X metrics](../experiments/results/conditional_star_smoke_20261103/per_x_metrics.csv)
- [Validation-selected integrated results](../experiments/results/conditional_star_smoke_20261103/selected_integrated_metrics.csv)
- [Full bandwidth sensitivity results](../experiments/results/conditional_star_smoke_20261103/integrated_metrics.csv)
- [Paired per-X W2/TV comparisons](../experiments/results/conditional_star_smoke_20261103/paired_comparisons.csv)
- [Paired integrated W2/TV comparisons](../experiments/results/conditional_star_smoke_20261103/integrated_paired_comparisons.csv)
- [Base coincidence records](../experiments/results/conditional_star_smoke_20261103/base_coincidences.csv)

The final run took **48.86 seconds** end to end. Regression tests and all smoke
attempts together remained well within the approved ten-minute execution
budget. Earlier development output directories are not the authoritative
final validation record. One intermediate validation attempt encountered the
Windows path-length limit; filenames were shortened while retaining full
method identifiers in CSVs and sample metadata, and the final run completed.
The interrupted attempt was preserved rather than mislabeled as complete.

## Executed smoke budgets

For each of `S0_no_atom`, `S1_varying_atom`, and
`S2_near_vertex_no_atom`:

- One independent master seed: **100**.
- 48 independent reference pairs; 128 generator-training pairs; 48 validation
  pairs.
- Eight held-out validation conditions with 16 generated samples each.
- Three test midpoints: **1/6, 1/2, 5/6**.
- 32 generated and 32 independent true samples for each intrinsic OT score.
- 64 generated samples per condition for probability diagnostics.
- One independent evaluation draw after fitting.
- 40 optimizer updates per diffusion model, checkpointed at updates 20 and 40.
- 12 reverse steps, batch size 32, hidden width 32.
- All four decoder bandwidth ratios, all six full-vertex thresholds, and
  coordinate perturbation scales 0, 0.02, and 0.10.

The default five-seed proposal remains 512 references, 4096 training pairs,
1024 validation pairs, 21 test conditions, 512 samples per OT law, 4096
probability samples, three evaluation draws, 10000 optimizer updates, and 1000
reverse steps. Those larger settings have **not** been executed.

## Raw-output learned-method results

Threshold is zero. Decoder bandwidths and X-resampling bandwidths were selected
using held-out paired validation, not the analytic test law. W2 is RMS across
the three test conditions. TV and absolute center error are averaged across
conditions. Numbers below are rounded to four decimal places.

| Scenario | Method | Integrated intrinsic W2 | Seven-category TV | Absolute center error |
|---|---|---:|---:|---:|
| S0: no atom | Conditional coordinate DDPM + Gaussian Frechet | 0.4291 | 0.3145 | 0.0000 |
| S0: no atom | X-ignored coordinate DDPM + Gaussian Frechet | 0.4893 | 0.2656 | 0.0000 |
| S0: no atom | Conditional planar DDPM + segment projection | 0.6749 | 0.7156 | 0.0000 |
| S0: no atom | Conditional empirical Gaussian-X resampling | 0.3232 | 0.0531 | 0.0000 |
| S1: varying atom | Conditional coordinate DDPM + Gaussian Frechet | 0.4966 | 0.4247 | 0.1948 |
| S1: varying atom | X-ignored coordinate DDPM + Gaussian Frechet | 0.3515 | 0.2906 | 0.1896 |
| S1: varying atom | Conditional planar DDPM + segment projection | 0.7453 | 0.7568 | 0.2000 |
| S1: varying atom | Conditional empirical Gaussian-X resampling | 0.3102 | 0.1808 | 0.0594 |
| S2: near center, no atom | Conditional coordinate DDPM + Gaussian Frechet | 0.2230 | 0.2656 | 0.0781 |
| S2: near center, no atom | X-ignored coordinate DDPM + Gaussian Frechet | 0.2656 | 0.3718 | 0.0625 |
| S2: near center, no atom | Conditional planar DDPM + segment projection | 0.6970 | 0.6906 | 0.0000 |
| S2: near center, no atom | Conditional empirical Gaussian-X resampling | 0.1965 | 0.1323 | 0.0000 |

These are extremely undertrained smoke networks. Empirical resampling has lower
observed W2 and TV in these smoke rows, but this is **not** a fair full-budget
superiority result.

The primary model does not improve both W2 and TV over the X-ignored model in
S0 or S1. It does so descriptively in S2, but the smoke is too small to establish
conditioning. Every scenario's saved diagnostic explicitly records
`conditioning_established=false`.

## Decoder bottleneck visible before generator interpretation

Using held-out paired true coordinates, the primary sample-medoid encoder, and
the prespecified oracle ratio `h/h_scale=0.10`:

| Scenario | Paired intrinsic reconstruction MSE | Observed true center fraction | Exact decoded center fraction |
|---|---:|---:|---:|
| S0 | 0.001424 | 0.0000 | 0.0000 |
| S1 | 0.000943 | 0.2708 | 0.0417 |
| S2 | 0.002393 | 0.0000 | 0.0417 |

These are empirical fractions among 48 validation responses, not population
conditional center probabilities. In S1, the **small reconstruction MSE does
not preserve the observed central-atom fraction**. S2 also exhibits false exact
central outputs in this paired diagnostic despite having no true center atom.

This supports investigating decoder atom preservation before increasing
generator complexity. It does not show that longer DDPM training would resolve
the atom discrepancy.

The controlled-encoder diagnostics, per-bandwidth results, and near-zero sign
margins are saved separately. A tiny positive decoded radius is never counted
as an exact center in the raw results.

For the S2 known law, thresholding at radius 0.01 creates an **11.4625% false
central atom even with perfect source samples**. This exact value is tested.
Threshold sensitivity must therefore be reported, not optimized on test data.

The medoid and empirical Frechet mean coincide in **one of the three smoke
reference samples**. This small-sample frequency is descriptive, not an
estimate of a general coincidence probability.

## Runtime split

Seconds, rounded to four decimals:

| Scenario | Fitting including validation | Learned-method sampling | All recorded decoder passes | Planar projection |
|---|---:|---:|---:|---:|
| S0 | 0.6594 | 0.0524 | 0.0362 | 0.0003 |
| S1 | 0.8082 | 0.0425 | 0.0414 | 0.0004 |
| S2 | 0.7077 | 0.0555 | 0.0408 | 0.0004 |

Decoder timings include learned-generator, oracle, and perturbed-coordinate
decoding. End-to-end wall time also includes exact OT, diagnostic calculations,
plotting, CSV/archive writing, and other setup. These phase columns are not a
complete partition of the 48.86-second wall time. Optimizer-only and validation
times are components of fitting and are also available in each runtime CSV.

## Shareable figures

### S0: no atom

- [Conditional probabilities](../experiments/results/conditional_star_smoke_20261103/S0_no_atom/seed_100/conditional_probabilities.png)
- [Raw versus thresholded sensitivity](../experiments/results/conditional_star_smoke_20261103/S0_no_atom/seed_100/raw_threshold_sensitivity.png)
- [Reference geometry](../experiments/results/conditional_star_smoke_20261103/S0_no_atom/seed_100/reference_geometry.png)

### S1: covariate-dependent atom

- [Conditional probabilities](../experiments/results/conditional_star_smoke_20261103/S1_varying_atom/seed_100/conditional_probabilities.png)
- [Paired W2/TV comparisons](../experiments/results/conditional_star_smoke_20261103/S1_varying_atom/seed_100/paired_w2_tv_comparisons.png)
- [Bandwidth sensitivity](../experiments/results/conditional_star_smoke_20261103/S1_varying_atom/seed_100/bandwidth_sensitivity.png)

### S2: substantial near-center mass without an atom

- [Raw versus thresholded sensitivity](../experiments/results/conditional_star_smoke_20261103/S2_near_vertex_no_atom/seed_100/raw_threshold_sensitivity.png)
- [Conditional probabilities](../experiments/results/conditional_star_smoke_20261103/S2_near_vertex_no_atom/seed_100/conditional_probabilities.png)
- [Conditional joint arm/radius CDFs](../experiments/results/conditional_star_smoke_20261103/S2_near_vertex_no_atom/seed_100/conditional_radius_cdfs.png)

Each scenario has six figures, including figures not individually linked above.
All plotted probabilities are point estimates; Monte Carlo standard errors are
available in the raw CSV, not represented as training-seed confidence bands.

## Verification

The complete existing experiment suite plus new conditional tests passed:
**42 tests**. Tests cover intrinsic geometry, all known laws, reproducibility,
independent stream identifiers, the S2 exact threshold bias, the exact Frechet
formula, small-positive-radius handling, reference multiplicities, scale
clipping, controlled-encoder distortion bounds, certified support radii,
projection, full-vertex categories, event scoring, sampling, checkpointing,
second-stage component gradients, mocked second-stage wiring, and aggregation.

A separate post-run verification confirmed:

- **1899** finite per-X metric rows across all three scenarios.
- **105** generated sample archives with valid shapes and consistent `(J,R)`.
- **18** figure files and **9** fitted DDPM checkpoints.
- Complete root and per-scenario markers.
- Matching recorded source hashes for the final implementation.
- No launched second-stage study.
- No undefined one-seed training variability reported as zero.

## Remaining work and decisions

| Item | Status |
|---|---|
| First-stage implementation and bounded smoke | Complete |
| Full-budget single-seed conditioning check | Not run |
| Five independent training seeds 100-104 | Not run |
| Training-size extensions 1024 and 8192 | Not run; conditioning prerequisite remains unmet |
| Learned residual decoder and projected MLP ablations | Implemented and component-tested; experiment runs not launched |
| k=6/12 and empirical Frechet-mean base comparisons | Implemented and mocked wiring-tested; experiment runs not launched |
| Label-aware categorical/radial comparator | Implemented and component-tested; experiment run not launched |
| Exact Appendix-J aligned sampler | Not implemented; cited revised manuscript needed |
| Oracle-denoiser quadrature | Optional; not implemented |
| New arXiv theory paper | Abstract checked and relevance recorded; full theorem comparison not claimed |
| Superiority or recovery claim | Not established |

The next substantive decision is how to address the oracle decoder's center
mass discrepancy and the threshold-bias tradeoff before expanding the trained
generator study.
